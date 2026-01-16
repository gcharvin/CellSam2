# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from sam2.modeling.sam2_utils import MLP, LayerNorm2d, compute_iou


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        pred_div_scores: bool = False,
        pred_div_scores_mlp: bool = False,
        pred_iou_thresh: float = None,
        obj_score_thresh: float = None,
        div_obj_score_thresh: float = None,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)

        self.pred_div_scores = pred_div_scores
        if self.pred_div_scores:
            self.div_score_token = nn.Embedding(1, transformer_dim)

        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        if self.pred_div_scores:
            self.pred_div_score_head = nn.Linear(transformer_dim, 1)
            if pred_div_scores_mlp:
                self.pred_div_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        self.pred_iou_thresh = pred_iou_thresh
        self.obj_score_thresh = obj_score_thresh
        self.div_obj_score_thresh = div_obj_score_thresh
        self._debug_div = self._env_flag("SAM2_DEBUG_DIV")
        self._debug_div_eval = self._env_flag("SAM2_DEBUG_DIV_EVAL")
        self._debug_div_freq = int(os.environ.get("SAM2_DEBUG_DIV_FREQ", "50"))
        self._debug_div_step = 0
        self._debug_div_tb_dir = os.environ.get("SAM2_DEBUG_DIV_TB_DIR")
        self._debug_div_writer = None
        self._debug_div_rank = int(os.environ.get("RANK", "0"))
        self._debug_div_log_now = False

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        is_dividing: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          repeat_image (bool): whether to repeat the image embeddings for each prompt
          high_res_features (Optional[List[torch.Tensor]]): optional high resolution features
          is_dividing (Optional[torch.Tensor]): optional tensor indicating dividing cells for training
          gt_masks (Optional[torch.Tensor]): ground truth masks for training

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
          torch.Tensor: batched object score logits
          torch.Tensor: batched division score logits
          torch.Tensor: batched post-split object score logits
          torch.Tensor: batched is_dividing
        """
        masks, iou_pred, mask_tokens_out, object_score_logits, div_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        is_dividing_provided = is_dividing is not None
        debug_log_now = self._start_div_debug() if (self.training or self._debug_div_eval) else False
        self._debug_div_log_now = debug_log_now

        if is_dividing is not None and is_dividing.device != div_score_logits.device:
            is_dividing = is_dividing.to(div_score_logits.device)

        # Determine which cells are dividing
        if is_dividing is None:
            is_dividing = (
                (div_score_logits[:, 0] > self.div_obj_score_thresh)
                & (object_score_logits[:, 0] > self.obj_score_thresh)
                & (iou_pred[:, 1:3] > self.pred_iou_thresh).any(1)
            )
        
        # Ensure is_dividing is a flat boolean tensor
        is_dividing = is_dividing.view(-1)

        if debug_log_now:
            with torch.no_grad():
                div_scores = div_score_logits[:, 0].detach()
                obj_scores = object_score_logits[:, 0].detach()
                iou_pair = iou_pred[:, 1:3].detach()
                iou_max = iou_pair.max(1).values
                num_objects = is_dividing.numel()
                num_div_input = int(is_dividing.sum().item())

                pred_dividing = None
                gate_div = gate_obj = gate_iou = gate_all = None
                if (
                    self.div_obj_score_thresh is not None
                    and self.obj_score_thresh is not None
                    and self.pred_iou_thresh is not None
                ):
                    gate_div = div_scores > self.div_obj_score_thresh
                    gate_obj = obj_scores > self.obj_score_thresh
                    gate_iou = (iou_pair > self.pred_iou_thresh).any(1)
                    gate_all = gate_div & gate_obj & gate_iou
                    pred_dividing = gate_all

                num_div_pred = int(pred_dividing.sum().item()) if pred_dividing is not None else None
                if num_div_pred is None or num_div_pred == 0:
                    self._debug_div_log_now = False
                else:
                    num_div_mismatch = None
                    if is_dividing_provided and pred_dividing is not None:
                        num_div_mismatch = int((pred_dividing != is_dividing).sum().item())

                    scalars = {
                        "div_debug/num_objects": float(num_objects),
                        "div_debug/num_div_input": float(num_div_input),
                        "div_debug/num_div_pred": float(num_div_pred)
                        if num_div_pred is not None
                        else None,
                        "div_debug/num_div_mismatch": float(num_div_mismatch)
                        if num_div_mismatch is not None
                        else None,
                        "div_debug/gate_div": float(gate_div.sum().item())
                        if gate_div is not None
                        else None,
                        "div_debug/gate_obj": float(gate_obj.sum().item())
                        if gate_obj is not None
                        else None,
                        "div_debug/gate_iou": float(gate_iou.sum().item())
                        if gate_iou is not None
                        else None,
                        "div_debug/gate_all": float(gate_all.sum().item())
                        if gate_all is not None
                        else None,
                        "div_debug/div_score_mean": float(div_scores.mean().item()),
                        "div_debug/div_score_min": float(div_scores.min().item()),
                        "div_debug/div_score_max": float(div_scores.max().item()),
                        "div_debug/obj_score_mean": float(obj_scores.mean().item()),
                        "div_debug/iou_max_mean": float(iou_max.mean().item()),
                    }

                    if num_div_input > 0 and num_div_input < num_objects:
                        scalars["div_debug/div_score_mean_div"] = float(
                            div_scores[is_dividing].mean().item()
                        )
                        scalars["div_debug/div_score_mean_non_div"] = float(
                            div_scores[~is_dividing].mean().item()
                        )

                    if is_dividing_provided and gate_all is not None:
                        gt_div = is_dividing
                        gt_div_count = int(gt_div.sum().item())
                        if gt_div_count > 0:
                            gt_gate_pass = int((gt_div & gate_all).sum().item())
                            fn_mask = gt_div & ~gate_all
                            scalars["div_debug/gt_gate_recall"] = (
                                gt_gate_pass / gt_div_count
                            )
                            scalars["div_debug/gt_gate_pass"] = float(gt_gate_pass)
                            scalars["div_debug/gt_gate_fn"] = float(fn_mask.sum().item())
                            scalars["div_debug/gt_fn_fail_div"] = float(
                                (fn_mask & ~gate_div).sum().item()
                            )
                            scalars["div_debug/gt_fn_fail_obj"] = float(
                                (fn_mask & ~gate_obj).sum().item()
                            )
                            scalars["div_debug/gt_fn_fail_iou"] = float(
                                (fn_mask & ~gate_iou).sum().item()
                            )

                    source = "provided" if is_dividing_provided else "pred"
                    mode = "train" if self.training else "eval"
                    msg = (
                        "Div debug(step=%s, mode=%s, source=%s): n=%s div_in=%s div_pred=%s"
                        " gate(div/obj/iou/all)=%s/%s/%s/%s"
                    ) % (
                        self._debug_div_step,
                        mode,
                        source,
                        num_objects,
                        num_div_input,
                        num_div_pred,
                        int(gate_div.sum().item()) if gate_div is not None else -1,
                        int(gate_obj.sum().item()) if gate_obj is not None else -1,
                        int(gate_iou.sum().item()) if gate_iou is not None else -1,
                        int(gate_all.sum().item()) if gate_all is not None else -1,
                    )
                    self._log_div_debug(scalars, msg)
        
        # Create masks for dividing and non-dividing cells
        div_mask = is_dividing  # [B] bool
        no_div_mask = ~div_mask

        # Handle non-dividing cells (single mask per cell)
        pred_masks = masks[no_div_mask, 0:1]  # Keep dim for proper shape
        pred_ious = iou_pred[no_div_mask, 0:1]
        pred_tokens = mask_tokens_out[no_div_mask, 0:1]

        # Process dividing cells if any exist
        if div_mask.sum() > 0:
            # Always keep the primary mask as the mother, and select a single bud mask.
            pred_mother_masks = masks[div_mask][:, 0:1]  # [N, 1, H, W]
            pred_mother_ious = iou_pred[div_mask][:, 0:1]  # [N, 1]
            pred_mother_tokens = mask_tokens_out[div_mask][:, 0:1]  # [N, 1, C]

            if self.training and gt_masks is not None:
                pred_bud_masks, pred_bud_ious, pred_bud_tokens = self._match_bud_masks_to_gt(
                    gt_masks, masks, iou_pred, mask_tokens_out, div_mask
                )
            else:
                pred_bud_masks, pred_bud_ious, pred_bud_tokens = self._select_bud_masks(
                    masks, iou_pred, mask_tokens_out, div_mask
                )

            # Interleave mother + bud for each dividing object.
            pred_div_masks = torch.cat([pred_mother_masks, pred_bud_masks], dim=1)
            pred_div_ious = torch.cat([pred_mother_ious, pred_bud_ious], dim=1)
            pred_div_tokens = torch.cat([pred_mother_tokens, pred_bud_tokens], dim=1)

            # Reshape to have each mask as a separate item in batch
            pred_div_masks = pred_div_masks.flatten(0, 1).unsqueeze(1)  # [N*2, 1, H, W]
            pred_div_ious = pred_div_ious.flatten(0, 1).unsqueeze(1)  # [N*2, 1]
            pred_div_tokens = pred_div_tokens.flatten(0, 1).unsqueeze(1)  # [N*2, 1, C]

            # Combine results from non-dividing and dividing cells
            pred_masks = torch.cat([pred_masks, pred_div_masks], dim=0)
            pred_ious = torch.cat([pred_ious, pred_div_ious], dim=0)
            pred_tokens = torch.cat([pred_tokens, pred_div_tokens], dim=0)

        # Update object_score_logits to match the new output structure
        post_split_object_score_logits = None
        if object_score_logits is not None:
            # For non-dividing cells, keep single score
            pred_scores = object_score_logits[no_div_mask]
            
            # Handle dividing cells if any exist
            if div_mask.any():
                # For dividing cells, duplicate scores for mother + bud
                pred_div_scores = object_score_logits[div_mask].repeat_interleave(2, dim=0)
                # Combine scores
                post_split_object_score_logits = torch.cat([pred_scores, pred_div_scores], dim=0)
            else:
                # If no dividing cells, just use the non-dividing scores
                post_split_object_score_logits = pred_scores

        object_score_logits_dict = {"pre_div" : object_score_logits, "post_div" : post_split_object_score_logits}

        # Return all outputs
        return pred_masks, pred_ious, pred_tokens, object_score_logits_dict, div_score_logits, is_dividing

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )

        if self.pred_div_scores:
            output_tokens = torch.cat(
                [output_tokens, self.div_score_token.weight], dim=0
            )

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        if self.pred_div_scores:
            div_score_logits = self.pred_div_score_head(hs[:, -1, :])
        else:
            div_score_logits = -float('inf') * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits, div_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out

    def _match_bud_masks_to_gt(self, gt_masks, masks, iou_pred, mask_tokens_out, div_mask):
        """
        Match predicted bud masks to ground truth by selecting the best of masks 1/2.

        Assumes one bud per dividing object; GT buds are appended last in gt_masks.
        """
        assert self.training
        pred_bud_masks = masks[div_mask][:, 1:3]  # [N, 2, H, W]
        pred_bud_masks_sigmoid = pred_bud_masks.sigmoid()
        pred_bud_ious = iou_pred[div_mask][:, 1:3]
        pred_bud_tokens = mask_tokens_out[div_mask][:, 1:3]

        num_div = int(div_mask.sum().item())
        gts = gt_masks[-num_div:]  # [N, 1, H, W]
        gts = F.interpolate(
            gts.float(),
            size=pred_bud_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # [N, h, w]

        iou_0 = compute_iou(pred_bud_masks_sigmoid[:, 0], gts)
        iou_1 = compute_iou(pred_bud_masks_sigmoid[:, 1], gts)
        choose_second = iou_1 > iou_0

        if self._debug_div_log_now:
            with torch.no_grad():
                best_iou = torch.maximum(iou_0, iou_1)
                scalars = {
                    "div_debug/bud_iou0_mean": float(iou_0.mean().item()),
                    "div_debug/bud_iou1_mean": float(iou_1.mean().item()),
                    "div_debug/bud_best_iou_mean": float(best_iou.mean().item()),
                    "div_debug/bud_best_iou_lt_0.1_frac": float(
                        (best_iou < 0.1).float().mean().item()
                    ),
                    "div_debug/bud_choose_second_frac": float(choose_second.float().mean().item()),
                }
                self._log_div_debug(scalars, None)

        idx = choose_second.long()
        batch_idx = torch.arange(num_div, device=pred_bud_masks.device)
        pred_bud_masks = pred_bud_masks[batch_idx, idx].unsqueeze(1)
        pred_bud_ious = pred_bud_ious[batch_idx, idx].unsqueeze(1)
        pred_bud_tokens = pred_bud_tokens[batch_idx, idx].unsqueeze(1)

        return pred_bud_masks, pred_bud_ious, pred_bud_tokens

    def _select_bud_masks(self, masks, iou_pred, mask_tokens_out, div_mask):
        """Select a single bud mask in inference by minimizing overlap with the mother mask."""
        pred_bud_masks = masks[div_mask][:, 1:3]  # [N, 2, H, W]
        pred_bud_ious = iou_pred[div_mask][:, 1:3]
        pred_bud_tokens = mask_tokens_out[div_mask][:, 1:3]

        mother_masks = masks[div_mask][:, 0].sigmoid()
        bud0 = pred_bud_masks[:, 0].sigmoid()
        bud1 = pred_bud_masks[:, 1].sigmoid()

        overlap0 = compute_iou(bud0, mother_masks)
        overlap1 = compute_iou(bud1, mother_masks)
        choose_second = overlap1 < overlap0

        idx = choose_second.long()
        num_div = pred_bud_masks.shape[0]
        batch_idx = torch.arange(num_div, device=pred_bud_masks.device)
        pred_bud_masks = pred_bud_masks[batch_idx, idx].unsqueeze(1)
        pred_bud_ious = pred_bud_ious[batch_idx, idx].unsqueeze(1)
        pred_bud_tokens = pred_bud_tokens[batch_idx, idx].unsqueeze(1)

        return pred_bud_masks, pred_bud_ious, pred_bud_tokens

    @staticmethod
    def _env_flag(name: str) -> bool:
        value = os.environ.get(name, "").strip().lower()
        return value in ("1", "true", "yes", "y", "on")

    def _start_div_debug(self) -> bool:
        if not self._debug_div:
            return False
        self._debug_div_step += 1
        if self._debug_div_freq <= 0:
            return True
        return self._debug_div_step % self._debug_div_freq == 0

    def _get_debug_writer(self):
        if not self._debug_div_tb_dir or self._debug_div_rank != 0:
            return None
        if self._debug_div_writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self._debug_div_writer = SummaryWriter(log_dir=self._debug_div_tb_dir)
        return self._debug_div_writer

    def _log_div_debug(self, scalars, message):
        if self._debug_div_rank != 0:
            return
        if message:
            logging.info(message)
        writer = self._get_debug_writer()
        if writer is None:
            return
        for key, value in scalars.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.item()
            writer.add_scalar(key, value, self._debug_div_step)

import cv2
import logging
import numpy as np
import torch
from torchvision.ops import batched_nms
from tqdm import tqdm

from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.amg import MaskData,batched_mask_to_box
from sam2.utils.misc import load_video_frames, read_image
from sam2.utils.transforms import SAM2Transforms


logger = logging.getLogger(__name__)


class SAM2AutomaticCellTracker:
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: int = 32,
        points_per_batch: int = 32,
        obj_score_thresh: float = 0,
        pred_iou_thresh: float = 0.7,
        div_obj_score_thresh: float = 0,
        box_nms_thresh: float = 0.5,
        max_hole_area: int = 0,
        max_sprinkle_area: int = 0,
        mask_threshold: float = 0.0,
        segment: bool = False,
        use_heatmap: bool = False,
        min_mask_area: int = 10,
    ) -> None:
        """Using a SAM 2 model, generates and tracks masks for an entire video.
        Generates a grid of point prompts over the first frame, then tracks the detected cells
        throughout the video.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          points_per_side (int): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          segment (bool): Whether to segment or track.

        """
        self.model = model
        self.model.sam_mask_decoder.pred_iou_thresh = pred_iou_thresh
        self.model.sam_mask_decoder.obj_score_thresh = obj_score_thresh
        self.model.sam_mask_decoder.div_obj_score_thresh = div_obj_score_thresh
        self.device = model.device

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.obj_score_thresh = obj_score_thresh

        self.pred_iou_thresh = pred_iou_thresh
        self.obj_score_thresh = obj_score_thresh
        self.div_obj_score_thresh = div_obj_score_thresh
        self.segment = segment
        self.use_heatmap = use_heatmap
        # Bud masks can be small in the asymmetric setting, so keep a low area cutoff.
        self.min_mask_area = min_mask_area

        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

    @torch.inference_mode()
    def init_state(self,video_path,res_path,offload_video_to_cpu=False,offload_state_to_cpu=False,
                   async_loading_frames=False,max_frame_num_to_track=None,):
        """Initialize an inference state."""
        compute_device = self.model.device  # device of the model
        images, video_height, video_width, resized_image_size, padding = (
            load_video_frames(
                video_path=video_path,
                image_size=self.model.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=async_loading_frames,
                compute_device=compute_device,
                transforms=self._transforms,
            )
        )
        inference_state = {}
        inference_state["res_path"] = res_path
        inference_state["video_path"] = video_path
        inference_state["res_track"] = np.zeros((0, 4))
        inference_state["resized_image_size"] = resized_image_size
        inference_state["model_image_size"] = self.model.image_size
        inference_state["padding"] = padding
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["parent_ids"] = {}
        inference_state["max_frame_num_to_track"] = max_frame_num_to_track
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_ids"] = None
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        inference_state["memory_dict"] = {"mask_mem_pos_enc": None}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    def predict(self,video_path,res_path, offload_video_to_cpu=True,offload_state_to_cpu=False,max_frame_num_to_track=None,):
        """Predict and track cells throughout a video.

        Args:
            video_path: Path to the video file
            offload_video_to_cpu: Whether to offload video frames to CPU to save GPU memory
            offload_state_to_cpu: Whether to offload inference state to CPU

        Returns:
            Dictionary of tracking results with frame indices as keys

        """
        # Initialize the video state
        inference_state = self.init_state(
            video_path=video_path,
            res_path=res_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            max_frame_num_to_track=max_frame_num_to_track,
        )

        if not self.use_heatmap:
            # Generate points for the first frame or whole video if segment is True
            inference_state = self.generate_proportional_point_grid(inference_state)

        # Detect and track the detected cells through the video
        tracking_results = self.track_cells(inference_state)

        self.save_tracking_results(inference_state, tracking_results)

        return tracking_results

    def generate_proportional_point_grid(self, inference_state):
        """Generate a grid of (x, y) points with density proportional to the image size.

        Args:
            inference_state: The video inference state

        Returns:
            points: torch.Tensor of shape [N, 2] where each row is (x, y)
            labels: torch.Tensor of shape [N] with all 1s (foreground)

        """
        resized_H, resized_W = inference_state["resized_image_size"]
        scale_factor_y = resized_H / self.model.image_size  # preserve relative density
        scale_factor_x = resized_W / self.model.image_size  # preserve relative density

        # Estimate number of points in each dimension
        points_y = int(self.points_per_side * scale_factor_y)
        points_x = int(self.points_per_side * scale_factor_x)

        # Avoid zero division or very few points
        points_y = max(1, points_y)
        points_x = max(1, points_x)

        # Generate evenly spaced coordinates with proper offsets
        # When only 1 point in a dimension, place it in the center
        if points_y == 1:
            ys = np.array([resized_H // 2], dtype=int)
        else:
            # Add offset to avoid placing points at the very edge
            offset_y = resized_H / (2 * points_y)
            ys = np.linspace(offset_y, resized_H - 1 - offset_y, points_y, dtype=int)

        if points_x == 1:
            xs = np.array([resized_W // 2], dtype=int)
        else:
            # Add offset to avoid placing points at the very edge
            offset_x = resized_W / (2 * points_x)
            xs = np.linspace(offset_x, resized_W - 1 - offset_x, points_x, dtype=int)

        # Images are center padded during training
        xs += (self.model.image_size - inference_state["resized_image_size"][1]) // 2
        ys += (self.model.image_size - inference_state["resized_image_size"][0]) // 2

        points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)[:, None]

        # Convert points to tensor
        points = torch.tensor(points, device=self.device, dtype=torch.float32)  # [N, 2]

        # Create corresponding labels tensor (all foreground)
        labels = torch.ones((len(points), 1), dtype=torch.int, device=self.device)  # [N]

        # Save points and labels in inference_state for later reference
        inference_state["point_inputs"][0] = { "point_coords": points, "point_labels": labels,}

        if self.segment:
            for i in range(1, len(inference_state["images"])):
                inference_state["point_inputs"][i] = {"point_coords": points,"point_labels": labels,}

        return inference_state

    def track_cells(self, inference_state):
        """Track the detected cells throughout the video.

        Args:
            inference_state: The video inference state

        Returns:
            Dictionary of tracking results with frame indices as keys

        """
        tracking_results = []

        # Start propagation through the video
        for frame_idx, inference_state, track_mask in self.propagate_in_video(inference_state):
            tracking_results.append(track_mask)

            self.save_ctc(track_mask, frame_idx, inference_state)

        return tracking_results

    @torch.inference_mode()
    def propagate_in_video(self,inference_state,start_frame_idx=0,):
        """Propagate the input points across frames to track in the entire video."""
        num_frames = inference_state["num_frames"]
        max_frame_num_to_track = inference_state["max_frame_num_to_track"]

        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames

        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            if frame_idx == 0 or self.segment:
                if self.use_heatmap:
                    input_points, point_labels = self.get_input_points_from_heatmap(inference_state, frame_idx)
                    inference_state["point_inputs"][frame_idx] = {"point_coords": input_points,"point_labels": point_labels,}
                tracking_object_ids = None
                batch_size = inference_state["point_inputs"][frame_idx]["point_coords"].shape[0]
                is_init_cond_frame = True
            else:
                tracking_object_ids = inference_state["obj_ids"][frame_idx - 1]
                batch_size = len(tracking_object_ids)
                is_init_cond_frame = False

            if batch_size == 0:
                inference_state["obj_ids"][frame_idx] = torch.zeros(0, device=self.device, dtype=torch.int32)
                inference_state["parent_ids"][frame_idx] = torch.zeros(0, device=self.device, dtype=torch.int32)
                track_mask = np.zeros((inference_state["video_height"], inference_state["video_width"]),dtype=np.uint16,)
                yield frame_idx, inference_state, track_mask

            else:
                # Retrieve image features (only need to compute once for all objects)
                (_,_,current_vision_feats,current_vision_pos_embeds,feat_sizes,) =\
                    self._get_image_feature(inference_state, frame_idx, batch_size)

                # Run the core tracking step
                current_out, sam_outputs, high_res_features, pix_feat = (
                    self.model._track_step(
                        is_init_cond_frame=is_init_cond_frame,
                        current_vision_feats=current_vision_feats,
                        current_vision_pos_embeds=current_vision_pos_embeds,
                        feat_sizes=feat_sizes,
                        point_inputs=inference_state["point_inputs"].get(frame_idx, None),
                        mask_inputs=None,
                        num_frames=inference_state["num_frames"],
                        prev_sam_mask_logits=None,
                        tracking_object_ids=tracking_object_ids,
                        memory_dict=inference_state["memory_dict"],
                    )
                )

                # update cell tracks
                inference_state, track_mask = self.update_cell_tracks(
                    inference_state,
                    frame_idx,
                    sam_outputs,
                    current_out,
                    tracking_object_ids,
                )

                if not self.segment and frame_idx > 0 and self.use_heatmap:
                    input_points, point_labels = self.get_input_points_from_heatmap(inference_state, frame_idx)

                    if input_points.shape[0] > 0:
                        input_points_copy = input_points.clone()

                        pad_left, pad_right, pad_top, pad_bottom = inference_state["padding"]
                        input_points[:, 0, 0] -= pad_left
                        input_points[:, 0, 1] -= pad_top

                        input_points[:, 0, 0] = input_points[:, 0, 0] * (inference_state["video_width"]/ inference_state["resized_image_size"][1])
                        input_points[:, 0, 1] = input_points[:, 0, 1] * (inference_state["video_height"]/ inference_state["resized_image_size"][0])

                        # Convert to numpy and int
                        input_points_np = input_points.cpu().numpy().astype(np.int32)
                        track_cell_ids = track_mask[input_points_np[:, 0, 1], input_points_np[:, 0, 0]]

                        # Find indices where track_cell_ids is 0 (background)
                        background_point_indices = np.where(track_cell_ids == 0)[0]

                        if len(background_point_indices) > 0:
                            input_points = input_points_copy[background_point_indices]
                            point_labels = point_labels[background_point_indices]

                            inference_state["point_inputs"][frame_idx] = {"point_coords": input_points,"point_labels": point_labels,}
                            batch_size = input_points.shape[0]

                            # Retrieve image features (only need to compute once for all objects)
                            (_,_,current_vision_feats,current_vision_pos_embeds,feat_sizes,) = self._get_image_feature(inference_state, frame_idx, batch_size)
                            # Run the core tracking step
                            current_out, sam_outputs, high_res_features, pix_feat = (
                                self.model._track_step(
                                    is_init_cond_frame=True,
                                    current_vision_feats=current_vision_feats,
                                    current_vision_pos_embeds=current_vision_pos_embeds,
                                    feat_sizes=feat_sizes,
                                    point_inputs=inference_state["point_inputs"].get(frame_idx, None),
                                    mask_inputs=None,
                                    num_frames=inference_state["num_frames"],
                                    prev_sam_mask_logits=None,
                                    tracking_object_ids=None,
                                    memory_dict=inference_state["memory_dict"],
                                )
                            )

                            inference_state, detected_mask = self.update_cell_tracks(inference_state,frame_idx,sam_outputs,current_out,heatmap_input=True,)

                            if detected_mask.sum() > 0:
                                detected_cells = np.unique(detected_mask)
                                detected_cells = detected_cells[detected_cells != 0]

                                if len(inference_state["lost_obj_ids"][frame_idx]) > 0:
                                    lost_obj_ids = inference_state["lost_obj_ids"][frame_idx]
                                    lost_high_res_masks = inference_state["lost_high_res_masks"][frame_idx]

                                    # Calculate IoU between each detected cell and lost cell
                                    ious = np.zeros((len(detected_cells), len(lost_obj_ids)))
                                    for i, detected_id in enumerate(detected_cells):
                                        detected_mask_binary = (detected_mask == detected_id)
                                        for j, lost_id in enumerate(lost_obj_ids):
                                            intersection = np.logical_and(detected_mask_binary,lost_high_res_masks[j],).sum()
                                            union = np.logical_or(detected_mask_binary,lost_high_res_masks[j],).sum()
                                            ious[i, j] = (intersection / union if union > 0 else 0)

                                    # Find the lost cell with the highest IoU for each detected cell
                                    max_ious = np.max(ious, axis=1)
                                    lost_cell_indices = np.argmax(ious, axis=1)

                                    # Process each detected cell in order of IoU
                                    sorted_indices = np.argsort(-max_ious)  # Sort by descending IoU
                                    processed_lost_cells = set()
                                    cells_to_remove = (set())  # Track which cells to remove

                                    for idx in sorted_indices:
                                        if max_ious[idx] > 0:  # If cell has positive object score, it is assumed to be match if there is any overlap
                                            detected_cell_id = int(detected_cells[idx])
                                            lost_idx = lost_cell_indices[idx]
                                            lost_cell_id = int(lost_obj_ids[lost_idx])

                                            # Skip if this lost cell was already matched
                                            if lost_cell_id in processed_lost_cells:
                                                continue

                                            if (frame_idx - 1 in inference_state["memory_dict"][lost_cell_id]["frame_idx"]):
                                                detected_mask[detected_mask == detected_cell_id] = lost_cell_id
                                                inference_state["memory_dict"][lost_cell_id]["mask_mem_features"] = torch.cat(
                                                    (inference_state["memory_dict"][lost_cell_id]["mask_mem_features"],
                                                        inference_state["memory_dict"][detected_cell_id]["mask_mem_features"],
                                                    ), dim=0,)
                                                inference_state["memory_dict"][lost_cell_id]["obj_ptr"] = torch.cat(
                                                    (inference_state["memory_dict"][lost_cell_id]["obj_ptr"],
                                                        inference_state["memory_dict"][detected_cell_id]["obj_ptr"],
                                                    ),dim=0,)
                                                inference_state["memory_dict"][lost_cell_id]["frame_idx"].append(frame_idx)
                                                inference_state["obj_ids"][frame_idx][inference_state["obj_ids"][frame_idx]== detected_cell_id] = lost_cell_id

                                                cells_to_remove.add(detected_cell_id)  # Mark for removal
                                                processed_lost_cells.add(lost_cell_id)

                                                del inference_state["memory_dict"][detected_cell_id]

                                    # Remove the cells after processing all matches
                                    detected_cells = detected_cells[~np.isin(detected_cells, list(cells_to_remove))]

                                # Handle remaining detected cells
                                for detected_cell_id in detected_cells:
                                    # Get binary mask for current detected cell
                                    detected_mask_binary = (detected_mask == detected_cell_id)

                                    # Get all unique track IDs that overlap with this detected cell
                                    overlapping_track_ids = np.unique(track_mask[detected_mask_binary])
                                    overlapping_track_ids = overlapping_track_ids[overlapping_track_ids > 0]  # Remove background (0)

                                    if len(overlapping_track_ids) > 0:
                                        # Calculate IoU with each overlapping track
                                        best_iou = 0
                                        best_track_id = None

                                        for track_id in overlapping_track_ids:
                                            track_mask_binary = track_mask == track_id
                                            intersection = np.logical_and(detected_mask_binary, track_mask_binary).sum()
                                            union = np.logical_or(detected_mask_binary, track_mask_binary).sum()
                                            iou = (intersection / union if union > 0 else 0)

                                            if iou > best_iou:
                                                best_iou = iou
                                                best_track_id = track_id

                                        if best_iou > 0.05:  # If there's any overlap, assume it's the same cell
                                            # Update the detected mask to use the best matching track ID
                                            detected_mask[detected_mask_binary] = (best_track_id)
                                            inference_state["memory_dict"][best_track_id]["mask_mem_features"][-1] =\
                                                inference_state["memory_dict"][detected_cell_id]["mask_mem_features"][0]
                                            inference_state["memory_dict"][best_track_id]["obj_ptr"][-1] =\
                                                inference_state["memory_dict"][detected_cell_id]["obj_ptr"][0]
                                            inference_state["parent_ids"][frame_idx] = \
                                                inference_state["parent_ids"][frame_idx][inference_state["obj_ids"][frame_idx]!= detected_cell_id]

                                            inference_state["obj_ids"][frame_idx] = inference_state["obj_ids"][frame_idx][inference_state["obj_ids"][frame_idx]!= detected_cell_id]


                                            del inference_state["memory_dict"][detected_cell_id]

                                track_mask[(detected_mask > 0) * (track_mask == 0)] = detected_mask[(detected_mask > 0) * (track_mask == 0)]

                yield frame_idx, inference_state, track_mask

    @torch.inference_mode()
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            # Clone the image to avoid inference mode tensor issues
            image = image.clone()
            backbone_out = self.model.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self.model._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def update_cell_tracks(self,inference_state,
                           frame_idx,
                           sam_outputs,
                           current_out,
                           tracking_object_ids=None,
                           heatmap_input=False,):
        """Update the cell tracks based on the current output and SAM outputs."""
        obj_ids = tracking_object_ids

        # Unpack SAM outputs
        (ious,low_res_masks,high_res_masks,obj_ptr,object_score_logits_dict,div_score_logits,is_dividing,) = sam_outputs

        #
        save_masks = torch.zeros_like(high_res_masks)
        # Track division pairs whose mask order is swapped to preserve mother continuity.
        swapped_div_indices = set()

        # Keep only largest connected component for each mask
        for i in range(high_res_masks.shape[0]):
            mask = high_res_masks[i, 0].cpu().numpy()
            mask_binary = mask > self.mask_threshold
            if mask_binary.any():
                # Find connected components
                num_labels, labels = cv2.connectedComponents(mask_binary.astype(np.uint8))
                if num_labels > 1:  # If there are multiple components
                    # Find sizes of all components
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    # Get label of largest component (excluding background label 0)
                    largest_label = unique_labels[1:][np.argmax(counts[1:])]
                    # Keep only largest component
                    mask_binary = labels == largest_label
                    save_masks[i, 0][torch.from_numpy(mask_binary).to(high_res_masks.device)] =\
                        high_res_masks[i, 0][torch.from_numpy(mask_binary).to(high_res_masks.device)]
                else:
                    save_masks[i, 0][high_res_masks[i, 0] > self.mask_threshold] =\
                        high_res_masks[i, 0][high_res_masks[i, 0] > self.mask_threshold]

                # Get max scores across all masks for each pixel
        if obj_ids is not None and is_dividing is not None and is_dividing.any():
            # Swap division masks so the "mother" stays temporally consistent with its prior mask.
            prev_masks = inference_state.get("prev_masks")
            prev_obj_ids = inference_state.get("prev_obj_ids")
            if prev_masks is not None and prev_obj_ids is not None:
                prev_masks = prev_masks.to(high_res_masks.device)
                prev_obj_ids = prev_obj_ids.to(high_res_masks.device)
                non_div_count = int((~is_dividing).sum().item())
                div_indices = torch.nonzero(is_dividing, as_tuple=True)[0]

                def swap_mask_pair(idx0, idx1):
                    high_res_masks[[idx0, idx1]] = high_res_masks[[idx1, idx0]]
                    low_res_masks[[idx0, idx1]] = low_res_masks[[idx1, idx0]]
                    save_masks[[idx0, idx1]] = save_masks[[idx1, idx0]]
                    ious[[idx0, idx1]] = ious[[idx1, idx0]]
                    obj_ptr[[idx0, idx1]] = obj_ptr[[idx1, idx0]]
                    for key, value in object_score_logits_dict.items():
                        if value.shape[0] == high_res_masks.shape[0]:
                            object_score_logits_dict[key][[idx0, idx1]] = value[[idx1, idx0]]

                for div_counter, obj_idx in enumerate(div_indices.tolist()):
                    mother_id = obj_ids[obj_idx]
                    prev_idx = torch.nonzero(prev_obj_ids == mother_id, as_tuple=True)[0]
                    if prev_idx.numel() == 0:
                        continue
                    prev_mask = prev_masks[prev_idx[0], 0] > self.mask_threshold
                    mask_idx0 = non_div_count + 2 * div_counter
                    mask_idx1 = mask_idx0 + 1
                    cand0 = save_masks[mask_idx0, 0] > self.mask_threshold
                    cand1 = save_masks[mask_idx1, 0] > self.mask_threshold

                    inter0 = (cand0 & prev_mask).sum()
                    union0 = (cand0 | prev_mask).sum()
                    iou0 = inter0.float() / union0.float() if union0 > 0 else 0.0

                    inter1 = (cand1 & prev_mask).sum()
                    union1 = (cand1 | prev_mask).sum()
                    iou1 = inter1.float() / union1.float() if union1 > 0 else 0.0

                    # Swap if the second mask is more consistent with the previous mother.
                    if iou1 > iou0:
                        swap_mask_pair(mask_idx0, mask_idx1)
                        swapped_div_indices.add(int(obj_idx))
        argmax_scores = torch.max(save_masks[:, 0], dim=0)[1]  # shape: (H, W)

        # Count pixels for each mask index (excluding background)
        valid_mask = save_masks[:, 0].sum(0) > 0
        valid_indices = argmax_scores[valid_mask]
        max_mask_area = torch.bincount(valid_indices.flatten(), minlength=len(save_masks))

        keep_tokens = (
            (object_score_logits_dict["post_div"][:, 0] > self.obj_score_thresh)
            * (ious[:, 0] > self.pred_iou_thresh)
            * (max_mask_area > self.min_mask_area)
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=high_res_masks[keep_tokens].flatten(0, 1),
            save_masks=save_masks[keep_tokens].flatten(0, 1),
            iou_preds=ious[keep_tokens].flatten(0, 1),
            obj_scores=object_score_logits_dict["post_div"][keep_tokens].flatten(0, 1),
            obj_ptr=obj_ptr[keep_tokens],
        )

        data["boxes"] = batched_mask_to_box(data["save_masks"] > self.mask_threshold)
        data["conf"] = data["obj_scores"].sigmoid() * data["iou_preds"]

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["conf"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        ).sort()[0]

        data.filter(keep_by_nms)

        removed_indices = torch.nonzero(keep_tokens)[
            ~torch.isin(torch.arange(keep_tokens.sum(), device=keep_tokens.device), keep_by_nms)]
        keep_tokens[removed_indices] = False

        # Store which cells are predicted to be objects but are not kept by NMS or iou score or mask threshold
        valid_next_frame_mask = object_score_logits_dict["post_div"][:, 0] > self.obj_score_thresh

        if heatmap_input:
            obj_ids = torch.arange(
                inference_state["max_obj_id"] + 1,
                inference_state["max_obj_id"] + 1 + data["masks"].shape[0],
                device=self.device,
                dtype=torch.int32,
            )
            prev_obj_ids = obj_ids.clone()
            inference_state["obj_ids"][frame_idx] = torch.cat([inference_state["obj_ids"][frame_idx], obj_ids])
            inference_state["max_obj_id"] = max(obj_ids.tolist() + [inference_state["max_obj_id"]])
            mother_ids = []
            daughter_ids_list = []
            parent_ids = torch.zeros(len(obj_ids), device=self.device, dtype=torch.int32)
            inference_state["parent_ids"][frame_idx] = torch.cat([inference_state["parent_ids"][frame_idx], parent_ids])

        elif obj_ids is None:  # only in first frame
            num_cells = data["masks"].shape[0]
            obj_ids = torch.arange(num_cells, device=self.device, dtype=torch.int32) + 1
            prev_obj_ids = obj_ids.clone()
            inference_state["obj_ids"] = {frame_idx: obj_ids}
            inference_state["max_obj_id"] = max(obj_ids.tolist())
            mother_ids = []
            daughter_ids_list = []
            parent_ids = torch.zeros(len(obj_ids), device=self.device, dtype=torch.int32)
            inference_state["parent_ids"] = {frame_idx: parent_ids}
            inference_state["lost_obj_ids"] = {frame_idx: torch.zeros(0, device=self.device, dtype=torch.int32)}
            inference_state["lost_high_res_masks"] = {}
        else:
            # Build post-division object IDs in the same order as SAM masks.
            prev_obj_ids = obj_ids.clone()
            mother_ids = obj_ids[is_dividing]
            daughter_ids_list = prev_obj_ids.new_zeros((len(prev_obj_ids), 2), dtype=torch.int32)

            non_div_indices = torch.nonzero(~is_dividing, as_tuple=True)[0]
            div_indices = torch.nonzero(is_dividing, as_tuple=True)[0]

            obj_ids_for_masks = []
            for idx in non_div_indices.tolist():
                obj_ids_for_masks.append(int(obj_ids[idx].item()))

            bud_ids = []
            non_div_count = len(non_div_indices)
            for div_counter, idx in enumerate(div_indices.tolist()):
                mother_id = int(obj_ids[idx].item())
                # Asymmetric division: keep the mother ID and mint a new bud ID.
                bud_id = inference_state["max_obj_id"] + 1 + div_counter
                bud_ids.append(bud_id)
                daughter_ids_list[idx, 0] = bud_id

                mask_idx0 = non_div_count + 2 * div_counter
                mask_idx1 = mask_idx0 + 1

                # Keep IDs aligned with any continuity-based swap in mask order.
                if idx in swapped_div_indices:
                    obj_ids_for_masks.extend([bud_id, mother_id])
                else:
                    obj_ids_for_masks.extend([mother_id, bud_id])

            obj_ids = torch.tensor(obj_ids_for_masks, device=self.device, dtype=torch.int32)

            # Now filter based on NMS results
            lost_obj_ids = obj_ids[valid_next_frame_mask * (~keep_tokens)]
            # If cell divided but is lost through NMS, we remove from error correction as this gets overly complicated
            lost_obj_ids = [obj_id for obj_id in lost_obj_ids if obj_id in prev_obj_ids]
            inference_state["lost_obj_ids"][frame_idx] = lost_obj_ids
            if len(lost_obj_ids) > 0:
                lost_high_res_masks = high_res_masks[valid_next_frame_mask * (~keep_tokens)].flatten(0, 1)
                lost_high_res_masks[:, (data["masks"] > self.mask_threshold).sum(0) > 0] = -torch.inf
                lost_high_res_masks = self.postprocess_mask(lost_high_res_masks, inference_state)
                inference_state["lost_high_res_masks"][frame_idx] = (lost_high_res_masks > self.mask_threshold)

            obj_ids = obj_ids[keep_tokens]

            parent_ids = torch.zeros(len(obj_ids), device=self.device, dtype=torch.int32)

            # Update parent IDs for buds that survived NMS, preserving mother IDs in asym mode.
            for div_idx, mother_id in enumerate(mother_ids.tolist()):
                mother_id = int(mother_id)
                bud_id = bud_ids[div_idx]
                mother_present = bool((obj_ids == mother_id).any())
                bud_present = bool((obj_ids == bud_id).any())

                if bud_present and mother_present:
                    parent_ids[obj_ids == bud_id] = mother_id
                elif bud_present and not mother_present:
                    # If mother was dropped but bud survived, keep the mother ID to avoid ID churn.
                    obj_ids[obj_ids == bud_id] = mother_id
                    daughter_ids_list[div_indices[div_idx], 0] = 0
                else:
                    # Bud was dropped.
                    daughter_ids_list[div_indices[div_idx], 0] = 0

            inference_state["obj_ids"][frame_idx] = obj_ids
            inference_state["max_obj_id"] = max(obj_ids.tolist() + [inference_state["max_obj_id"]])
            inference_state["parent_ids"][frame_idx] = parent_ids

        current_out["pred_masks_high_res"] = data["masks"]
        current_out["pred_object_score_logits"] = data["obj_scores"]
        current_out["obj_ptr"] = data["obj_ptr"]

        if not heatmap_input:
            assert current_out["pred_masks_high_res"].shape[0] == len(obj_ids)

        # Retrieve image features (only need to compute once for all objects)
        (_,_,current_vision_feats,current_vision_pos_embeds,feat_sizes,) = self._get_image_feature(
            inference_state, frame_idx, current_out["pred_masks_high_res"].shape[0])

        if not self.segment:
            inference_state["memory_dict"] = self.model._update_memory_features(
                current_vision_feats,
                feat_sizes,
                inference_state["point_inputs"].get(frame_idx, None),
                run_mem_encoder=True,
                current_out=current_out,
                memory_dict=inference_state["memory_dict"],
                tracking_object_ids=obj_ids,
                frame_idx=frame_idx,
                mother_ids=mother_ids,
                prev_tracking_object_ids=prev_obj_ids,
                daughter_ids_list=daughter_ids_list,
            )

        assert data["save_masks"].shape[0] == data["masks"].shape[0]

        # If no masks are predicted, return an empty track mask
        if data["masks"].shape[0] == 0:
            track_mask = np.zeros((inference_state["video_height"], inference_state["video_width"]),dtype=np.uint16,)
            inference_state["prev_frame_idx"] = frame_idx
            inference_state["prev_masks"] = data["save_masks"].detach()
            inference_state["prev_obj_ids"] = obj_ids.detach()
            return inference_state, track_mask

        track_mask = self.postprocess_mask(data["save_masks"], inference_state)

        # Get the maximum value and index across all masks at each pixel position
        max_values = np.max(track_mask, axis=0)  # returns max values
        arg_max = np.argmax(track_mask, axis=0)  # returns indices

        # Create a mask filled with zeros (background)
        track_mask = np.zeros_like(arg_max)

        # For pixels above threshold, assign the corresponding object ID
        valid_pixels = max_values > self.mask_threshold
        obj_ids_np = obj_ids.cpu().numpy()
        track_mask[valid_pixels] = obj_ids_np[arg_max[valid_pixels]]

        if inference_state.get("prev_frame_idx") == frame_idx:
            # Merge heatmap detections into per-frame cache for next step.
            inference_state["prev_masks"] = torch.cat([inference_state["prev_masks"], data["save_masks"].detach()], dim=0)
            inference_state["prev_obj_ids"] = torch.cat([inference_state["prev_obj_ids"], obj_ids.detach()], dim=0)
        else:
            inference_state["prev_frame_idx"] = frame_idx
            inference_state["prev_masks"] = data["save_masks"].detach()
            inference_state["prev_obj_ids"] = obj_ids.detach()

        return inference_state, track_mask

    def postprocess_mask(self, masks, inference_state):
        pad_left, pad_right, pad_top, pad_bottom = inference_state["padding"]

        pad_right = inference_state["model_image_size"] - pad_right
        pad_bottom = inference_state["model_image_size"] - pad_bottom

        masks = masks[:, pad_top:pad_bottom, pad_left:pad_right]
        masks = masks.permute(1, 2, 0).cpu().numpy()
        masks = cv2.resize(masks, (inference_state["video_width"], inference_state["video_height"]))
        if masks.ndim == 2:
            masks = masks[None, ...]
        else:
            masks = masks.transpose(2, 0, 1)

        return masks

    def get_input_points_from_heatmap(self, inference_state, frame_idx):
        (_,_,current_vision_feats,current_vision_pos_embeds,feat_sizes,) = self._get_image_feature(inference_state, frame_idx, 1)

        heatmap_predictions = self.model.get_heatmap_predictions(current_vision_feats, feat_sizes)[0, 0]
        input_points = self.model.extract_peak_points(heatmap_predictions)
        point_labels = torch.ones((input_points.shape[0], 1), dtype=torch.int, device=self.device)

        return input_points, point_labels
    
    def save_ctc(self, track_mask, frame_idx, inference_state):
        res_path = inference_state["res_path"]

        cell_ids_track_mask = np.unique(track_mask)
        cell_ids_track_mask = cell_ids_track_mask[cell_ids_track_mask != 0]

        # Use obj_ids from inference state to map parent IDs even if track_mask is filtered.
        cell_ids = inference_state["obj_ids"][frame_idx].cpu().numpy()

        if sorted(cell_ids_track_mask) != sorted(cell_ids):
            track_set = set(cell_ids_track_mask.tolist())
            obj_set = set(cell_ids.tolist())
            missing_ids = sorted(obj_set - track_set)
            extra_ids = sorted(track_set - obj_set)
            logger.warning("Mismatch between obj_ids and track_mask ids at frame %s "
                "(missing=%s extra=%s).",frame_idx, missing_ids[:10],extra_ids[:10],)

        if len(cell_ids_track_mask) > 0:
            assert max(cell_ids_track_mask) < 65536, "cell_id must be less than 65536"

        cv2.imwrite(str(res_path / f"mask{frame_idx:03d}.tif"), track_mask.astype(np.uint16))

        if not self.segment:
            parent_ids = inference_state["parent_ids"][frame_idx].cpu().numpy()
            res_track = inference_state["res_track"]
            # Build a safe lookup so missing IDs don't crash asymmetric lineage output.
            parent_lookup = {int(cell_id): int(parent_id) for cell_id, parent_id in zip(cell_ids, parent_ids, strict=False)}

            for cell_id in cell_ids_track_mask.tolist():
                parent_id = parent_lookup.get(int(cell_id), 0)
                if cell_id not in res_track[:, 0]:
                    res_track = np.concatenate([res_track, np.array([[cell_id, frame_idx, frame_idx, parent_id]]),],axis=0,)
                else:
                    prev_end = res_track[res_track[:, 0] == cell_id, 2]
                    if prev_end.size and prev_end[0] != frame_idx - 1:
                        logger.warning("Non-contiguous track for id %s at frame %s (prev_end=%s).",cell_id,frame_idx,int(prev_end[0]),)
                    res_track[res_track[:, 0] == cell_id, 2] = frame_idx

            np.savetxt(res_path / "res_track.txt", res_track, fmt="%d")

            inference_state["res_track"] = res_track

    def save_tracking_results(self, inference_state, tracking_results, alpha=0.3):
        res_path = inference_state["res_path"]

        if self.segment:
            num_colors = 1000
        else:
            num_colors = (inference_state["max_obj_id"] + 1)  # Add 1 to account for 0-based indexing
        colors = np.random.randint(0, 255, (num_colors, 3))
        color_stack = np.zeros((len(tracking_results), inference_state["video_height"], inference_state["video_width"], 3,),dtype=np.uint8,)
        # Keep a persistent mother->bud link overlay for asymmetric lineage visualization.
        parent_map = {}
        if not self.segment:
            res_track = inference_state.get("res_track")
            if res_track is not None and len(res_track) > 0:
                for cell_id, _, _, parent_id in res_track:
                    if parent_id != 0:
                        parent_map[int(cell_id)] = int(parent_id)

        for frame_idx, track_mask in enumerate(tracking_results):
            img = read_image(str(inference_state["video_path"] / f"t{frame_idx:03d}.tif"), return_np=True)

            # Create a colored overlay image
            overlay = np.zeros_like(img)

            cell_ids = np.unique(track_mask)
            cell_ids = cell_ids[cell_ids != 0]  # Exclude background (0)

            # Add colored masks for each cell
            for cell_id in cell_ids:
                mask = track_mask == cell_id
                overlay[mask] = colors[cell_id]

            # Blend original image with colored overlay
            color_stack[frame_idx] = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            centroids = {}
            for cell_id in cell_ids:
                mask = track_mask == cell_id
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0:
                    continue

                centroid_y = int(np.mean(y_coords))
                centroid_x = int(np.mean(x_coords))
                centroids[int(cell_id)] = (centroid_x, centroid_y)

                cv2.putText(
                    color_stack[frame_idx],
                    str(cell_id),
                    (centroid_x - 5, centroid_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (0, 0, 0),  # black color
                    1,  # Line thickness
                    cv2.LINE_AA,
                )

            if not self.segment and parent_map:
                for child_id, parent_id in parent_map.items():
                    child_centroid = centroids.get(child_id)
                    parent_centroid = centroids.get(parent_id)
                    if child_centroid is None or parent_centroid is None:
                        continue
                    cv2.line(
                        color_stack[frame_idx],
                        child_centroid,
                        parent_centroid,
                        (0, 0, 0),  # Black color
                        1,
                    )  # Line thickness

            # Add frame number to top of frame
            cv2.putText(
                color_stack[frame_idx],
                f"{frame_idx:03}",
                (0, 15),  # Position in top-left
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (255, 255, 255),  # White color
                1,  # Line thickness
                cv2.LINE_AA,
            )

        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        mode = "segment" if self.segment else "track"
        out = cv2.VideoWriter(
            str(res_path / f"pred_{mode}_video.mp4"),
            fourcc,
            10.0,  # 10 fps
            (inference_state["video_width"], inference_state["video_height"]),
        )

        for frame in color_stack:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()

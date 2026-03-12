#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight contextual bud ranking model and apply it to a prediction root."
    )
    parser.add_argument("--train-dataset-root", required=True)
    parser.add_argument("--apply-dataset-root", required=True)
    parser.add_argument("--apply-pred-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--refractory", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--blend-alpha", type=float, default=0.6)
    parser.add_argument("--model-type", choices=["baseline", "candidate_attn"], default="baseline")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.18)
    parser.add_argument("--val-group-by", choices=["video", "sample"], default="video")
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--candidate-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def load_dataset(root: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    arrays = dict(np.load(root / "dataset.npz", allow_pickle=True))
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return arrays, samples


class ContextRanker(nn.Module):
    def __init__(self, pair_dim: int, cand_dim: int, bud_dim: int, hidden_dim: int, dropout: float, model_type: str = "baseline", candidate_heads: int = 4):
        super().__init__()
        self.model_type = model_type
        self.frame_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.bud_encoder = nn.Sequential(
            nn.Linear(bud_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if model_type == "candidate_attn":
            self.candidate_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=candidate_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
        else:
            self.candidate_attn = None
            self.attn_norm = None
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_pair: torch.Tensor,
        x_pair_valid: torch.Tensor,
        x_cand: torch.Tensor,
        x_cand_valid: torch.Tensor,
        x_bud: torch.Tensor,
    ) -> torch.Tensor:
        frame_emb = self.frame_encoder(x_pair)
        frame_mask = x_pair_valid.unsqueeze(-1)
        frame_count = frame_mask.sum(dim=2).clamp(min=1)
        masked_frame = frame_emb * frame_mask
        mean_frame = masked_frame.sum(dim=2) / frame_count
        neg_inf = torch.full_like(frame_emb, -1e9)
        max_frame = torch.where(frame_mask, frame_emb, neg_inf).max(dim=2).values
        max_frame = torch.where(torch.isfinite(max_frame), max_frame, torch.zeros_like(max_frame))

        cand_emb = self.candidate_encoder(x_cand)
        bud_emb = self.bud_encoder(x_bud).unsqueeze(1).expand(-1, x_cand.shape[1], -1)

        candidate_repr = mean_frame + max_frame + cand_emb
        if self.candidate_attn is not None:
            attn_out, _ = self.candidate_attn(candidate_repr, candidate_repr, candidate_repr, key_padding_mask=~x_cand_valid)
            candidate_repr = self.attn_norm(candidate_repr + attn_out)
        cand_mask = x_cand_valid.unsqueeze(-1)
        cand_count = cand_mask.sum(dim=1).clamp(min=1)
        group_mean = (candidate_repr * cand_mask).sum(dim=1, keepdim=True) / cand_count.unsqueeze(1)
        group_mean = group_mean.expand_as(candidate_repr)
        combined = torch.cat(
            [
                candidate_repr,
                bud_emb,
                group_mean,
                candidate_repr - group_mean,
                cand_emb,
            ],
            dim=-1,
        )
        logits = self.score_head(combined).squeeze(-1)
        logits = logits.masked_fill(~x_cand_valid, -1e9)
        return logits


def filtered_indices(targets: np.ndarray) -> np.ndarray:
    return np.nonzero(targets >= 0)[0]


def split_train_val_indices(samples: list[dict], targets: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    valid_indices = filtered_indices(targets)
    if len(valid_indices) <= 4 or args.val_fraction <= 0.0:
        return valid_indices, np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    if args.val_group_by == "video":
        groups: dict[str, list[int]] = {}
        for idx in valid_indices.tolist():
            groups.setdefault(samples[idx]["video_id"], []).append(idx)
        video_ids = sorted(groups)
        rng.shuffle(video_ids)
        target_videos = max(1, int(round(len(video_ids) * args.val_fraction)))
        val_videos = set(video_ids[:target_videos])
        train_idx = [idx for vid, idxs in groups.items() if vid not in val_videos for idx in idxs]
        val_idx = [idx for vid, idxs in groups.items() if vid in val_videos for idx in idxs]
        if not train_idx or not val_idx:
            cutoff = max(1, int(round(len(valid_indices) * (1.0 - args.val_fraction))))
            return valid_indices[:cutoff], valid_indices[cutoff:]
        return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)
    order = rng.permutation(valid_indices)
    cutoff = max(1, int(round(len(order) * (1.0 - args.val_fraction))))
    cutoff = min(cutoff, len(order) - 1)
    return order[:cutoff], order[cutoff:]


def batch_tensors(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(arrays["x_pair"][indices]).to(device),
        torch.from_numpy(arrays["x_pair_valid"][indices]).to(device),
        torch.from_numpy(arrays["x_candidate_global"][indices]).to(device),
        torch.from_numpy(arrays["x_candidate_valid"][indices]).to(device),
        torch.from_numpy(arrays["x_bud_global"][indices]).to(device),
        torch.from_numpy(arrays["targets"][indices]).to(device),
    )


def train_model(arrays: dict[str, np.ndarray], samples: list[dict], args: argparse.Namespace) -> tuple[ContextRanker, dict]:
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ContextRanker(
        pair_dim=int(arrays["x_pair"].shape[-1]),
        cand_dim=int(arrays["x_candidate_global"].shape[-1]),
        bud_dim=int(arrays["x_bud_global"].shape[-1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        model_type=args.model_type,
        candidate_heads=args.candidate_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_indices, val_indices = split_train_val_indices(samples, arrays["targets"], args)
    rng = np.random.default_rng(args.seed)
    losses = []
    val_losses = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    patience = 0
    for epoch in range(args.epochs):
        order = rng.permutation(train_indices)
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            x_pair, x_pair_valid, x_cand, x_cand_valid, x_bud, targets = batch_tensors(arrays, batch_idx, device)
            logits = model(x_pair, x_pair_valid.bool(), x_cand, x_cand_valid.bool(), x_bud)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
        losses.append(epoch_loss / max(num_batches, 1))

        if len(val_indices) > 0:
            model.eval()
            with torch.no_grad():
                x_pair, x_pair_valid, x_cand, x_cand_valid, x_bud, targets = batch_tensors(arrays, val_indices, device)
                logits = model(x_pair, x_pair_valid.bool(), x_cand, x_cand_valid.bool(), x_bud)
                val_loss = float(F.cross_entropy(logits, targets).item())
            val_losses.append(val_loss)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_info = {
        "device": device.type,
        "model_type": args.model_type,
        "epochs_requested": args.epochs,
        "epochs_trained": len(losses),
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "final_train_loss": losses[-1] if losses else None,
        "min_train_loss": min(losses) if losses else None,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
        "num_train_samples": int(len(train_indices)),
        "num_val_samples": int(len(val_indices)),
        "val_group_by": args.val_group_by,
    }
    return model, train_info


def evaluate_model(model: ContextRanker, arrays: dict[str, np.ndarray], args: argparse.Namespace) -> dict:
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    valid_indices = filtered_indices(arrays["targets"])
    if len(valid_indices) == 0:
        return {"num_samples": 0, "top1_accuracy": None}
    x_pair, x_pair_valid, x_cand, x_cand_valid, x_bud, targets = batch_tensors(arrays, valid_indices, device)
    with torch.no_grad():
        logits = model.to(device)(x_pair, x_pair_valid.bool(), x_cand, x_cand_valid.bool(), x_bud)
        pred = logits.argmax(dim=1)
        acc = float((pred == targets).float().mean().item())
    return {"num_samples": int(len(valid_indices)), "top1_accuracy": acc}


def softmax_scores(logits: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    masked = np.where(valid_mask, logits, -1e9)
    masked = masked - np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked) * valid_mask
    denom = exp.sum(axis=-1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return exp / denom


def _api_cfg_from_args(args: argparse.Namespace):
    from tools.parentage_api import ParentageConfig

    return ParentageConfig(
        refractory=args.refractory,
        max_dist_factor=2.5,
        bud_max_area_ratio=0.6,
        min_bud_area=10,
        min_mother_age=3,
        min_score=0.12,
        min_track_length=2,
        first_frames=4,
        w_dist=0.6,
        w_size=0.3,
        w_motion=0.1,
        w_contact=0.1,
        w_neck=0.25,
        w_prebud=0.0,
        w_angle=0.20,
        w_maturity=0.0,
        w_track_quality=0.10,
        w_margin=0.15,
        w_lineage=0.0,
        motion_scale=2.0,
        interface_radius=4,
        use_border_neck=False,
        preferred_mother_age=24,
        lineage_margin=0.06,
        score_threshold=args.score_threshold,
        learned_score_alpha=args.blend_alpha,
        device=args.device,
    )


def apply_model(
    model: ContextRanker,
    arrays: dict[str, np.ndarray],
    samples: list[dict],
    apply_pred_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    from tools.parentage_api import (
        CandidateScore,
        ScoredCandidates,
        assign_parentage,
        build_candidates,
        write_assignments,
    )

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    cfg = _api_cfg_from_args(args)

    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(apply_pred_root, output_root)

    x_pair = torch.from_numpy(arrays["x_pair"]).to(device)
    x_pair_valid = torch.from_numpy(arrays["x_pair_valid"]).to(device).bool()
    x_cand = torch.from_numpy(arrays["x_candidate_global"]).to(device)
    x_cand_valid = torch.from_numpy(arrays["x_candidate_valid"]).to(device).bool()
    x_bud = torch.from_numpy(arrays["x_bud_global"]).to(device)

    with torch.no_grad():
        logits = model(x_pair, x_pair_valid, x_cand, x_cand_valid, x_bud).detach().cpu().numpy()
    probs = softmax_scores(logits, arrays["x_candidate_valid"])

    by_video: dict[str, list[tuple[dict, np.ndarray]]] = {}
    for sample, sample_probs in zip(samples, probs.tolist()):
        by_video.setdefault(sample["video_id"], []).append((sample, np.asarray(sample_probs, dtype=np.float64)))

    written = 0
    for video_id, items in by_video.items():
        seq_dir = output_root / video_id
        candidate_table = build_candidates(seq_dir=seq_dir, cfg=cfg)
        candidates_by_bud = candidate_table.candidates_by_bud
        scored_by_bud: dict[int, list[CandidateScore]] = {}
        rows: list[list[object]] = []

        for sample, sample_probs in items:
            bud_id = int(sample["pred_bud_id"])
            cand_list = candidates_by_bud.get(bud_id, [])
            by_mother = {cand.mother_id: cand for cand in cand_list}
            bud_scores: list[CandidateScore] = []
            for cand_info, prob in zip(sample["candidates"], sample_probs):
                mother_id = int(cand_info["mother_id"])
                cand = by_mother.get(mother_id)
                if cand is None:
                    continue
                heuristic_score = float(cand.score)
                blended_score = float(np.clip(args.blend_alpha * prob + (1.0 - args.blend_alpha) * heuristic_score, 0.0, 1.0))
                bud_scores.append(
                    CandidateScore(
                        bud_id=bud_id,
                        mother_id=mother_id,
                        heuristic_score=heuristic_score,
                        model_score=float(prob),
                        score=blended_score,
                        candidate=replace(cand, score=blended_score),
                    )
                )
                rows.append(
                    [
                        bud_id,
                        mother_id,
                        f"{prob:.6f}",
                        f"{heuristic_score:.6f}",
                        f"{blended_score:.6f}",
                        int(cand_info["label"]),
                        int(sample["target_index"]),
                    ]
                )
            if bud_scores:
                bud_scores.sort(key=lambda item: item.score, reverse=True)
                scored_by_bud[bud_id] = bud_scores

        scored_candidates = ScoredCandidates(
            scorer_name="context",
            candidate_table=candidate_table,
            scored_by_bud=scored_by_bud,
        )
        assignment = assign_parentage(candidate_table=candidate_table, scored_candidates=scored_candidates, cfg=cfg, mode="ilp")
        write_assignments(assignment, seq_dir / "res_track.txt")
        written += sum(1 for cand in assignment.assigned_by_bud.values() if cand.score >= args.score_threshold)

        with (seq_dir / "bud_parentage_context.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["bud_id", "mother_id", "model_prob", "heuristic_score", "blended_score", "label", "target_index"])
            writer.writerows(rows)

    return {"num_written_parent_assignments": int(written), "num_samples": int(len(samples))}


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_dataset_root).expanduser().resolve()
    apply_root = Path(args.apply_dataset_root).expanduser().resolve()
    apply_pred_root = Path(args.apply_pred_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_arrays, train_samples = load_dataset(train_root)
    apply_arrays, apply_samples = load_dataset(apply_root)
    model, train_info = train_model(train_arrays, train_samples, args)
    train_eval = evaluate_model(model, train_arrays, args)
    apply_eval = evaluate_model(model, apply_arrays, args)
    apply_info = apply_model(model, apply_arrays, apply_samples, apply_pred_root, output_root / "predictions", args)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "model_type": args.model_type,
            "candidate_heads": args.candidate_heads,
        },
        output_root / "context_ranker.pt",
    )

    summary = {
        "model": "context_ranker_v1",
        "train": train_info,
        "train_eval": train_eval,
        "apply_eval": apply_eval,
        "apply": apply_info,
        "blend_alpha": args.blend_alpha,
        "score_threshold": args.score_threshold,
    }
    (output_root / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

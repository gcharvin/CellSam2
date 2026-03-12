#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.online_bud_parentage import (
    Candidate,
    _load_res_track,
    _solve_global_ilp,
    _write_res_track,
)
from tools.learned_bud_rerank import build_seq_context


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
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def load_dataset(root: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    arrays = dict(np.load(root / "dataset.npz", allow_pickle=True))
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return arrays, samples


class ContextRanker(nn.Module):
    def __init__(self, pair_dim: int, cand_dim: int, bud_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
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


def batch_tensors(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(arrays["x_pair"][indices]).to(device),
        torch.from_numpy(arrays["x_pair_valid"][indices]).to(device),
        torch.from_numpy(arrays["x_candidate_global"][indices]).to(device),
        torch.from_numpy(arrays["x_candidate_valid"][indices]).to(device),
        torch.from_numpy(arrays["x_bud_global"][indices]).to(device),
        torch.from_numpy(arrays["targets"][indices]).to(device),
    )


def train_model(arrays: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[ContextRanker, dict]:
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
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    valid_indices = filtered_indices(arrays["targets"])
    rng = np.random.default_rng(args.seed)
    losses = []
    for _ in range(args.epochs):
        order = rng.permutation(valid_indices)
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

    model.eval()
    train_info = {
        "device": device.type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "num_train_samples": int(len(valid_indices)),
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


def apply_model(
    model: ContextRanker,
    arrays: dict[str, np.ndarray],
    samples: list[dict],
    apply_pred_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()

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
        seq_context = build_seq_context(seq_dir, argparse.Namespace(
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
        ))
        track_infos = seq_context["track_infos"]
        candidates_by_bud = seq_context["candidates_by_bud"]
        res_track = _load_res_track(seq_dir / "res_track.txt")
        all_candidates: list[Candidate] = []
        rows: list[list[object]] = []

        for sample, sample_probs in items:
            bud_id = int(sample["pred_bud_id"])
            cand_list = candidates_by_bud.get(bud_id, [])
            by_mother = {cand.mother_id: cand for cand in cand_list}
            for cand_info, prob in zip(sample["candidates"], sample_probs):
                mother_id = int(cand_info["mother_id"])
                cand = by_mother.get(mother_id)
                if cand is None:
                    continue
                heuristic_score = float(cand.score)
                cand.score = float(np.clip(args.blend_alpha * prob + (1.0 - args.blend_alpha) * heuristic_score, 0.0, 1.0))
                all_candidates.append(cand)
                rows.append(
                    [
                        bud_id,
                        mother_id,
                        f"{prob:.6f}",
                        f"{heuristic_score:.6f}",
                        f"{cand.score:.6f}",
                        int(cand_info["label"]),
                        int(sample["target_index"]),
                    ]
                )

        assigned = _solve_global_ilp(
            bud_ids=sorted({cand.bud_id for cand in all_candidates}),
            candidates=all_candidates,
            track_infos=track_infos,
            refractory_frames=args.refractory,
        )
        for bud_id, cand in assigned.items():
            if cand.score < args.score_threshold:
                continue
            mask = res_track[:, 0] == bud_id
            if mask.any():
                res_track[mask, 3] = cand.mother_id
                written += 1
        _write_res_track(seq_dir / "res_track.txt", res_track)

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

    train_arrays, _ = load_dataset(train_root)
    apply_arrays, apply_samples = load_dataset(apply_root)
    model, train_info = train_model(train_arrays, args)
    train_eval = evaluate_model(model, train_arrays, args)
    apply_eval = evaluate_model(model, apply_arrays, args)
    apply_info = apply_model(model, apply_arrays, apply_samples, apply_pred_root, output_root / "predictions", args)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
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


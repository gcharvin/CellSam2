#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnose_bud_events import classify_event, load_track_rows, rows_to_dict
from tools.learned_bud_rerank import build_seq_context, list_videos, map_parent_to_gt
from tools.online_bud_parentage import _contact_score, _mother_radius, _neck_score

PAIR_FEATURE_NAMES = [
    "frame_exists",
    "bud_visible",
    "candidate_alive",
    "dist_norm",
    "contact",
    "neck",
    "size_ratio",
    "mother_age",
    "dir_x",
    "dir_y",
    "is_best_dist",
    "rank_dist",
    "margin_best_dist",
    "is_best_contact",
    "rank_contact",
    "margin_best_contact",
    "num_alive_candidates",
]

CANDIDATE_GLOBAL_FEATURE_NAMES = [
    "heuristic_score",
    "dist",
    "size_ratio",
    "motion",
    "contact",
    "neck",
    "prebud",
    "angle",
    "maturity",
    "track_quality",
    "margin",
    "mother_age",
    "proposal_match",
    "lineage",
    "mapped_parent_gt",
    "label",
    "alive_fraction",
    "best_dist_fraction",
    "best_contact_fraction",
    "mean_dist_norm",
    "min_dist_norm",
    "mean_contact",
    "max_contact",
    "mean_neck",
    "max_neck",
]

BUD_GLOBAL_FEATURE_NAMES = [
    "support_frames",
    "support_total",
    "dominance",
    "onset_delay",
    "track_length",
    "num_candidates",
    "first_area",
    "last_area",
    "area_growth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a contextual bud->mother ranking dataset with framewise and relative candidate features."
    )
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--videos", default="")
    parser.add_argument("--window-pre", type=int, default=1)
    parser.add_argument("--window-post", type=int, default=3)
    parser.add_argument("--analysis-frames", type=int, default=5)
    parser.add_argument("--birth-search-window", type=int, default=3)
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    parser.add_argument("--stable-support", type=int, default=2)
    parser.add_argument("--stable-fraction", type=float, default=0.6)
    parser.add_argument("--parent-map-threshold", type=float, default=0.2)
    parser.add_argument("--refractory", type=int, default=8)
    parser.add_argument("--max-dist-factor", type=float, default=2.5)
    parser.add_argument("--bud-max-area-ratio", type=float, default=0.6)
    parser.add_argument("--min-bud-area", type=int, default=10)
    parser.add_argument("--min-mother-age", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--first-frames", type=int, default=4)
    parser.add_argument("--w-dist", type=float, default=0.6)
    parser.add_argument("--w-size", type=float, default=0.3)
    parser.add_argument("--w-motion", type=float, default=0.1)
    parser.add_argument("--w-contact", type=float, default=0.1)
    parser.add_argument("--w-neck", type=float, default=0.25)
    parser.add_argument("--w-prebud", type=float, default=0.0)
    parser.add_argument("--w-angle", type=float, default=0.20)
    parser.add_argument("--w-maturity", type=float, default=0.0)
    parser.add_argument("--w-track-quality", type=float, default=0.10)
    parser.add_argument("--w-margin", type=float, default=0.15)
    parser.add_argument("--w-lineage", type=float, default=0.0)
    parser.add_argument("--motion-scale", type=float, default=2.0)
    parser.add_argument("--interface-radius", type=int, default=4)
    parser.add_argument("--use-border-neck", action="store_true")
    parser.add_argument("--preferred-mother-age", type=int, default=24)
    parser.add_argument("--lineage-margin", type=float, default=0.06)
    return parser.parse_args()


def build_context_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        refractory=args.refractory,
        max_dist_factor=args.max_dist_factor,
        bud_max_area_ratio=args.bud_max_area_ratio,
        min_bud_area=args.min_bud_area,
        min_mother_age=args.min_mother_age,
        min_score=args.min_score,
        min_track_length=args.min_track_length,
        first_frames=args.first_frames,
        w_dist=args.w_dist,
        w_size=args.w_size,
        w_motion=args.w_motion,
        w_contact=args.w_contact,
        w_neck=args.w_neck,
        w_prebud=args.w_prebud,
        w_angle=args.w_angle,
        w_maturity=args.w_maturity,
        w_track_quality=args.w_track_quality,
        w_margin=args.w_margin,
        w_lineage=args.w_lineage,
        motion_scale=args.motion_scale,
        interface_radius=args.interface_radius,
        use_border_neck=args.use_border_neck,
        preferred_mother_age=args.preferred_mother_age,
        lineage_margin=args.lineage_margin,
    )


def safe_float(value: float | int | None) -> float:
    if value is None:
        return -1.0
    return float(value)


def compute_pair_frame_features(frame_idx: int, candidate, candidate_ids: list[int], seq_context: dict, interface_radius: int, use_border_neck: bool) -> tuple[list[float], bool]:
    stats_cache = seq_context["stats_cache"]
    bin_masks = seq_context["bin_masks"]
    track_infos = seq_context["track_infos"]
    frame_stats = stats_cache.get(frame_idx, {})
    bud_stats = frame_stats.get(candidate.bud_id)
    mother_stats = frame_stats.get(candidate.mother_id)
    bud_mask = bin_masks.get(frame_idx, {}).get(candidate.bud_id)
    mother_mask = bin_masks.get(frame_idx, {}).get(candidate.mother_id)

    frame_exists = 1.0 if frame_stats else 0.0
    bud_visible = 1.0 if bud_stats is not None and bud_mask is not None and bud_mask.any() else 0.0
    candidate_alive = 1.0 if mother_stats is not None and mother_mask is not None and mother_mask.any() else 0.0

    if bud_visible == 0.0 or candidate_alive == 0.0:
        return [frame_exists, bud_visible, candidate_alive] + [0.0] * (len(PAIR_FEATURE_NAMES) - 3), False

    radius = max(_mother_radius(mother_stats.area), 1e-6)
    dist_norm = float(np.hypot(bud_stats.cx - mother_stats.cx, bud_stats.cy - mother_stats.cy) / radius)
    contact = float(_contact_score(bud_mask, mother_mask, interface_radius))
    neck = float(_neck_score(bud_mask, mother_mask, interface_radius, use_border_neck))
    size_ratio = float(bud_stats.area / max(mother_stats.area, 1))
    mother_age = float(max(0, frame_idx - track_infos[candidate.mother_id].start))
    dx = float((bud_stats.cx - mother_stats.cx) / radius)
    dy = float((bud_stats.cy - mother_stats.cy) / radius)

    alive_distances = []
    alive_contacts = []
    for mother_id in candidate_ids:
        other_stats = frame_stats.get(mother_id)
        other_mask = bin_masks.get(frame_idx, {}).get(mother_id)
        if other_stats is None or other_mask is None or not other_mask.any():
            continue
        other_radius = max(_mother_radius(other_stats.area), 1e-6)
        other_dist = float(np.hypot(bud_stats.cx - other_stats.cx, bud_stats.cy - other_stats.cy) / other_radius)
        other_contact = float(_contact_score(bud_mask, other_mask, interface_radius))
        alive_distances.append((mother_id, other_dist))
        alive_contacts.append((mother_id, other_contact))

    alive_distances.sort(key=lambda item: item[1])
    alive_contacts.sort(key=lambda item: item[1], reverse=True)
    num_alive_candidates = float(len(alive_distances))

    dist_rank_idx = next((idx for idx, (mid, _) in enumerate(alive_distances) if mid == candidate.mother_id), len(alive_distances) - 1)
    contact_rank_idx = next((idx for idx, (mid, _) in enumerate(alive_contacts) if mid == candidate.mother_id), len(alive_contacts) - 1)

    rank_dist = float(dist_rank_idx / max(len(alive_distances) - 1, 1)) if alive_distances else 1.0
    rank_contact = float(contact_rank_idx / max(len(alive_contacts) - 1, 1)) if alive_contacts else 1.0
    best_dist = alive_distances[0][1] if alive_distances else dist_norm
    best_contact = alive_contacts[0][1] if alive_contacts else contact
    margin_best_dist = float(max(0.0, dist_norm - best_dist))
    margin_best_contact = float(max(0.0, best_contact - contact))
    is_best_dist = 1.0 if alive_distances and alive_distances[0][0] == candidate.mother_id else 0.0
    is_best_contact = 1.0 if alive_contacts and alive_contacts[0][0] == candidate.mother_id else 0.0

    values = [
        frame_exists,
        bud_visible,
        candidate_alive,
        dist_norm,
        contact,
        neck,
        size_ratio,
        mother_age,
        dx,
        dy,
        is_best_dist,
        rank_dist,
        margin_best_dist,
        is_best_contact,
        rank_contact,
        margin_best_contact,
        num_alive_candidates,
    ]
    return values, True


def summarize_candidate_frames(frame_features: list[list[float]], valid_flags: list[bool]) -> dict[str, float]:
    valid_rows = [row for row, valid in zip(frame_features, valid_flags) if valid]
    if not valid_rows:
        return {
            "alive_fraction": 0.0,
            "best_dist_fraction": 0.0,
            "best_contact_fraction": 0.0,
            "mean_dist_norm": 0.0,
            "min_dist_norm": 0.0,
            "mean_contact": 0.0,
            "max_contact": 0.0,
            "mean_neck": 0.0,
            "max_neck": 0.0,
        }
    arr = np.asarray(valid_rows, dtype=np.float32)
    idx = {name: i for i, name in enumerate(PAIR_FEATURE_NAMES)}
    return {
        "alive_fraction": float(np.mean(arr[:, idx["candidate_alive"]])),
        "best_dist_fraction": float(np.mean(arr[:, idx["is_best_dist"]])),
        "best_contact_fraction": float(np.mean(arr[:, idx["is_best_contact"]])),
        "mean_dist_norm": float(np.mean(arr[:, idx["dist_norm"]])),
        "min_dist_norm": float(np.min(arr[:, idx["dist_norm"]])),
        "mean_contact": float(np.mean(arr[:, idx["contact"]])),
        "max_contact": float(np.max(arr[:, idx["contact"]])),
        "mean_neck": float(np.mean(arr[:, idx["neck"]])),
        "max_neck": float(np.max(arr[:, idx["neck"]])),
    }


def build_sample(video_id: str, gt_event: dict[str, int], event: dict, seq_context: dict, gt_dir: Path, args: argparse.Namespace) -> dict | None:
    pred_bud_id = event.get("dominant_pred_id")
    if pred_bud_id is None:
        return None
    candidates_by_bud = seq_context["candidates_by_bud"]
    if pred_bud_id not in candidates_by_bud:
        return None

    track_infos = seq_context["track_infos"]
    stats_cache = seq_context["stats_cache"]
    pred_tracks = rows_to_dict(load_track_rows(seq_context["seq_dir"] / "res_track.txt"))
    bud_track = track_infos[pred_bud_id]
    frame_offsets = list(range(-args.window_pre, args.window_post + 1))
    frame_ids = [bud_track.start + off for off in frame_offsets]
    candidate_list = sorted(candidates_by_bud[pred_bud_id], key=lambda item: item.score, reverse=True)
    proposal_parent = pred_tracks.get(pred_bud_id, {}).get("parent", 0)

    first_stats = stats_cache.get(bud_track.start, {}).get(pred_bud_id)
    last_stats = stats_cache.get(bud_track.end, {}).get(pred_bud_id)
    first_area = int(first_stats.area) if first_stats is not None else 0
    last_area = int(last_stats.area) if last_stats is not None else 0
    bud_global = {
        "support_frames": int(event.get("support_frames", 0)),
        "support_total": int(event.get("support_total", 0)),
        "dominance": float(event.get("dominance", 0.0)),
        "onset_delay": safe_float(event.get("onset_delay")),
        "track_length": float(bud_track.end - bud_track.start + 1),
        "num_candidates": float(len(candidate_list)),
        "first_area": float(first_area),
        "last_area": float(last_area),
        "area_growth": float(last_area - first_area),
    }

    candidates = []
    positive_indices = []
    candidate_ids = [cand.mother_id for cand in candidate_list]
    for cand_idx, cand in enumerate(candidate_list):
        mapped_parent_gt = map_parent_to_gt(
            seq_dir=seq_context["seq_dir"],
            gt_dir=gt_dir,
            frame_idx=cand.frame_idx,
            pred_parent_id=cand.mother_id,
            threshold=args.parent_map_threshold,
        )
        label = 1 if mapped_parent_gt == gt_event["parent"] else 0
        if label == 1:
            positive_indices.append(cand_idx)

        frame_features = []
        valid_flags = []
        for frame_idx in frame_ids:
            feats, valid = compute_pair_frame_features(
                frame_idx=frame_idx,
                candidate=cand,
                candidate_ids=candidate_ids,
                seq_context=seq_context,
                interface_radius=args.interface_radius,
                use_border_neck=args.use_border_neck,
            )
            frame_features.append(feats)
            valid_flags.append(valid)

        frame_summary = summarize_candidate_frames(frame_features, valid_flags)
        candidate_global = {
            "heuristic_score": float(cand.score),
            "dist": float(cand.dist),
            "size_ratio": float(cand.size_ratio),
            "motion": float(cand.motion),
            "contact": float(cand.contact),
            "neck": float(cand.neck),
            "prebud": float(cand.prebud),
            "angle": float(cand.angle),
            "maturity": float(cand.maturity),
            "track_quality": float(cand.track_quality),
            "margin": float(cand.margin),
            "mother_age": float(cand.mother_age),
            "proposal_match": 1.0 if cand.mother_id == proposal_parent else 0.0,
            "lineage": float(cand.lineage),
            "mapped_parent_gt": safe_float(mapped_parent_gt),
            "label": float(label),
            **frame_summary,
        }
        candidates.append(
            {
                "mother_id": int(cand.mother_id),
                "label": int(label),
                "mapped_parent_gt": mapped_parent_gt,
                "global_features": candidate_global,
                "frame_features": frame_features,
                "frame_valid": valid_flags,
            }
        )

    target_index = positive_indices[0] if positive_indices else -1
    return {
        "video_id": video_id,
        "gt_bud_id": int(gt_event["id"]),
        "gt_parent_id": int(gt_event["parent"]),
        "pred_bud_id": int(pred_bud_id),
        "bud_start": int(bud_track.start),
        "bud_end": int(bud_track.end),
        "frame_offsets": frame_offsets,
        "frame_ids": frame_ids,
        "target_index": int(target_index),
        "positive_indices": positive_indices,
        "bud_global_features": bud_global,
        "candidates": candidates,
        "event_category": event.get("category"),
        "event_group": event.get("group"),
    }


def write_npz(samples: list[dict], output_root: Path) -> dict:
    num_samples = len(samples)
    max_candidates = max(len(sample["candidates"]) for sample in samples)
    num_frames = len(samples[0]["frame_offsets"]) if samples else 0
    pair_dim = len(PAIR_FEATURE_NAMES)
    cand_dim = len(CANDIDATE_GLOBAL_FEATURE_NAMES)
    bud_dim = len(BUD_GLOBAL_FEATURE_NAMES)

    x_pair = np.zeros((num_samples, max_candidates, num_frames, pair_dim), dtype=np.float32)
    x_pair_valid = np.zeros((num_samples, max_candidates, num_frames), dtype=np.bool_)
    x_cand = np.zeros((num_samples, max_candidates, cand_dim), dtype=np.float32)
    x_cand_valid = np.zeros((num_samples, max_candidates), dtype=np.bool_)
    x_bud = np.zeros((num_samples, bud_dim), dtype=np.float32)
    targets = np.full((num_samples,), -1, dtype=np.int64)
    positive_mask = np.zeros((num_samples, max_candidates), dtype=np.bool_)
    candidate_ids = np.full((num_samples, max_candidates), -1, dtype=np.int64)
    sample_ids = []

    for sample_idx, sample in enumerate(samples):
        sample_ids.append(f"{sample['video_id']}:{sample['pred_bud_id']}")
        x_bud[sample_idx] = np.asarray([sample["bud_global_features"][name] for name in BUD_GLOBAL_FEATURE_NAMES], dtype=np.float32)
        targets[sample_idx] = int(sample["target_index"])
        for cand_idx, candidate in enumerate(sample["candidates"]):
            x_cand_valid[sample_idx, cand_idx] = True
            candidate_ids[sample_idx, cand_idx] = int(candidate["mother_id"])
            positive_mask[sample_idx, cand_idx] = bool(candidate["label"])
            x_cand[sample_idx, cand_idx] = np.asarray(
                [candidate["global_features"][name] for name in CANDIDATE_GLOBAL_FEATURE_NAMES],
                dtype=np.float32,
            )
            x_pair[sample_idx, cand_idx] = np.asarray(candidate["frame_features"], dtype=np.float32)
            x_pair_valid[sample_idx, cand_idx] = np.asarray(candidate["frame_valid"], dtype=np.bool_)

    np.savez_compressed(
        output_root / "dataset.npz",
        x_pair=x_pair,
        x_pair_valid=x_pair_valid,
        x_candidate_global=x_cand,
        x_candidate_valid=x_cand_valid,
        x_bud_global=x_bud,
        targets=targets,
        positive_mask=positive_mask,
        candidate_ids=candidate_ids,
        sample_ids=np.asarray(sample_ids, dtype=object),
        frame_offsets=np.asarray(samples[0]["frame_offsets"] if samples else [], dtype=np.int32),
    )
    return {
        "num_samples": num_samples,
        "max_candidates": int(max_candidates),
        "num_frames": int(num_frames),
        "pair_dim": int(pair_dim),
        "candidate_global_dim": int(cand_dim),
        "bud_global_dim": int(bud_dim),
    }


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root).expanduser().resolve()
    pred_root = Path(args.pred_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    context_args = build_context_args(args)
    samples = []
    skipped = {"missing_tracklet": 0, "no_candidates": 0, "other": 0}

    for video_id in list_videos(gt_root, args.videos):
        gt_dir = gt_root / f"{video_id}_GT" / "TRA"
        pred_dir = pred_root / video_id
        if not pred_dir.exists():
            continue
        gt_tracks = rows_to_dict(load_track_rows(gt_dir / "man_track.txt"))
        pred_tracks = rows_to_dict(load_track_rows(pred_dir / "res_track.txt"))
        seq_context = build_seq_context(pred_dir, context_args)
        for gt_event in gt_tracks.values():
            if gt_event["parent"] <= 0:
                continue
            event = classify_event(
                gt_event=gt_event,
                pred_tracks=pred_tracks,
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                analysis_frames=args.analysis_frames,
                birth_search_window=args.birth_search_window,
                overlap_threshold=args.overlap_threshold,
                stable_support=args.stable_support,
                stable_fraction=args.stable_fraction,
                parent_map_threshold=args.parent_map_threshold,
            )
            sample = build_sample(video_id, gt_event, event, seq_context, gt_dir, args)
            if sample is None:
                category = event.get("category") or "other"
                if category == "missing_tracklet":
                    skipped["missing_tracklet"] += 1
                elif event.get("dominant_pred_id") is not None:
                    skipped["no_candidates"] += 1
                else:
                    skipped["other"] += 1
                continue
            samples.append(sample)

    samples_path = output_root / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")

    npz_meta = write_npz(samples, output_root) if samples else {
        "num_samples": 0,
        "max_candidates": 0,
        "num_frames": 0,
        "pair_dim": len(PAIR_FEATURE_NAMES),
        "candidate_global_dim": len(CANDIDATE_GLOBAL_FEATURE_NAMES),
        "bud_global_dim": len(BUD_GLOBAL_FEATURE_NAMES),
    }

    positives = [len(sample["positive_indices"]) for sample in samples]
    summary = {
        "gt_root": str(gt_root),
        "pred_root": str(pred_root),
        "videos": list_videos(gt_root, args.videos),
        "num_samples": len(samples),
        "num_unique_target": int(sum(1 for n in positives if n == 1)),
        "num_multi_positive": int(sum(1 for n in positives if n > 1)),
        "num_no_positive": int(sum(1 for n in positives if n == 0)),
        "avg_candidates": float(np.mean([len(sample["candidates"]) for sample in samples])) if samples else 0.0,
        "skipped": skipped,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "candidate_global_feature_names": CANDIDATE_GLOBAL_FEATURE_NAMES,
        "bud_global_feature_names": BUD_GLOBAL_FEATURE_NAMES,
        "npz": npz_meta,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

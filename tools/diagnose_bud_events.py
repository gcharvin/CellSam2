#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_track_rows(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    data = np.loadtxt(path, dtype=np.int32)
    if data.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def rows_to_dict(rows: np.ndarray) -> dict[int, dict[str, int]]:
    return {
        int(obj_id): {
            "id": int(obj_id),
            "start": int(start),
            "end": int(end),
            "parent": int(parent),
        }
        for obj_id, start, end, parent in rows.tolist()
    }


def load_mask(path: Path):
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def best_pred_for_gt(gt_mask: np.ndarray, pred_mask: np.ndarray, gt_label: int):
    gt_bin = gt_mask == gt_label
    gt_area = int(gt_bin.sum())
    if gt_area == 0:
        return None
    labels, counts = np.unique(pred_mask[gt_bin], return_counts=True)
    best = None
    for pred_label, overlap in zip(labels.tolist(), counts.tolist()):
        if pred_label == 0 or overlap <= 0:
            continue
        pred_bin = pred_mask == pred_label
        pred_area = int(pred_bin.sum())
        inter = int(overlap)
        union = pred_area + gt_area - inter
        iou = inter / union if union > 0 else 0.0
        gt_coverage = inter / gt_area
        pred_coverage = inter / pred_area if pred_area > 0 else 0.0
        candidate = {
            "pred_label": int(pred_label),
            "gt_coverage": float(gt_coverage),
            "pred_coverage": float(pred_coverage),
            "iou": float(iou),
            "intersection": inter,
        }
        if best is None or candidate["gt_coverage"] > best["gt_coverage"]:
            best = candidate
    return best


def best_gt_for_pred(gt_mask: np.ndarray, pred_mask: np.ndarray, pred_label: int):
    pred_bin = pred_mask == pred_label
    pred_area = int(pred_bin.sum())
    if pred_area == 0:
        return None
    labels, counts = np.unique(gt_mask[pred_bin], return_counts=True)
    best = None
    for gt_label, overlap in zip(labels.tolist(), counts.tolist()):
        if gt_label == 0 or overlap <= 0:
            continue
        gt_bin = gt_mask == gt_label
        gt_area = int(gt_bin.sum())
        inter = int(overlap)
        union = pred_area + gt_area - inter
        iou = inter / union if union > 0 else 0.0
        pred_coverage = inter / pred_area
        gt_coverage = inter / gt_area if gt_area > 0 else 0.0
        candidate = {
            "gt_label": int(gt_label),
            "pred_coverage": float(pred_coverage),
            "gt_coverage": float(gt_coverage),
            "iou": float(iou),
            "intersection": inter,
        }
        if best is None or candidate["pred_coverage"] > best["pred_coverage"]:
            best = candidate
    return best


def frame_path(mask_dir: Path, frame_idx: int, prefix: str) -> Path:
    return mask_dir / f"{prefix}{frame_idx:03d}.tif"


def classify_event(
    gt_event: dict[str, int],
    pred_tracks: dict[int, dict[str, int]],
    gt_dir: Path,
    pred_dir: Path,
    analysis_frames: int,
    birth_search_window: int,
    overlap_threshold: float,
    stable_support: int,
    stable_fraction: float,
    parent_map_threshold: float,
):
    gt_bud_id = gt_event["id"]
    gt_parent_id = gt_event["parent"]
    gt_start = gt_event["start"]
    gt_end = gt_event["end"]
    last_frame = min(gt_end, gt_start + analysis_frames - 1)

    matches = []
    unique_pred_labels = set()
    label_support = Counter()
    label_cov_sum = defaultdict(float)

    for frame_idx in range(gt_start, last_frame + 1):
        gt_mask = load_mask(frame_path(gt_dir, frame_idx, "man_track"))
        pred_mask = load_mask(frame_path(pred_dir, frame_idx, "mask"))
        if gt_mask is None or pred_mask is None:
            continue
        best = best_pred_for_gt(gt_mask, pred_mask, gt_bud_id)
        if best is None:
            continue
        if best["gt_coverage"] < overlap_threshold:
            continue
        best["frame"] = frame_idx
        matches.append(best)
        label = best["pred_label"]
        unique_pred_labels.add(label)
        label_support[label] += 1
        label_cov_sum[label] += best["gt_coverage"]

    detection_frame = min((m["frame"] for m in matches), default=None)
    if detection_frame is None:
        return {
            "category": "missing_tracklet",
            "group": "tracking",
            "gt_bud_id": gt_bud_id,
            "gt_parent_id": gt_parent_id,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "onset_delay": None,
            "dominant_pred_id": None,
            "dominant_pred_parent_id": None,
            "mapped_pred_parent_gt": None,
            "support_frames": 0,
            "unique_pred_labels": 0,
            "match_frames": [],
        }

    onset_delay = detection_frame - gt_start
    dominant_pred_id = max(
        label_support,
        key=lambda label: (label_support[label], label_cov_sum[label], -label),
    )
    support_frames = int(label_support[dominant_pred_id])
    support_total = len(matches)
    dominance = support_frames / max(support_total, 1)
    is_stable = support_frames >= stable_support and dominance >= stable_fraction

    pred_row = pred_tracks.get(dominant_pred_id)
    pred_parent_id = int(pred_row["parent"]) if pred_row is not None else 0
    mapped_pred_parent_gt = None

    if onset_delay > birth_search_window:
        category = "delayed_tracklet"
        group = "tracking"
    elif not is_stable:
        category = "fragmented_tracklet"
        group = "tracking"
    else:
        if pred_parent_id <= 0:
            category = "stable_orphan"
            group = "parentage"
        else:
            mapped = None
            for frame_idx in range(detection_frame, min(last_frame, detection_frame + 2) + 1):
                gt_mask = load_mask(frame_path(gt_dir, frame_idx, "man_track"))
                pred_mask = load_mask(frame_path(pred_dir, frame_idx, "mask"))
                if gt_mask is None or pred_mask is None:
                    continue
                best_parent = best_gt_for_pred(gt_mask, pred_mask, pred_parent_id)
                if best_parent is None:
                    continue
                if best_parent["pred_coverage"] >= parent_map_threshold:
                    mapped = int(best_parent["gt_label"])
                    break
            mapped_pred_parent_gt = mapped
            if mapped_pred_parent_gt == gt_parent_id:
                category = "stable_correct_parent"
                group = "correct"
            else:
                category = "stable_wrong_parent"
                group = "parentage"

    return {
        "category": category,
        "group": group,
        "gt_bud_id": gt_bud_id,
        "gt_parent_id": gt_parent_id,
        "gt_start": gt_start,
        "gt_end": gt_end,
        "onset_delay": int(onset_delay),
        "dominant_pred_id": int(dominant_pred_id),
        "dominant_pred_parent_id": int(pred_parent_id),
        "mapped_pred_parent_gt": mapped_pred_parent_gt,
        "support_frames": int(support_frames),
        "support_total": int(support_total),
        "dominance": float(dominance),
        "unique_pred_labels": int(len(unique_pred_labels)),
        "match_frames": [int(m["frame"]) for m in matches],
        "match_pred_labels": [int(m["pred_label"]) for m in matches],
    }


def summarize(results_by_video: dict[str, list[dict]]):
    overall_counts = Counter()
    group_counts = Counter()
    per_video = {}
    all_events = []
    for video_id, events in results_by_video.items():
        counts = Counter(event["category"] for event in events)
        groups = Counter(event["group"] for event in events)
        per_video[video_id] = {
            "num_events": len(events),
            "categories": dict(sorted(counts.items())),
            "groups": dict(sorted(groups.items())),
        }
        overall_counts.update(counts)
        group_counts.update(groups)
        all_events.extend(events)

    total_events = len(all_events)
    total_failures = total_events - group_counts.get("correct", 0)
    parentage_failures = group_counts.get("parentage", 0)
    tracking_failures = group_counts.get("tracking", 0)

    if parentage_failures >= tracking_failures * 1.25:
        recommendation = "global_parentage_optimization_looks_promising"
    elif tracking_failures >= parentage_failures * 1.25:
        recommendation = "tracking_persistence_improvement_looks_higher_priority"
    else:
        recommendation = "mixed_error_profile"

    return {
        "overall": {
            "num_events": total_events,
            "num_failures": total_failures,
            "categories": dict(sorted(overall_counts.items())),
            "groups": dict(sorted(group_counts.items())),
            "fixable_parentage_fraction_of_failures": (
                parentage_failures / total_failures if total_failures > 0 else 0.0
            ),
            "tracking_fraction_of_failures": (
                tracking_failures / total_failures if total_failures > 0 else 0.0
            ),
            "recommendation": recommendation,
        },
        "per_video": per_video,
        "events": all_events,
    }


def render_markdown(summary: dict, args: argparse.Namespace) -> str:
    overall = summary["overall"]
    lines = [
        "# Bud Event Diagnostic",
        "",
        "## Inputs",
        "",
        f"- GT root: `{args.gt_root}`",
        f"- Pred root: `{args.pred_root}`",
        f"- Videos: `{args.videos}`",
        f"- Analysis frames: `{args.analysis_frames}`",
        f"- Birth search window: `{args.birth_search_window}`",
        f"- Overlap threshold: `{args.overlap_threshold}`",
        "",
        "## Overall",
        "",
        f"- Number of GT bud events: `{overall['num_events']}`",
        f"- Number of failures: `{overall['num_failures']}`",
        f"- Recommendation: `{overall['recommendation']}`",
        f"- Parentage-fixable fraction of failures: `{overall['fixable_parentage_fraction_of_failures']:.3f}`",
        f"- Tracking-related fraction of failures: `{overall['tracking_fraction_of_failures']:.3f}`",
        "",
        "## Categories",
        "",
    ]
    for key, value in overall["categories"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Per Video", ""])
    for video_id, video_summary in summary["per_video"].items():
        lines.append(f"### {video_id}")
        lines.append("")
        lines.append(f"- Events: `{video_summary['num_events']}`")
        for key, value in video_summary["categories"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether yeast budding failures are mainly parentage or tracking problems."
    )
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--videos", default="12,13,14")
    parser.add_argument("--analysis-frames", type=int, default=5)
    parser.add_argument("--birth-search-window", type=int, default=3)
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    parser.add_argument("--stable-support", type=int, default=2)
    parser.add_argument("--stable-fraction", type=float, default=0.6)
    parser.add_argument("--parent-map-threshold", type=float, default=0.2)
    parser.add_argument("--save-json", default="")
    parser.add_argument("--save-markdown", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    video_ids = [video_id.strip() for video_id in args.videos.split(",") if video_id.strip()]

    results_by_video = {}
    for video_id in video_ids:
        gt_dir = gt_root / f"{video_id}_GT" / "TRA"
        pred_dir = pred_root / video_id
        gt_tracks = rows_to_dict(load_track_rows(gt_dir / "man_track.txt"))
        pred_tracks = rows_to_dict(load_track_rows(pred_dir / "res_track.txt"))
        gt_events = [row for row in gt_tracks.values() if row["parent"] > 0]
        gt_events.sort(key=lambda row: (row["start"], row["id"]))

        results = []
        for gt_event in gt_events:
            event_result = classify_event(
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
            event_result["video_id"] = video_id
            results.append(event_result)
        results_by_video[video_id] = results

    summary = summarize(results_by_video)
    markdown = render_markdown(summary, args)

    print(json.dumps(summary["overall"], indent=2))
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
    if args.save_markdown:
        out_path = Path(args.save_markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown)


if __name__ == "__main__":
    main()

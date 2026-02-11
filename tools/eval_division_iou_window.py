#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_mask(path: Path):
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def load_events_from_track(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    data = np.loadtxt(path, dtype=np.int32)
    if data.size == 0:
        return []
    if data.ndim == 1:
        data = data.reshape(1, -1)
    events = []
    for row in data:
        obj_id, start_frame, _end_frame, parent_id = row.tolist()
        if parent_id > 0:
            events.append((int(parent_id), int(start_frame)))
    return events



def best_iou_label(gt_mask, pred_mask, pred_label, iou_thresh):
    if gt_mask is None or pred_mask is None:
        return None
    if pred_label == 0:
        return None
    pred_bin = pred_mask == pred_label
    if not pred_bin.any():
        return None
    best_label = None
    best_iou = 0.0
    for gt_label in np.unique(gt_mask):
        if gt_label == 0:
            continue
        gt_bin = gt_mask == gt_label
        inter = np.logical_and(pred_bin, gt_bin).sum()
        if inter == 0:
            continue
        union = np.logical_or(pred_bin, gt_bin).sum()
        if union == 0:
            continue
        iou = inter / union
        if iou > best_iou:
            best_iou = iou
            best_label = int(gt_label)
    if best_iou < iou_thresh:
        return None
    return best_label


def match_predictions_to_gt(gt_events, pred_events, gt_mask_dir, pred_mask_dir,
                            gt_mask_prefix, temporal_tolerance, iou_thresh, ):
    gt_by_mother = {}
    for mother_id, frame_gt in gt_events:
        gt_by_mother.setdefault(mother_id, []).append(frame_gt)

    matched_gt = set()
    tp = 0
    fp = 0

    for pred_parent_id, pred_frame in pred_events:
        gt_mask_path = gt_mask_dir / f"{gt_mask_prefix}{pred_frame:03d}.tif"
        pred_mask_path = pred_mask_dir / f"mask{pred_frame:03d}.tif"
        gt_mask = load_mask(gt_mask_path)
        pred_mask = load_mask(pred_mask_path)
        gt_mother = best_iou_label(gt_mask, pred_mask, pred_parent_id, iou_thresh)
        if gt_mother is None:
            fp += 1
            continue
        candidates = gt_by_mother.get(gt_mother, [])
        if not candidates:
            fp += 1
            continue
        # pick closest unmatched GT event within the time window
        best = None
        best_dist = None
        for frame_gt in candidates:
            if (gt_mother, frame_gt) in matched_gt:
                continue
            if abs(frame_gt - pred_frame) <= temporal_tolerance:
                dist = abs(frame_gt - pred_frame)
                if best is None or dist < best_dist:
                    best = frame_gt
                    best_dist = dist
        if best is None:
            fp += 1
            continue
        matched_gt.add((gt_mother, best))
        tp += 1

    fn = len(gt_events) - len(matched_gt)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision,3),
        "recall": round(recall,3),
        "f1": round(f1,3),
    }
def eval_division_by_video(gt_video_dir: Path, pred_video_dir: Path, temporal_tolerance: int, iou_thresh: float, gt_mask_prefix: str = "man_track"):
    """
    Evaluate division events for a single video.

    Parameters
    ----------
    gt_video_dir : Path
        Path to GT video directory (e.g. .../12_GT/TRA)
    pred_video_dir : Path
        Path to prediction directory (e.g. .../12)
    temporal_tolerance : int
        Temporal tolerance (in frames)
    iou_thresh : float
        IoU threshold for mother ID matching
    gt_mask_prefix : str
        Prefix for GT masks (default: 'man_track')

    Returns
    -------
    dict
        Metrics dictionary with tp / fp / fn / precision / recall / f1
    """

    gt_events = load_events_from_track(gt_video_dir / "man_track.txt")
    pred_events = load_events_from_track(pred_video_dir / "summary" / "res_track.txt")

    metrics_by_video = match_predictions_to_gt(
        gt_events=gt_events,
        pred_events=pred_events,
        gt_mask_dir=gt_video_dir,
        pred_mask_dir=pred_video_dir,
        gt_mask_prefix=gt_mask_prefix,
        temporal_tolerance=temporal_tolerance,
        iou_thresh=iou_thresh,
    )

    return metrics_by_video

def eval_division_all_video(gt_root, pred_root, video_ids, temporal_tolerance, iou_thresh, gt_mask_prefix="man_track"):
    """
    Evaluate division events over multiple videos.

    Returns
    -------
    dict
        Per-video metrics + overall aggregated metrics
    """

    gt_root = Path(gt_root)
    pred_root = Path(pred_root)

    results_all_videos = {}
    total = {"tp": 0, "fp": 0, "fn": 0}

    for vid in video_ids:
        gt_video_dir = gt_root / f"{vid}_GT" / "TRA"
        pred_video_dir = pred_root / vid

        metrics = eval_division_by_video(
            gt_video_dir=gt_video_dir,
            pred_video_dir=pred_video_dir,
            temporal_tolerance=temporal_tolerance,
            iou_thresh=iou_thresh,
            gt_mask_prefix=gt_mask_prefix,
        )

        results_all_videos[vid] = metrics
        total["tp"] += metrics["tp"]
        total["fp"] += metrics["fp"]
        total["fn"] += metrics["fn"]

    precision = total["tp"] / (total["tp"] + total["fp"] + 1e-12)
    recall = total["tp"] / (total["tp"] + total["fn"] + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    results_all_videos["overall"] = {
        "tp": total["tp"],
        "fp": total["fp"],
        "fn": total["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return results_all_videos



if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate division events with IoU-based ID matching and time window.")
#     parser.add_argument("--gt_root", required=True)
#     parser.add_argument("--pred_root", required=True)
#     parser.add_argument("--videos", default="12,13,14")
#     parser.add_argument("--delay", type=int, default=3)
#     parser.add_argument("--iou_thresh", type=float, default=0.5)
#     parser.add_argument("--save_json",  default= "")
#     args = parser.parse_args()
#     video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
#     results = eval_division_all_video(args.gt_root, args.pred_root, video_ids, args.delay,
#                                      args.iou_thresh, gt_mask_prefix="man_track")
#     print(f"For videos {video_ids=}")
#     print(json.dumps(results, indent=2))
#
#     if args.save_json:
#         out_path = Path(args.save_json)
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         out_path.write_text(json.dumps(results, indent=2))


# python tools/eval_division_iou_window.py --gt_root /home/hcourtei/Projects/Cell_proj/data/moma_N_1_checked/moma/val/CTC --pred_root /home/hcourtei/Projects/Cell_proj/CellSam2Gilles/results/model:moma_N3_checked_v100/data_vers:moma8

    gt_video_dir = Path("/home/hcourtei/Projects/Cell_proj/data/moma_N_0_checked/moma/val/CTC/12_GT/TRA")
    pred_video_dir = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/eval_model/model:moma_N0_checked_v100/data_vers:moma_N0_checked/12")
    temporal_tolerance=3
    iou_thresh=0.5
    print(f"{temporal_tolerance=} {iou_thresh=}")
    metrics_by_video = eval_division_by_video(
        gt_video_dir,
        pred_video_dir,
        temporal_tolerance=3,
        iou_thresh=0.5,
        gt_mask_prefix="man_track",
    )
    #
    print(json.dumps(metrics_by_video, indent=2))


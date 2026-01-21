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


def load_events_from_man_track(path: Path):
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


def load_events_from_res_track(path: Path):
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


def match_predictions_to_gt(
    gt_events,
    pred_events,
    gt_mask_dir,
    pred_mask_dir,
    gt_mask_prefix,
    window,
    iou_thresh,
):
    gt_by_mother = {}
    for mother_id, frame_gt in gt_events:
        gt_by_mother.setdefault(mother_id, []).append(frame_gt)

    matched_gt = set()
    tp = 0
    fp = 0

    for pred_parent_id, pred_frame in pred_events:
        gt_mask_path = gt_mask_dir / f\"{gt_mask_prefix}{pred_frame:03d}.tif\"
        pred_mask_path = pred_mask_dir / f\"mask{pred_frame:03d}.tif\"
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
            if abs(frame_gt - pred_frame) <= window:
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
        \"tp\": tp,
        \"fp\": fp,
        \"fn\": fn,
        \"precision\": precision,
        \"recall\": recall,
        \"f1\": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description=\"Evaluate division events with IoU-based ID matching and time window.\"
    )
    parser.add_argument(\"--gt-root\", required=True)
    parser.add_argument(\"--pred-root\", required=True)
    parser.add_argument(\"--videos\", default=\"12,13,14\")
    parser.add_argument(\"--window\", type=int, default=3)
    parser.add_argument(\"--iou-thresh\", type=float, default=0.5)
    parser.add_argument(\"--gt-mask-prefix\", default=\"man_track\")
    parser.add_argument(\"--save-json\", default=\"\")
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    video_ids = [v.strip() for v in args.videos.split(\",\") if v.strip()]

    results = {}
    total = {\"tp\": 0, \"fp\": 0, \"fn\": 0}

    for vid in video_ids:
        gt_dir = gt_root / f\"{vid}_GT\" / \"TRA\"
        pred_dir = pred_root / vid
        gt_events = load_events_from_man_track(gt_dir / \"man_track.txt\")
        pred_events = load_events_from_res_track(pred_dir / \"res_track.txt\")
        metrics = match_predictions_to_gt(
            gt_events,
            pred_events,
            gt_dir,
            pred_dir,
            args.gt_mask_prefix,
            args.window,
            args.iou_thresh,
        )
        results[vid] = metrics
        total[\"tp\"] += metrics[\"tp\"]
        total[\"fp\"] += metrics[\"fp\"]
        total[\"fn\"] += metrics[\"fn\"]

    precision = total[\"tp\"] / (total[\"tp\"] + total[\"fp\"] + 1e-12)
    recall = total[\"tp\"] / (total[\"tp\"] + total[\"fn\"] + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    results[\"overall\"] = {
        \"tp\": total[\"tp\"],
        \"fp\": total[\"fp\"],
        \"fn\": total[\"fn\"],
        \"precision\": precision,
        \"recall\": recall,
        \"f1\": f1,
    }

    print(json.dumps(results, indent=2))
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))


if __name__ == \"__main__\":
    main()

#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_list(arg, cast=float):
    return [cast(x) for x in arg.split(",") if x.strip() != ""]


def load_events(path: Path):
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


def compute_metrics(gt_events, pred_events):
    gt_set = set(gt_events)
    pred_set = set(pred_events)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_loose_metrics(gt_events, pred_events):
    gt_frames = set(frame for _, frame in gt_events)
    pred_frames = set(frame for _, frame in pred_events)
    tp = len(gt_frames & pred_frames)
    fp = len(pred_frames - gt_frames)
    fn = len(gt_frames - pred_frames)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_inference(video_dir, out_root, args, thresholds):
    res_dir = out_root / thresholds["key"] / video_dir.name
    res_track = res_dir / "res_track.txt"
    if res_track.exists():
        return
    res_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "inference/track_cells.py",
        "--video_path",
        str(video_dir),
        "--res_path",
        str(out_root / thresholds["key"]),
        "--model_name",
        args.model_name,
        "--checkpoint_num",
        str(args.checkpoint_num),
        "--pred_iou_thresh",
        str(thresholds["pred_iou_thresh"]),
        "--obj_score_thresh",
        str(thresholds["obj_score_thresh"]),
        "--div_obj_score_thresh",
        str(thresholds["div_obj_score_thresh"]),
        "--box_nms_thresh",
        str(thresholds["box_nms_thresh"]),
        "--min_mask_area",
        str(thresholds["min_mask_area"]),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Sweep inference thresholds and evaluate division recall/FP.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="val/CTC")
    parser.add_argument("--videos", default="12,13,14")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-num", type=int, default=12)
    parser.add_argument("--out-root", default="results/threshold_sweep_val")
    parser.add_argument("--pred-iou-list", default="0.2,0.3,0.5")
    parser.add_argument("--obj-score-list", default="0.0")
    parser.add_argument("--div-score-list", default="-6,-4,-2")
    parser.add_argument("--box-nms-list", default="0.7")
    parser.add_argument("--min-area-list", default="10")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    video_dirs = [split_dir / vid for vid in video_ids]

    pred_iou_list = parse_list(args.pred_iou_list, float)
    obj_score_list = parse_list(args.obj_score_list, float)
    div_score_list = parse_list(args.div_score_list, float)
    box_nms_list = parse_list(args.box_nms_list, float)
    min_area_list = parse_list(args.min_area_list, int)

    rows = []
    combos = itertools.product(pred_iou_list, obj_score_list, div_score_list, box_nms_list, min_area_list)
    for pred_iou, obj_score, div_score, box_nms, min_area in combos:
        key = (f"piou{pred_iou}_obj{obj_score}_div{div_score}_nms{box_nms}_min{min_area}")
        thresholds = {
            "key": key,
            "pred_iou_thresh": pred_iou,
            "obj_score_thresh": obj_score,
            "div_obj_score_thresh": div_score,
            "box_nms_thresh": box_nms,
            "min_mask_area": min_area,
        }
        print(f"==> Running {key}")
        for video_dir in video_dirs:
            run_inference(video_dir, out_root, args, thresholds)

        gt_events_all = []
        pred_events_all = []
        for video_dir in video_dirs:
            gt_path = split_dir / f"{video_dir.name}_GT/TRA/man_track.txt"
            pred_path = out_root / key / video_dir.name / "res_track.txt"
            gt_events_all.extend(load_events(gt_path))
            pred_events_all.extend(load_events(pred_path))

        strict = compute_metrics(gt_events_all, pred_events_all)
        loose = compute_loose_metrics(gt_events_all, pred_events_all)
        row = {
            "key": key,
            "pred_iou": pred_iou,
            "obj_score": obj_score,
            "div_score": div_score,
            "box_nms": box_nms,
            "min_area": min_area,
            "tp": strict["tp"],
            "fp": strict["fp"],
            "fn": strict["fn"],
            "precision": strict["precision"],
            "recall": strict["recall"],
            "f1": strict["f1"],
            "loose_precision": loose["precision"],
            "loose_recall": loose["recall"],
            "loose_f1": loose["f1"],
        }
        rows.append(row)

    out_path = out_root / "sweep_metrics.tsv"
    with out_path.open("w") as f:
        header = [
            "key",
            "pred_iou",
            "obj_score",
            "div_score",
            "box_nms",
            "min_area",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "loose_precision",
            "loose_recall",
            "loose_f1",
        ]
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[h]) for h in header) + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

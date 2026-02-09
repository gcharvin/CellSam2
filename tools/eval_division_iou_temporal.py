#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def load_mask(path: Path):
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def load_division_events(track_file: Path):
    """
    Return list of (mother_id, start_frame)
    """
    if not track_file.exists():
        return []

    data = np.loadtxt(track_file, dtype=int)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    events = []
    for obj_id, frame_start, _, parent_id in data:
        if parent_id > 0:
            events.append((parent_id, frame_start))
    return events


# ------------------------------------------------------------
# Masks & overlap
# ------------------------------------------------------------
def binary_mask(mask, label):
    if mask is None:
        return None
    m = (mask == label)
    return m if m.any() else None


def overlap_ratio(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / union

# ------------------------------------------------------------
# Temporal + spatial matching
# ------------------------------------------------------------
def find_best_future_overlap(
    gt_mask_dir,
    pred_mask_dir,
    gt_label,
    pred_label,
    gt_frame,
    max_frame_offset,
):
    best = 0.0

    for dt in range(max_frame_offset + 1):
        t = gt_frame + dt

        gt_mask = load_mask(gt_mask_dir / f"man_track{t:03d}.tif")
        pred_mask = load_mask(pred_mask_dir / f"mask{t:03d}.tif")
        if gt_mask is None or pred_mask is None:
            continue

        gt_bin = binary_mask(gt_mask, gt_label)
        pred_bin = binary_mask(pred_mask, pred_label)
        if gt_bin is None or pred_bin is None:
            continue

        best = max(best, overlap_ratio(gt_bin, pred_bin))

    return best

def eval_division_by_video(
    gt_video_dir: Path,
    pred_video_dir: Path,
    temporal_tolerance: int = 3,
    max_future_frame_offset: int = 2,
    min_overlap_ratio: float = 0.3,
    verbose: bool =False
):
    gt_events = load_division_events(gt_video_dir / "man_track.txt")
    pred_events = load_division_events(pred_video_dir / "summary" / "res_track.txt")

    matched_gt = set()
    tp = fp = 0

    for pred_mother, pred_start_frame in pred_events:
        best_match = None
        best_overlap_score = 0.0

        for gt_mother, gt_start_frame in gt_events:
            if (gt_mother, gt_start_frame) in matched_gt:
                continue
            if abs(pred_start_frame - gt_start_frame) > temporal_tolerance:
                continue

            overlap_score = find_best_future_overlap(
                gt_video_dir,
                pred_video_dir,
                gt_label=gt_mother,
                pred_label=pred_mother,
                gt_frame=gt_start_frame,
                max_frame_offset=max_future_frame_offset,
            )

            if overlap_score > best_overlap_score:
                best_overlap_score = overlap_score
                best_match = (gt_mother, gt_start_frame)

        if best_match is not None and best_overlap_score >= min_overlap_ratio:
            gt_m, gt_start_frame = best_match
            if verbose:
                print(
                f"pred_start_frame={pred_start_frame:3d} | "
                f"gt_start_frame={gt_start_frame:3d} | "
                f"Δt={pred_start_frame - gt_start_frame:+d} | "
                f"score={best_overlap_score:.3f}"
            )
            tp += 1
            matched_gt.add(best_match)
        else:
            fp += 1

    fn = len(gt_events) - len(matched_gt)

    precision = round(tp / (tp + fp + 1e-12), 3)
    recall = round(tp / (tp + fn + 1e-12), 3)
    f1 = round(2 * precision * recall / (precision + recall + 1e-12), 3)

    metrics = {"gt_video_dir": str(gt_video_dir),
               "pred_video_dir": str(pred_video_dir),
               "temporal_tolerance": temporal_tolerance,
               "max_future_frame_offset": max_future_frame_offset,
               "min_overlap_ratio": min_overlap_ratio,
               "metrics": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
               }

    return metrics


def count_parentless_predictions(res_track_path):
    with open(res_track_path, 'r') as f:
        lines = f.readlines()

    parentless_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4 and int(parts[3]) == 0:
            parentless_count += 1

    return parentless_count

def load_track_events(track_file):
    events = []
    with open(track_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            obj_id, start, end, parent = map(int, parts)
            events.append((obj_id, start, end, parent))

    return events

def build_lineage(track_events):
    lineage = defaultdict(list)
    for obj_id, start, end, parent in track_events:
        if parent != 0:
            lineage[parent].append(obj_id)

    return lineage

def compare_lineages(gt_lineage, pred_lineage):
    correct_lineages = 0
    total_lineages = len(gt_lineage)

    for parent in gt_lineage:
        if parent in pred_lineage:
            gt_children = set(gt_lineage[parent])
            pred_children = set(pred_lineage[parent])

            if gt_children == pred_children:
                correct_lineages += 1

    precision = correct_lineages / len(pred_lineage) if pred_lineage else 0
    recall = correct_lineages / total_lineages if total_lineages else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1


if __name__ == '__main__':
    temporal_tolerance = 3        # tolérance temporelle GT ↔ préd
    max_future_frame_offset = 4   # apparition tardive du bourgeon
    min_overlap_ratio = 0.2

    gt_video_dir = Path("/home/hcourtei/Projects/Cell_proj/data/moma_N_0_checked/moma/val/CTC/12_GT/TRA")
    pred_video_dir = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/eval_model/model:moma_N0_checked_v100/data_vers:moma_N0_checked/12")

    metrics_by_video = eval_division_by_video(gt_video_dir, pred_video_dir,temporal_tolerance,  max_future_frame_offset, min_overlap_ratio,verbose=True)

    print(json.dumps(metrics_by_video, indent=2))

    res_track_path = pred_video_dir / "summary" / "res_track.txt"
    parentless_predictions = count_parentless_predictions(res_track_path)
    print(f"Nombre de bourgeons sans parents : {parentless_predictions}")

    gt_track_path =  pred_video_dir / "summary" / "man_track.txt"

    gt_events = load_track_events(gt_track_path)
    pred_events = load_track_events(res_track_path)

    # Construire les lignées cellulaires
    gt_lineage = build_lineage(gt_events)
    pred_lineage = build_lineage(pred_events)

    precision, recall, f1 = compare_lineages(gt_lineage, pred_lineage)
    print(f"Précision des lignées : {precision:.3f}")
    print(f"Rappel des lignées : {recall:.3f}")
    print(f"F1 des lignées : {f1:.3f}")
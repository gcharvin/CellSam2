#!/usr/bin/env python3
import json
from pathlib import Path
import numpy
import numpy as np
import cv2

def load_mask(path: Path):
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def load_full_division_events(track_file: Path):
    """
    Retourne une liste d'événements de division complets : (mother_id, bud_id, start_frame)
    """
    if not track_file.exists():
        return []

    data = np.loadtxt(track_file, dtype=int)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    events = []
    for obj_id, frame_start, _, parent_id in data:
        if parent_id > 0:  # C'est un bourgeon
            events.append((parent_id, obj_id, frame_start))  # (mère, bourgeon, frame)
    return events


# ------------------------------------------------------------
# Masks & iou
# ------------------------------------------------------------
def binary_mask(mask, label):
    if mask is None:
        return None
    m = (mask == label)
    return m if m.any() else None


def compute_iou(a : numpy.bool, b: numpy.bool):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / union


# ------------------------------------------------------------
# Temporal + spatial matching
# ------------------------------------------------------------
def find_best_future_overlap(gt_mask_dir, pred_mask_dir, gt_label, pred_label, gt_frame, max_frame_offset):
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

        best = max(best, compute_iou(gt_bin, pred_bin))

    return best


def count_parentless_predictions(res_track_path):
    with open(res_track_path, 'r') as f:
        lines = f.readlines()

    parentless_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4 and int(parts[3]) == 0:
            parentless_count += 1

    return parentless_count

def match_division_events(gt_events, pred_events, gt_mask_dir, pred_mask_dir, temporal_tolerance, max_future_frame_offset, iou_thresh_mother, iou_thresh_bud):
    matched_gt = set()
    tp = 0
    time_errors = []
    iou_mother_scores = []
    iou_bud_scores = []

    for pred_mother_id, pred_bud_id, pred_frame in pred_events:
        best_match = None
        best_mother_score = 0.0
        best_bud_score = 0.0

        for gt_mother_id, gt_bud_id, gt_frame in gt_events:
            if (gt_mother_id, gt_bud_id, gt_frame) in matched_gt:
                continue
            if abs(pred_frame - gt_frame) > temporal_tolerance:
                continue

            # Calculer l'IoU pour la mère
            mother_score = find_best_future_overlap(
                gt_mask_dir, pred_mask_dir,
                gt_label=gt_mother_id, pred_label=pred_mother_id,
                gt_frame=gt_frame, max_frame_offset=max_future_frame_offset
            )

            # Calculer l'IoU pour le bourgeon
            bud_score = find_best_future_overlap(
                gt_mask_dir, pred_mask_dir,
                gt_label=gt_bud_id, pred_label=pred_bud_id,
                gt_frame=gt_frame, max_frame_offset=max_future_frame_offset
            )

            if mother_score > best_mother_score and bud_score > best_bud_score:
                best_mother_score = mother_score
                best_bud_score = bud_score
                best_match = (gt_mother_id, gt_bud_id, gt_frame)

        if best_match is not None and best_mother_score >= iou_thresh_mother and best_bud_score >= iou_thresh_bud:
            tp += 1
            matched_gt.add(best_match)
            # Ajouter les erreurs temporelles et les scores IoU
            gt_mother_id, gt_bud_id, gt_frame = best_match
            time_errors.append(abs(pred_frame - gt_frame))
            iou_mother_scores.append(best_mother_score)
            iou_bud_scores.append(best_bud_score)

    fp = len(pred_events) - tp
    fn = len(gt_events) - len(matched_gt)

    # Calculer les métriques de localisation
    avg_time_error = round(np.mean(time_errors), 3) if time_errors else 0.0
    std_time_error = round(np.std(time_errors), 3) if time_errors else 0.0
    avg_iou_mother = round(np.mean(iou_mother_scores), 3) if iou_mother_scores else 0.0
    avg_iou_bud = round(np.mean(iou_bud_scores), 3) if iou_bud_scores else 0.0

    return tp, fp, fn, avg_time_error, std_time_error, avg_iou_mother, avg_iou_bud



def eval_division_by_video(
    gt_video_dir: Path,
    pred_video_dir: Path,
    temporal_tolerance: int = 3,
    max_future_frame_offset: int = 2,
    iou_thresh_mother: float = 0.5,
    iou_thresh_bud: float = 0.3,
):
    # Charger les événements complets (mère + bourgeon + frame)
    gt_events = load_full_division_events(gt_video_dir / "man_track.txt")
    pred_events = load_full_division_events(pred_video_dir / "summary" / "res_track.txt")

    # Compter les bourgeons sans parents
    parentless_pred = count_parentless_predictions(pred_video_dir / "summary" / "res_track.txt")
    # Nombre total de bourgeons prédits
    total_pred_buds = len(pred_events)

    # Calcul du ratio des bourgeons sans parents
    parentless_ratio = round(parentless_pred / total_pred_buds, 3) if total_pred_buds > 0 else 0.0


    # Appariement spatio-temporel
    # Appariement spatio-temporel
    tp, fp, fn, avg_time_error, std_time_error, avg_iou_mother, avg_iou_bud = match_division_events(
        gt_events, pred_events,
        gt_video_dir, pred_video_dir,
        temporal_tolerance, max_future_frame_offset, iou_thresh_mother, iou_thresh_bud
    )

    # Calcul des métriques
    precision, recall, f1 = compute_metrics(tp, fp, fn)

    metrics = {
        "gt_video_dir": str(gt_video_dir),
        "pred_video_dir": str(pred_video_dir),
        "temporal_tolerance": temporal_tolerance,
        "max_future_frame_offset": max_future_frame_offset,
        "iou_thresh_mother": iou_thresh_mother,
        "iou_thresh_bud": iou_thresh_bud,
        "metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "parentless_pred": parentless_pred,
            "parentless_ratio": parentless_ratio,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_time_error": avg_time_error,
            "std_time_error": std_time_error,
            "avg_iou_mother": avg_iou_mother,
            "avg_iou_bud": avg_iou_bud,
        }
    }
    return metrics


def compute_metrics(tp: int, fp: int, fn: int) -> tuple:

    precision = round(tp / (tp + fp + 1e-12), 3)
    recall = round(tp / (tp + fn + 1e-12), 3)
    f1 = round(2 * precision * recall / (precision + recall + 1e-12), 3) if (precision + recall) > 0 else 0.0

    return  precision, recall, f1


if __name__ == '__main__':
    temporal_tolerance = 3        # tolérance temporelle GT ↔ préd
    max_future_frame_offset = 3   # apparition tardive du bourgeon
    iou_thresh_mother = 0.5         # seuil IoU pour les mères
    iou_thresh_bud = 0.2           # seuil IoU pour les bourgeons

    gt_video_dir = Path("/home/hcourtei/Projects/Cell_proj/data/moma_N_0_checked/moma/val/CTC/12_GT/TRA")
    pred_video_dir = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/eval_model/model:moma_N0_checked_v100/data_vers:moma_N0_checked/12")


    metrics_by_video = eval_division_by_video(
        gt_video_dir,
        pred_video_dir,
        temporal_tolerance,
        max_future_frame_offset,
        iou_thresh_mother,
        iou_thresh_bud,
    )

    print("metrics_by_video")
    print(json.dumps(metrics_by_video, indent=2))

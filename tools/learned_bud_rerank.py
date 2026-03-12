#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from scipy.optimize import minimize

import hydra


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tools.diagnose_bud_events import (
    best_gt_for_pred,
    classify_event,
    frame_path,
    load_track_rows,
    rows_to_dict,
)
from tools.online_bud_parentage import (
    Candidate,
    ObjStats,
    TrackInfo,
    _build_global_candidates,
    _collect_frame_cache,
    _contact_score,
    _load_res_track,
    _load_track_infos,
    _mother_radius,
    _neck_score,
    _solve_global_ilp,
    _sorted_mask_files,
    _write_res_track,
)


FEATURE_NAMES = [
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
    "temporal_support",
    "contact_persistence",
    "neck_persistence",
    "attachment_persistence",
    "dist_stability",
    "separation_trend",
    "framewise_best_fraction",
    "sam2_cosine_mean",
    "sam2_cosine_min",
    "sam2_cosine_trend",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a learned bud->mother reranker with spatio-temporal features."
    )
    parser.add_argument("--train-gt-root", required=True)
    parser.add_argument("--train-pred-root", required=True)
    parser.add_argument("--apply-gt-root", required=True)
    parser.add_argument("--apply-pred-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-videos", default="")
    parser.add_argument("--apply-videos", default="")
    parser.add_argument("--train-image-root", default="")
    parser.add_argument("--apply-image-root", default="")
    parser.add_argument("--sam2-model-name", default="")
    parser.add_argument("--sam2-checkpoint-num", type=int, default=None)
    parser.add_argument("--sam2-device", default="")
    parser.add_argument("--objective", choices=["classification", "pairwise"], default="pairwise")
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
    parser.add_argument("--w-contact", type=float, default=0.0)
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
    parser.add_argument("--analysis-frames", type=int, default=5)
    parser.add_argument("--birth-search-window", type=int, default=3)
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    parser.add_argument("--stable-support", type=int, default=2)
    parser.add_argument("--stable-fraction", type=float, default=0.6)
    parser.add_argument("--parent-map-threshold", type=float, default=0.2)
    parser.add_argument("--reg-strength", type=float, default=0.1)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def list_videos(root: Path, explicit: str) -> List[str]:
    if explicit.strip():
        return [item.strip() for item in explicit.split(",") if item.strip()]
    return sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.endswith("_GT"))


def setup_hydra(model_name: str) -> Tuple[str, str]:
    config_path = f"../sam2_logs/{model_name}"
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=config_path)
    return config_path, "config_resolved"


def normalize_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image.astype(np.float32)
        min_val = float(image.min())
        max_val = float(image.max())
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            min_val = float(image.min())
            max_val = float(image.max())
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape {image.shape}")


def frame_image_path(image_dir: Path, frame_idx: int) -> Path:
    return image_dir / f"t{frame_idx:03d}.tif"


class SAM2EmbeddingExtractor:
    def __init__(self, model_name: str, checkpoint_num: int | None, device: str | None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _, config_name = setup_hydra(model_name)
        if checkpoint_num is None:
            checkpoint = ROOT / "sam2_logs" / model_name / "checkpoints" / "checkpoint.pt"
        else:
            checkpoint = ROOT / "sam2_logs" / model_name / "checkpoints" / f"checkpoint_{checkpoint_num}.pt"
        self.model = build_sam2(config_name, str(checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.frame_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def get_frame_feature_map(self, image_dir: Path, frame_idx: int) -> np.ndarray | None:
        key = (str(image_dir), frame_idx)
        if key in self.frame_cache:
            return self.frame_cache[key]
        image = cv2.imread(str(frame_image_path(image_dir, frame_idx)), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        image_rgb = normalize_to_rgb(image)
        self.predictor._transforms._set_hw_params(image_rgb, self.model.image_size)
        input_image = self.predictor._transforms(image_rgb)[None, ...].to(self.device)
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.predictor._bb_feat_sizes[::-1])
        ][::-1]
        feat = feats[-1][0].detach().cpu().numpy()
        self.frame_cache[key] = feat
        return feat

    def pool_object_embedding(self, image_dir: Path, frame_idx: int, object_mask: np.ndarray) -> np.ndarray | None:
        feat = self.get_frame_feature_map(image_dir, frame_idx)
        if feat is None or not object_mask.any():
            return None
        _, feat_h, feat_w = feat.shape
        resized_mask = cv2.resize(
            object_mask.astype(np.uint8),
            (feat_w, feat_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if not resized_mask.any():
            return None
        vec = feat[:, resized_mask].mean(axis=1)
        norm = np.linalg.norm(vec)
        if norm <= 1e-8:
            return None
        return vec / norm


def resolve_image_dir(image_root: Path | None, video_id: str) -> Path | None:
    if image_root is None:
        return None
    candidate = image_root / video_id
    return candidate if candidate.exists() else None


def build_seq_context(seq_dir: Path, args: argparse.Namespace) -> dict:
    mask_files = _sorted_mask_files(seq_dir)
    track_infos = _load_track_infos(seq_dir / "res_track.txt")
    masks, stats_cache, bin_masks = _collect_frame_cache(mask_files)
    bud_ids, candidates = _build_global_candidates(
        track_infos=track_infos,
        masks=masks,
        stats_cache=stats_cache,
        bin_masks=bin_masks,
        refractory_frames=args.refractory,
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
    by_bud: Dict[int, List[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_bud[cand.bud_id].append(cand)
    for cand_list in by_bud.values():
        cand_list.sort(key=lambda item: item.score, reverse=True)
    return {
        "seq_dir": seq_dir,
        "track_infos": track_infos,
        "bud_ids": bud_ids,
        "candidates_by_bud": by_bud,
        "masks": masks,
        "stats_cache": stats_cache,
        "bin_masks": bin_masks,
        "feature_cache": {},
    }


def map_parent_to_gt(seq_dir: Path, gt_dir: Path, frame_idx: int, pred_parent_id: int, threshold: float):
    mapped = None
    for offset in range(0, 3):
        gt_mask = cv2.imread(str(frame_path(gt_dir, frame_idx + offset, "man_track")), cv2.IMREAD_UNCHANGED)
        pred_mask = cv2.imread(str(seq_dir / f"mask{frame_idx + offset:03d}.tif"), cv2.IMREAD_UNCHANGED)
        if gt_mask is None or pred_mask is None:
            continue
        best_parent = best_gt_for_pred(gt_mask, pred_mask, pred_parent_id)
        if best_parent is None:
            continue
        if best_parent["pred_coverage"] >= threshold:
            mapped = int(best_parent["gt_label"])
            break
    return mapped


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    denom = max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8)
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def normalized_distance(bud_stats: ObjStats, mother_stats: ObjStats) -> float:
    radius = _mother_radius(mother_stats.area)
    if radius <= 1e-6:
        return 1.0
    return math.hypot(bud_stats.cx - mother_stats.cx, bud_stats.cy - mother_stats.cy) / radius


def compute_temporal_features(cand: Candidate, seq_context: dict, args: argparse.Namespace) -> dict:
    track_infos: Dict[int, TrackInfo] = seq_context["track_infos"]
    stats_cache = seq_context["stats_cache"]
    bin_masks = seq_context["bin_masks"]
    bud_track = track_infos[cand.bud_id]
    end_frame = min(bud_track.end, bud_track.start + args.first_frames - 1)
    frame_range = range(bud_track.start, end_frame + 1)

    norm_dists: List[float] = []
    contact_series: List[float] = []
    neck_series: List[float] = []
    framewise_best_flags: List[float] = []

    for frame_idx in frame_range:
        frame_stats = stats_cache.get(frame_idx, {})
        bud_stats = frame_stats.get(cand.bud_id)
        mother_stats = frame_stats.get(cand.mother_id)
        bud_mask = bin_masks.get(frame_idx, {}).get(cand.bud_id)
        mother_mask = bin_masks.get(frame_idx, {}).get(cand.mother_id)
        if bud_stats is None or mother_stats is None or bud_mask is None or mother_mask is None:
            continue

        norm_dist = normalized_distance(bud_stats, mother_stats)
        norm_dists.append(norm_dist)
        contact_series.append(_contact_score(bud_mask, mother_mask, args.interface_radius))
        neck_series.append(
            _neck_score(
                bud_mask,
                mother_mask,
                args.interface_radius,
                args.use_border_neck,
            )
        )

        best_norm_dist = norm_dist
        for other_id, other_track in track_infos.items():
            if other_id in (cand.bud_id, cand.mother_id):
                continue
            if other_track.start > bud_track.start - args.min_mother_age:
                continue
            if other_track.end < frame_idx:
                continue
            other_stats = frame_stats.get(other_id)
            if other_stats is None:
                continue
            best_norm_dist = min(best_norm_dist, normalized_distance(bud_stats, other_stats))
        framewise_best_flags.append(1.0 if abs(norm_dist - best_norm_dist) < 1e-6 else 0.0)

    if not norm_dists:
        return {
            "temporal_support": 0.0,
            "contact_persistence": 0.0,
            "neck_persistence": 0.0,
            "attachment_persistence": 0.0,
            "dist_stability": 0.0,
            "separation_trend": 0.0,
            "framewise_best_fraction": 0.0,
        }

    support = len(norm_dists)
    contact_arr = np.asarray(contact_series, dtype=float)
    neck_arr = np.asarray(neck_series, dtype=float)
    dist_arr = np.asarray(norm_dists, dtype=float)
    return {
        "temporal_support": float(np.clip(support / max(float(args.first_frames), 1.0), 0.0, 1.0)),
        "contact_persistence": float((contact_arr > 0.05).mean()),
        "neck_persistence": float((neck_arr > 0.10).mean()),
        "attachment_persistence": float(np.clip((0.5 * contact_arr + 0.5 * neck_arr).mean(), 0.0, 1.0)),
        "dist_stability": float(math.exp(-float(dist_arr.std()))),
        "separation_trend": float(np.clip(0.5 + 0.5 * (dist_arr[0] - dist_arr[-1]), 0.0, 1.0)),
        "framewise_best_fraction": float(np.mean(framewise_best_flags)),
    }


def compute_sam2_features(
    cand: Candidate,
    seq_context: dict,
    image_dir: Path | None,
    extractor: SAM2EmbeddingExtractor | None,
    args: argparse.Namespace,
) -> dict:
    if image_dir is None or extractor is None:
        return {
            "sam2_cosine_mean": 0.0,
            "sam2_cosine_min": 0.0,
            "sam2_cosine_trend": 0.0,
        }

    track_infos: Dict[int, TrackInfo] = seq_context["track_infos"]
    bin_masks = seq_context["bin_masks"]
    bud_track = track_infos[cand.bud_id]
    end_frame = min(bud_track.end, bud_track.start + args.first_frames - 1)
    frame_range = range(bud_track.start, end_frame + 1)
    cosine_values: List[float] = []
    for frame_idx in frame_range:
        bud_mask = bin_masks.get(frame_idx, {}).get(cand.bud_id)
        mother_mask = bin_masks.get(frame_idx, {}).get(cand.mother_id)
        if bud_mask is None or mother_mask is None:
            continue
        bud_embed = extractor.pool_object_embedding(image_dir, frame_idx, bud_mask)
        mother_embed = extractor.pool_object_embedding(image_dir, frame_idx, mother_mask)
        cosine_values.append(cosine_similarity(bud_embed, mother_embed))

    if not cosine_values:
        return {
            "sam2_cosine_mean": 0.0,
            "sam2_cosine_min": 0.0,
            "sam2_cosine_trend": 0.0,
        }

    return {
        "sam2_cosine_mean": float(np.mean(cosine_values)),
        "sam2_cosine_min": float(np.min(cosine_values)),
        "sam2_cosine_trend": float(cosine_values[-1] - cosine_values[0]) if len(cosine_values) > 1 else 0.0,
    }


def get_extra_features(
    cand: Candidate,
    seq_context: dict,
    image_dir: Path | None,
    extractor: SAM2EmbeddingExtractor | None,
    args: argparse.Namespace,
) -> dict:
    cache_key = (cand.bud_id, cand.mother_id)
    cache = seq_context["feature_cache"]
    if cache_key not in cache:
        temporal = compute_temporal_features(cand, seq_context, args)
        sam2 = compute_sam2_features(cand, seq_context, image_dir, extractor, args)
        cache[cache_key] = {**temporal, **sam2}
    return cache[cache_key]


def features_from_candidate(candidate: Candidate, proposal_parent: int, extra_features: dict) -> np.ndarray:
    return np.asarray(
        [
            candidate.score,
            candidate.dist,
            candidate.size_ratio,
            candidate.motion,
            candidate.contact,
            candidate.neck,
            candidate.prebud,
            candidate.angle,
            candidate.maturity,
            candidate.track_quality,
            candidate.margin,
            float(candidate.mother_age),
            1.0 if candidate.mother_id == proposal_parent else 0.0,
            extra_features["temporal_support"],
            extra_features["contact_persistence"],
            extra_features["neck_persistence"],
            extra_features["attachment_persistence"],
            extra_features["dist_stability"],
            extra_features["separation_trend"],
            extra_features["framewise_best_fraction"],
            extra_features["sam2_cosine_mean"],
            extra_features["sam2_cosine_min"],
            extra_features["sam2_cosine_trend"],
        ],
        dtype=np.float64,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_classification_model(x: np.ndarray, y: np.ndarray, reg_strength: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std

    pos = max(float(y.sum()), 1.0)
    neg = max(float((1 - y).sum()), 1.0)
    sample_weights = np.where(y > 0.5, 0.5 / pos, 0.5 / neg)

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        logits = z @ theta[:-1] + theta[-1]
        probs = sigmoid(logits)
        eps = 1e-9
        loss = -np.sum(sample_weights * (y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps)))
        loss += 0.5 * reg_strength * np.sum(theta[:-1] ** 2)
        error = (probs - y) * sample_weights
        grad_w = z.T @ error + reg_strength * theta[:-1]
        grad_b = np.sum(error)
        return float(loss), np.concatenate([grad_w, np.asarray([grad_b])])

    theta0 = np.zeros(z.shape[1] + 1, dtype=np.float64)
    result = minimize(
        fun=lambda th: objective(th)[0],
        x0=theta0,
        jac=lambda th: objective(th)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"classification fit failed: {result.message}")
    return mean, std, result.x


def fit_pairwise_model(rows: List[dict], x: np.ndarray, reg_strength: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std

    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row["video_id"], row["pred_bud_id"])].append(idx)

    diffs: List[np.ndarray] = []
    for indices in groups.values():
        pos_idx = [idx for idx in indices if rows[idx]["label"] == 1]
        neg_idx = [idx for idx in indices if rows[idx]["label"] == 0]
        for left_idx in pos_idx:
            for right_idx in neg_idx:
                diffs.append(z[left_idx] - z[right_idx])
    if not diffs:
        raise RuntimeError("No positive/negative candidate pairs available for pairwise ranking.")
    pair_x = np.stack(diffs, axis=0)

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        scores = pair_x @ theta[:-1]
        probs = sigmoid(scores)
        eps = 1e-9
        loss = -np.mean(np.log(probs + eps))
        loss += 0.5 * reg_strength * np.sum(theta[:-1] ** 2)
        grad_w = -(pair_x.T @ (1.0 - probs)) / pair_x.shape[0] + reg_strength * theta[:-1]
        return float(loss), np.concatenate([grad_w, np.asarray([0.0])])

    theta0 = np.zeros(z.shape[1] + 1, dtype=np.float64)
    result = minimize(
        fun=lambda th: objective(th)[0],
        x0=theta0,
        jac=lambda th: objective(th)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"pairwise fit failed: {result.message}")
    return mean, std, result.x, int(pair_x.shape[0])


def predict_prob(x: np.ndarray, mean: np.ndarray, std: np.ndarray, theta: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    logits = z @ theta[:-1] + theta[-1]
    return sigmoid(logits)


def build_training_set(
    train_gt_root: Path,
    train_pred_root: Path,
    train_image_root: Path | None,
    embedding_extractor: SAM2EmbeddingExtractor | None,
    args: argparse.Namespace,
):
    rows = []
    for video_id in list_videos(train_gt_root, args.train_videos):
        gt_dir = train_gt_root / f"{video_id}_GT" / "TRA"
        pred_dir = train_pred_root / video_id
        if not pred_dir.exists():
            continue
        image_dir = resolve_image_dir(train_image_root, video_id)
        pred_tracks = rows_to_dict(load_track_rows(pred_dir / "res_track.txt"))
        gt_tracks = rows_to_dict(load_track_rows(gt_dir / "man_track.txt"))
        seq_context = build_seq_context(pred_dir, args)
        candidates_by_bud = seq_context["candidates_by_bud"]
        seen_pairs = set()
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
            pred_bud_id = event.get("dominant_pred_id")
            if pred_bud_id is None or pred_bud_id not in candidates_by_bud:
                continue
            proposal_parent = pred_tracks.get(pred_bud_id, {}).get("parent", 0)
            for candidate in candidates_by_bud[pred_bud_id]:
                key = (video_id, pred_bud_id, candidate.mother_id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                mapped_parent = map_parent_to_gt(
                    seq_dir=pred_dir,
                    gt_dir=gt_dir,
                    frame_idx=candidate.frame_idx,
                    pred_parent_id=candidate.mother_id,
                    threshold=args.parent_map_threshold,
                )
                extra = get_extra_features(candidate, seq_context, image_dir, embedding_extractor, args)
                rows.append(
                    {
                        "video_id": video_id,
                        "pred_bud_id": pred_bud_id,
                        "pred_mother_id": candidate.mother_id,
                        "gt_parent_id": gt_event["parent"],
                        "mapped_parent_gt": mapped_parent,
                        "label": 1 if mapped_parent == gt_event["parent"] else 0,
                        "features": features_from_candidate(candidate, proposal_parent, extra),
                    }
                )
    if not rows:
        raise RuntimeError("No training rows built for learned reranker.")
    x = np.stack([row["features"] for row in rows], axis=0)
    y = np.asarray([row["label"] for row in rows], dtype=np.float64)
    return rows, x, y


def write_model(path: Path, mean: np.ndarray, std: np.ndarray, theta: np.ndarray, objective: str) -> None:
    payload = {
        "objective": objective,
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "theta": theta.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_model_to_root(
    pred_root: Path,
    image_root: Path | None,
    output_root: Path,
    mean: np.ndarray,
    std: np.ndarray,
    theta: np.ndarray,
    embedding_extractor: SAM2EmbeddingExtractor | None,
    args: argparse.Namespace,
):
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(pred_root, output_root)

    for video_id in sorted(path.name for path in output_root.iterdir() if path.is_dir()):
        seq_dir = output_root / video_id
        image_dir = resolve_image_dir(image_root, video_id)
        seq_context = build_seq_context(seq_dir, args)
        track_infos = seq_context["track_infos"]
        candidates_by_bud = seq_context["candidates_by_bud"]
        res_track = _load_res_track(seq_dir / "res_track.txt")
        all_candidates: List[Candidate] = []
        extra_by_pair: Dict[Tuple[int, int], dict] = {}

        for bud_id, cand_list in candidates_by_bud.items():
            proposal_parent = track_infos[bud_id].parent
            feature_rows = []
            for cand in cand_list:
                extra = get_extra_features(cand, seq_context, image_dir, embedding_extractor, args)
                extra_by_pair[(cand.bud_id, cand.mother_id)] = extra
                feature_rows.append(features_from_candidate(cand, proposal_parent, extra))
            probs = predict_prob(np.stack(feature_rows, axis=0), mean, std, theta)
            for cand, prob in zip(cand_list, probs.tolist()):
                cand.score = float(prob)
                all_candidates.append(cand)

        assigned = _solve_global_ilp(
            bud_ids=sorted(candidates_by_bud.keys()),
            candidates=all_candidates,
            track_infos=track_infos,
            refractory_frames=args.refractory,
        )

        for bud_id, cand in assigned.items():
            if cand.score < args.score_threshold:
                continue
            bud_mask = res_track[:, 0] == bud_id
            if bud_mask.any():
                res_track[bud_mask, 3] = cand.mother_id
        _write_res_track(seq_dir / "res_track.txt", res_track)

        summary_path = seq_dir / "bud_parentage_learned.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "bud_id", "mother_id", "frame", "score", "mother_age", "dist", "size_ratio", "motion",
                    "contact", "neck", "prebud", "angle", "maturity", "track_quality", "margin", "lineage",
                    "temporal_support", "contact_persistence", "neck_persistence", "attachment_persistence",
                    "dist_stability", "separation_trend", "framewise_best_fraction",
                    "sam2_cosine_mean", "sam2_cosine_min", "sam2_cosine_trend", "status",
                ]
            )
            best_by_bud = {}
            for cand in sorted(all_candidates, key=lambda item: item.score, reverse=True):
                best_by_bud.setdefault(cand.bud_id, cand)
            for bud_id in sorted(candidates_by_bud):
                cand = assigned.get(bud_id, best_by_bud.get(bud_id))
                status = "assigned" if bud_id in assigned and cand.score >= args.score_threshold else "kept_proposal"
                extra = extra_by_pair.get((cand.bud_id, cand.mother_id), {})
                writer.writerow(
                    [
                        cand.bud_id, cand.mother_id, cand.frame_idx, f"{cand.score:.6f}", cand.mother_age,
                        f"{cand.dist:.6f}", f"{cand.size_ratio:.6f}", f"{cand.motion:.6f}",
                        f"{cand.contact:.6f}", f"{cand.neck:.6f}", f"{cand.prebud:.6f}", f"{cand.angle:.6f}",
                        f"{cand.maturity:.6f}", f"{cand.track_quality:.6f}", f"{cand.margin:.6f}", f"{cand.lineage:.6f}",
                        f"{extra.get('temporal_support', 0.0):.6f}", f"{extra.get('contact_persistence', 0.0):.6f}",
                        f"{extra.get('neck_persistence', 0.0):.6f}", f"{extra.get('attachment_persistence', 0.0):.6f}",
                        f"{extra.get('dist_stability', 0.0):.6f}", f"{extra.get('separation_trend', 0.0):.6f}",
                        f"{extra.get('framewise_best_fraction', 0.0):.6f}",
                        f"{extra.get('sam2_cosine_mean', 0.0):.6f}", f"{extra.get('sam2_cosine_min', 0.0):.6f}",
                        f"{extra.get('sam2_cosine_trend', 0.0):.6f}", status,
                    ]
                )


def main() -> None:
    args = parse_args()
    train_gt_root = Path(args.train_gt_root).expanduser().resolve()
    train_pred_root = Path(args.train_pred_root).expanduser().resolve()
    apply_pred_root = Path(args.apply_pred_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    train_image_root = Path(args.train_image_root).expanduser().resolve() if args.train_image_root else None
    apply_image_root = Path(args.apply_image_root).expanduser().resolve() if args.apply_image_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    embedding_extractor = None
    if args.sam2_model_name:
        embedding_extractor = SAM2EmbeddingExtractor(
            model_name=args.sam2_model_name,
            checkpoint_num=args.sam2_checkpoint_num,
            device=args.sam2_device,
        )

    rows, x, y = build_training_set(train_gt_root, train_pred_root, train_image_root, embedding_extractor, args)
    if args.objective == "pairwise":
        mean, std, theta, num_pairs = fit_pairwise_model(rows, x, reg_strength=args.reg_strength)
    else:
        mean, std, theta = fit_classification_model(x, y, reg_strength=args.reg_strength)
        num_pairs = 0
    write_model(output_root / "learned_parentage_model.json", mean, std, theta, args.objective)
    apply_model_to_root(apply_pred_root, apply_image_root, output_root / "predictions", mean, std, theta, embedding_extractor, args)

    train_summary = {
        "objective": args.objective,
        "num_rows": len(rows),
        "num_positive": int(y.sum()),
        "num_negative": int((1 - y).sum()),
        "num_pairwise_examples": num_pairs,
        "feature_names": FEATURE_NAMES,
        "uses_sam2_embeddings": bool(args.sam2_model_name),
    }
    (output_root / "train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
    print(json.dumps(train_summary, indent=2))


if __name__ == "__main__":
    main()

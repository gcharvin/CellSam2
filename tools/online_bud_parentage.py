#!/usr/bin/env python3
"""
Online bud->mother parentage assignment (post-processing).

Runs sequentially over frames, using only past/current information
to assign a parent (mother) to each newly appeared bud.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp

    SCIPY_MILP_AVAILABLE = True
except Exception:
    Bounds = None
    LinearConstraint = None
    milp = None
    SCIPY_MILP_AVAILABLE = False


@dataclass
class ObjStats:
    area: int
    cx: float
    cy: float


@dataclass
class Candidate:
    bud_id: int
    mother_id: int
    frame_idx: int
    score: float
    dist: float
    size_ratio: float
    motion: float
    contact: float
    neck: float
    angle: float
    track_quality: float
    margin: float


@dataclass
class TrackInfo:
    track_id: int
    start: int
    end: int
    parent: int


def _list_sequence_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    if root.is_dir():
        mask_files = list(root.glob("mask*.tif"))
        if mask_files:
            return [root]
        subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
        seqs = [p for p in subdirs if list(p.glob("mask*.tif"))]
        if seqs:
            return seqs
    raise ValueError(f"No sequences with mask*.tif found under {root}")


def _frame_index_from_name(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def _sorted_mask_files(seq_dir: Path) -> List[Path]:
    files = list(seq_dir.glob("mask*.tif"))
    return sorted(files, key=lambda p: _frame_index_from_name(p.name))


def _stats_from_mask(mask: np.ndarray) -> Dict[int, ObjStats]:
    ids = np.unique(mask)
    ids = ids[ids > 0]
    stats: Dict[int, ObjStats] = {}
    for obj_id in ids.tolist():
        ys, xs = np.where(mask == obj_id)
        if ys.size == 0:
            continue
        area = int(ys.size)
        cx = float(xs.mean())
        cy = float(ys.mean())
        stats[int(obj_id)] = ObjStats(area=area, cx=cx, cy=cy)
    return stats


def _binary_masks(mask: np.ndarray, object_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    return {
        int(obj_id): (mask == int(obj_id))
        for obj_id in object_ids
    }


def _distance(a: ObjStats, b: ObjStats) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def _mother_radius(area: int) -> float:
    return math.sqrt(max(area, 1) / math.pi)


def _motion_score(prev: ObjStats | None, curr: ObjStats, radius: float, motion_scale: float) -> float:
    if prev is None:
        return 1.0
    speed = math.hypot(curr.cx - prev.cx, curr.cy - prev.cy)
    denom = max(motion_scale * radius, 1e-6)
    return math.exp(-speed / denom)


def _contact_score(
    bud_mask: np.ndarray,
    mother_mask: np.ndarray,
    interface_radius: int,
) -> float:
    if interface_radius <= 0:
        return 0.0
    bud_area = int(bud_mask.sum())
    if bud_area <= 0:
        return 0.0
    kernel_size = 2 * interface_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mother_uint8 = mother_mask.astype(np.uint8)
    dilated = cv2.dilate(mother_uint8, kernel, iterations=1).astype(bool)
    interface_ring = dilated & (~mother_mask)
    contact_pixels = int((interface_ring & bud_mask).sum())
    return min(1.0, contact_pixels / float(bud_area))


def _edge_proximity_score(
    bud_mask: np.ndarray,
    mother_mask: np.ndarray,
    interface_radius: int,
) -> float:
    if not bud_mask.any() or not mother_mask.any():
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    bud_uint8 = bud_mask.astype(np.uint8)
    bud_border = bud_uint8.astype(bool) & (~cv2.erode(bud_uint8, kernel, iterations=1).astype(bool))
    if not bud_border.any():
        bud_border = bud_mask
    dist_to_mother = cv2.distanceTransform((~mother_mask).astype(np.uint8), cv2.DIST_L2, 3)
    border_dist = dist_to_mother[bud_border]
    if border_dist.size == 0:
        return 0.0
    min_dist = float(border_dist.min())
    scale = max(float(interface_radius), 1.0)
    return math.exp(-min_dist / scale)


def _neck_score(
    bud_mask: np.ndarray,
    mother_mask: np.ndarray,
    interface_radius: int,
) -> float:
    contact = _contact_score(bud_mask, mother_mask, interface_radius)
    proximity = _edge_proximity_score(bud_mask, mother_mask, interface_radius)
    return 0.6 * contact + 0.4 * proximity


def _angle_consistency(direction_vectors: List[Tuple[float, float]]) -> float:
    if not direction_vectors:
        return 0.0
    if len(direction_vectors) == 1:
        return 1.0
    arr = np.asarray(direction_vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    arr = arr / norms
    mean_vec = arr.mean(axis=0)
    return float(np.clip(np.linalg.norm(mean_vec), 0.0, 1.0))


def _track_quality_score(track_length: int, bud_areas: List[int], first_frames: int) -> float:
    if track_length <= 0:
        return 0.0
    length_score = min(1.0, track_length / max(float(first_frames), 1.0))
    if not bud_areas:
        return 0.5 * length_score
    mean_area = max(float(np.mean(bud_areas)), 1.0)
    cv_area = float(np.std(bud_areas) / mean_area)
    stability = math.exp(-cv_area)
    return float(np.clip(length_score * stability, 0.0, 1.0))


def _load_res_track(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 4), dtype=int)
    data = np.loadtxt(path, dtype=int)
    if data.ndim == 1 and data.size == 4:
        data = data.reshape(1, 4)
    return data


def _write_res_track(path: Path, data: np.ndarray) -> None:
    if data.size == 0:
        return
    np.savetxt(path, data, fmt="%d")


def _score_pair(
    dist: float,
    dist_max: float,
    size_ratio: float,
    motion_score: float,
    contact_score: float,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
) -> float:
    dist_score = max(0.0, 1.0 - (dist / max(dist_max, 1e-6)))
    size_score = max(0.0, 1.0 - size_ratio)
    total_weight = max(1e-6, w_dist + w_size + w_motion + w_contact)
    return (
        w_dist * dist_score
        + w_size * size_score
        + w_motion * motion_score
        + w_contact * contact_score
    ) / total_weight


def _score_aggregate_pair(
    dist: float,
    dist_max: float,
    size_ratio: float,
    motion_score: float,
    contact_score: float,
    neck_score: float,
    angle_score: float,
    track_quality: float,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_angle: float,
    w_track_quality: float,
) -> float:
    dist_score = max(0.0, 1.0 - (dist / max(dist_max, 1e-6)))
    size_score = max(0.0, 1.0 - size_ratio)
    total_weight = max(
        1e-6,
        w_dist + w_size + w_motion + w_contact + w_neck + w_angle + w_track_quality,
    )
    return (
        w_dist * dist_score
        + w_size * size_score
        + w_motion * motion_score
        + w_contact * contact_score
        + w_neck * neck_score
        + w_angle * angle_score
        + w_track_quality * track_quality
    ) / total_weight


def _apply_margin_scores(candidates: List[Candidate], w_margin: float) -> List[Candidate]:
    if not candidates or w_margin <= 0.0:
        return candidates
    by_bud: Dict[int, List[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_bud[cand.bud_id].append(cand)

    for bud_candidates in by_bud.values():
        scores = sorted((cand.score for cand in bud_candidates), reverse=True)
        if not scores:
            continue
        best_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else 0.0
        for cand in bud_candidates:
            best_other = second_score if cand.score == best_score else best_score
            cand.margin = cand.score - best_other
            cand.score = float(np.clip(cand.score + w_margin * cand.margin, 0.0, 1.0))
    return candidates


def _assign_online(
    seq_dir: Path,
    refractory_frames: int,
    max_dist_factor: float,
    bud_max_area_ratio: float,
    min_bud_area: int,
    min_mother_age: int,
    min_score: float,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    motion_scale: float,
    interface_radius: int,
    out_suffix: str,
    inplace: bool,
) -> None:
    mask_files = _sorted_mask_files(seq_dir)
    if not mask_files:
        raise ValueError(f"No mask*.tif found in {seq_dir}")

    seen_ids: set[int] = set()
    first_seen: Dict[int, int] = {}
    last_bud_frame: Dict[int, int] = {}
    prev_centroids: Dict[int, ObjStats] = {}
    parent_by_bud: Dict[int, Tuple[int, int, float, float, float]] = {}
    assignments: List[Candidate] = []

    for frame_idx, mask_path in enumerate(mask_files):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)
        if mask is None:
            raise RuntimeError(f"Failed to read {mask_path}")
        stats = _stats_from_mask(mask)
        ids_present = set(stats.keys())
        masks = _binary_masks(mask, ids_present)

        # track first appearance
        new_ids = ids_present - seen_ids
        for obj_id in new_ids:
            first_seen[obj_id] = frame_idx
        seen_ids |= ids_present

        # Bud candidates: newly appeared objects above size threshold
        bud_candidates = [
            obj_id
            for obj_id in new_ids
            if stats[obj_id].area >= min_bud_area
        ]

        # Candidate mothers: present, not newly appeared, old enough
        mother_candidates = [
            obj_id
            for obj_id in ids_present
            if obj_id not in new_ids
            and (frame_idx - first_seen.get(obj_id, frame_idx)) >= min_mother_age
        ]

        # Build candidate pairs
        pairs: List[Candidate] = []
        for bud_id in bud_candidates:
            bud = stats[bud_id]
            for mother_id in mother_candidates:
                mother = stats[mother_id]

                # Refractory filter
                last_bud = last_bud_frame.get(mother_id)
                if last_bud is not None and (frame_idx - last_bud) < refractory_frames:
                    continue

                # Size filter
                if mother.area <= 0:
                    continue
                size_ratio = bud.area / float(mother.area)
                if size_ratio > bud_max_area_ratio:
                    continue

                # Distance filter
                radius = _mother_radius(mother.area)
                dist = _distance(bud, mother)
                dist_max = max_dist_factor * radius
                if dist_max <= 0 or dist > dist_max:
                    continue

                motion_score = _motion_score(prev_centroids.get(mother_id), mother, radius, motion_scale)
                contact_score = _contact_score(
                    bud_mask=masks[bud_id],
                    mother_mask=masks[mother_id],
                    interface_radius=interface_radius,
                )
                score = _score_pair(
                    dist=dist,
                    dist_max=dist_max,
                    size_ratio=size_ratio,
                    motion_score=motion_score,
                    contact_score=contact_score,
                    w_dist=w_dist,
                    w_size=w_size,
                    w_motion=w_motion,
                    w_contact=w_contact,
                )
                if score < min_score:
                    continue

                pairs.append(
                    Candidate(
                        bud_id=bud_id,
                        mother_id=mother_id,
                        frame_idx=frame_idx,
                        score=score,
                        dist=dist,
                        size_ratio=size_ratio,
                        motion=motion_score,
                        contact=contact_score,
                        neck=contact_score,
                        angle=1.0,
                        track_quality=1.0,
                        margin=0.0,
                    )
                )

        # Greedy assignment (online)
        pairs.sort(key=lambda c: c.score, reverse=True)
        assigned_buds: set[int] = set()
        assigned_mothers: set[int] = set()
        for cand in pairs:
            if cand.bud_id in assigned_buds or cand.mother_id in assigned_mothers:
                continue
            assigned_buds.add(cand.bud_id)
            assigned_mothers.add(cand.mother_id)
            parent_by_bud[cand.bud_id] = (
                cand.mother_id,
                cand.frame_idx,
                cand.score,
                cand.dist,
                cand.size_ratio,
            )
            last_bud_frame[cand.mother_id] = cand.frame_idx
            assignments.append(cand)

        # Update prev centroids
        for obj_id, st in stats.items():
            prev_centroids[obj_id] = st

    # Update res_track parent ids
    res_track_path = seq_dir / "res_track.txt"
    res_track = _load_res_track(res_track_path)
    if res_track.size == 0:
        return

    for bud_id, (mother_id, _, _, _, _) in parent_by_bud.items():
        mask = res_track[:, 0] == bud_id
        if mask.any():
            res_track[mask, 3] = mother_id

    out_path = res_track_path if inplace else seq_dir / f"res_track{out_suffix}.txt"
    _write_res_track(out_path, res_track)

    # Write assignment summary
    summary_path = seq_dir / f"bud_parentage{out_suffix}.csv"
    with summary_path.open("w") as f:
        f.write("bud_id,mother_id,frame,score,dist,size_ratio,motion,contact,neck,angle,track_quality,margin\n")
        for cand in assignments:
            f.write(
                f"{cand.bud_id},{cand.mother_id},{cand.frame_idx},"
                f"{cand.score:.4f},{cand.dist:.2f},{cand.size_ratio:.4f},"
                f"{cand.motion:.4f},{cand.contact:.4f},{cand.neck:.4f},"
                f"{cand.angle:.4f},{cand.track_quality:.4f},{cand.margin:.4f}\n"
            )


def _load_track_infos(path: Path) -> Dict[int, TrackInfo]:
    rows = _load_res_track(path)
    infos: Dict[int, TrackInfo] = {}
    for obj_id, start, end, parent in rows.tolist():
        infos[int(obj_id)] = TrackInfo(
            track_id=int(obj_id),
            start=int(start),
            end=int(end),
            parent=int(parent),
        )
    return infos


def _collect_frame_cache(mask_files: List[Path]):
    masks: Dict[int, np.ndarray] = {}
    stats_cache: Dict[int, Dict[int, ObjStats]] = {}
    bin_masks: Dict[int, Dict[int, np.ndarray]] = {}

    for frame_idx, mask_path in enumerate(mask_files):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)
        if mask is None:
            raise RuntimeError(f"Failed to read {mask_path}")
        masks[frame_idx] = mask
        stats = _stats_from_mask(mask)
        stats_cache[frame_idx] = stats
        bin_masks[frame_idx] = _binary_masks(mask, stats.keys())
    return masks, stats_cache, bin_masks


def _aggregate_pair_candidate(
    bud_id: int,
    mother_id: int,
    track_infos: Dict[int, TrackInfo],
    stats_cache: Dict[int, Dict[int, ObjStats]],
    bin_masks: Dict[int, Dict[int, np.ndarray]],
    max_dist_factor: float,
    bud_max_area_ratio: float,
    min_score: float,
    first_frames: int,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_angle: float,
    w_track_quality: float,
    motion_scale: float,
    interface_radius: int,
) -> Candidate | None:
    bud_track = track_infos[bud_id]
    end_frame = min(bud_track.end, bud_track.start + first_frames - 1)
    frame_range = range(bud_track.start, end_frame + 1)
    score_parts: List[Candidate] = []
    direction_vectors: List[Tuple[float, float]] = []
    bud_areas: List[int] = []
    dist_max_values: List[float] = []
    for frame_idx in frame_range:
        frame_stats = stats_cache.get(frame_idx, {})
        bud_stats = frame_stats.get(bud_id)
        mother_stats = frame_stats.get(mother_id)
        if bud_stats is None or mother_stats is None:
            continue
        if mother_stats.area <= 0:
            continue
        size_ratio = bud_stats.area / float(mother_stats.area)
        if size_ratio > bud_max_area_ratio:
            continue
        radius = _mother_radius(mother_stats.area)
        dist = _distance(bud_stats, mother_stats)
        dist_max = max_dist_factor * radius
        if dist_max <= 0 or dist > dist_max:
            continue

        direction_vectors.append((bud_stats.cx - mother_stats.cx, bud_stats.cy - mother_stats.cy))
        bud_areas.append(bud_stats.area)
        dist_max_values.append(dist_max)
        prev_stats = stats_cache.get(frame_idx - 1, {}).get(mother_id)
        motion_score = _motion_score(prev_stats, mother_stats, radius, motion_scale)
        contact_score = _contact_score(
            bud_mask=bin_masks[frame_idx][bud_id],
            mother_mask=bin_masks[frame_idx][mother_id],
            interface_radius=interface_radius,
        )
        score_parts.append(
            Candidate(
                bud_id=bud_id,
                mother_id=mother_id,
                frame_idx=frame_idx,
                score=0.0,
                dist=dist,
                size_ratio=size_ratio,
                motion=motion_score,
                contact=contact_score,
                neck=_neck_score(
                    bud_mask=bin_masks[frame_idx][bud_id],
                    mother_mask=bin_masks[frame_idx][mother_id],
                    interface_radius=interface_radius,
                ),
                angle=0.0,
                track_quality=0.0,
                margin=0.0,
            )
        )

    if not score_parts:
        return None

    support = len(score_parts)
    angle_score = _angle_consistency(direction_vectors)
    track_quality = _track_quality_score(
        track_length=bud_track.end - bud_track.start + 1,
        bud_areas=bud_areas,
        first_frames=first_frames,
    )
    avg_score = _score_aggregate_pair(
        dist=sum(part.dist for part in score_parts) / support,
        dist_max=sum(dist_max_values) / max(len(dist_max_values), 1),
        size_ratio=sum(part.size_ratio for part in score_parts) / support,
        motion_score=sum(part.motion for part in score_parts) / support,
        contact_score=sum(part.contact for part in score_parts) / support,
        neck_score=sum(part.neck for part in score_parts) / support,
        angle_score=angle_score,
        track_quality=track_quality,
        w_dist=w_dist,
        w_size=w_size,
        w_motion=w_motion,
        w_contact=w_contact,
        w_neck=w_neck,
        w_angle=w_angle,
        w_track_quality=w_track_quality,
    )
    if avg_score < min_score:
        return None
    return Candidate(
        bud_id=bud_id,
        mother_id=mother_id,
        frame_idx=bud_track.start,
        score=avg_score,
        dist=sum(part.dist for part in score_parts) / support,
        size_ratio=sum(part.size_ratio for part in score_parts) / support,
        motion=sum(part.motion for part in score_parts) / support,
        contact=sum(part.contact for part in score_parts) / support,
        neck=sum(part.neck for part in score_parts) / support,
        angle=angle_score,
        track_quality=track_quality,
        margin=0.0,
    )


def _build_global_candidates(
    track_infos: Dict[int, TrackInfo],
    stats_cache: Dict[int, Dict[int, ObjStats]],
    bin_masks: Dict[int, Dict[int, np.ndarray]],
    refractory_frames: int,
    max_dist_factor: float,
    bud_max_area_ratio: float,
    min_bud_area: int,
    min_mother_age: int,
    min_score: float,
    min_track_length: int,
    first_frames: int,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_angle: float,
    w_track_quality: float,
    w_margin: float,
    motion_scale: float,
    interface_radius: int,
) -> Tuple[List[int], List[Candidate]]:
    candidate_buds: List[int] = []
    candidates: List[Candidate] = []

    for bud_id, bud_track in sorted(track_infos.items(), key=lambda item: (item[1].start, item[0])):
        if bud_track.start <= 0:
            continue
        track_length = bud_track.end - bud_track.start + 1
        if track_length < min_track_length:
            continue
        start_stats = stats_cache.get(bud_track.start, {}).get(bud_id)
        if start_stats is None or start_stats.area < min_bud_area:
            continue
        candidate_buds.append(bud_id)

        end_frame = min(bud_track.end, bud_track.start + first_frames - 1)
        frame_range = range(bud_track.start, end_frame + 1)
        for mother_id, mother_track in track_infos.items():
            if mother_id == bud_id:
                continue
            if mother_track.start > bud_track.start - min_mother_age:
                continue
            if mother_track.end < bud_track.start:
                continue

            candidate = _aggregate_pair_candidate(
                bud_id=bud_id,
                mother_id=mother_id,
                track_infos=track_infos,
                stats_cache=stats_cache,
                bin_masks=bin_masks,
                max_dist_factor=max_dist_factor,
                bud_max_area_ratio=bud_max_area_ratio,
                min_score=min_score,
                first_frames=first_frames,
                w_dist=w_dist,
                w_size=w_size,
                w_motion=w_motion,
                w_contact=w_contact,
                w_neck=w_neck,
                w_angle=w_angle,
                w_track_quality=w_track_quality,
                motion_scale=motion_scale,
                interface_radius=interface_radius,
            )
            if candidate is None:
                continue
            candidates.append(candidate)

    return candidate_buds, _apply_margin_scores(candidates, w_margin)


def _load_parentage_scores(seq_dir: Path, out_suffix: str) -> Dict[int, float]:
    summary_path = seq_dir / f"bud_parentage{out_suffix}.csv"
    if not summary_path.exists():
        return {}
    lines = [line.strip() for line in summary_path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split(",")
    try:
        bud_idx = header.index("bud_id")
        score_idx = header.index("score")
    except ValueError:
        return {}
    scores: Dict[int, float] = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(bud_idx, score_idx):
            continue
        try:
            scores[int(parts[bud_idx])] = float(parts[score_idx])
        except ValueError:
            continue
    return scores


def _solve_global_greedy(
    bud_ids: List[int],
    candidates: List[Candidate],
    track_infos: Dict[int, TrackInfo],
    refractory_frames: int,
) -> Dict[int, Candidate]:
    assigned: Dict[int, Candidate] = {}
    mother_frames: Dict[int, List[int]] = defaultdict(list)
    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        if cand.bud_id in assigned:
            continue
        conflict = any(
            abs(track_infos[cand.bud_id].start - frame) < refractory_frames
            for frame in mother_frames[cand.mother_id]
        )
        if conflict:
            continue
        assigned[cand.bud_id] = cand
        mother_frames[cand.mother_id].append(track_infos[cand.bud_id].start)
    return assigned


def _solve_global_ilp(
    bud_ids: List[int],
    candidates: List[Candidate],
    track_infos: Dict[int, TrackInfo],
    refractory_frames: int,
) -> Dict[int, Candidate]:
    if not SCIPY_MILP_AVAILABLE or not candidates:
        return _solve_global_greedy(bud_ids, candidates, track_infos, refractory_frames)

    cand_indices_by_bud: Dict[int, List[int]] = defaultdict(list)
    for idx, cand in enumerate(candidates):
        cand_indices_by_bud[cand.bud_id].append(idx)

    num_cand = len(candidates)
    orphan_offset = num_cand
    num_vars = num_cand + len(bud_ids)
    objective = np.zeros(num_vars, dtype=float)
    integrality = np.ones(num_vars, dtype=int)
    lower = np.zeros(num_vars, dtype=float)
    upper = np.ones(num_vars, dtype=float)

    for idx, cand in enumerate(candidates):
        objective[idx] = -cand.score

    rows = []
    lb = []
    ub = []

    bud_to_orphan_idx = {
        bud_id: orphan_offset + bud_position for bud_position, bud_id in enumerate(bud_ids)
    }
    for bud_id in bud_ids:
        row = np.zeros(num_vars, dtype=float)
        for idx in cand_indices_by_bud.get(bud_id, []):
            row[idx] = 1.0
        row[bud_to_orphan_idx[bud_id]] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    candidates_by_mother: Dict[int, List[int]] = defaultdict(list)
    for idx, cand in enumerate(candidates):
        candidates_by_mother[cand.mother_id].append(idx)
    for mother_id, mother_cands in candidates_by_mother.items():
        for left_pos, left_idx in enumerate(mother_cands):
            left_bud = candidates[left_idx].bud_id
            left_start = track_infos[left_bud].start
            for right_idx in mother_cands[left_pos + 1 :]:
                right_bud = candidates[right_idx].bud_id
                right_start = track_infos[right_bud].start
                if abs(left_start - right_start) >= refractory_frames:
                    continue
                row = np.zeros(num_vars, dtype=float)
                row[left_idx] = 1.0
                row[right_idx] = 1.0
                rows.append(row)
                lb.append(-np.inf)
                ub.append(1.0)

    constraints = LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
    )
    if result.x is None or not result.success:
        return _solve_global_greedy(bud_ids, candidates, track_infos, refractory_frames)

    assigned: Dict[int, Candidate] = {}
    chosen = result.x[:num_cand] > 0.5
    for idx, use_edge in enumerate(chosen.tolist()):
        if use_edge:
            assigned[candidates[idx].bud_id] = candidates[idx]
    return assigned


def _assign_global(
    seq_dir: Path,
    refractory_frames: int,
    max_dist_factor: float,
    bud_max_area_ratio: float,
    min_bud_area: int,
    min_mother_age: int,
    min_score: float,
    min_track_length: int,
    first_frames: int,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_angle: float,
    w_track_quality: float,
    w_margin: float,
    motion_scale: float,
    interface_radius: int,
    out_suffix: str,
    inplace: bool,
) -> None:
    mask_files = _sorted_mask_files(seq_dir)
    if not mask_files:
        raise ValueError(f"No mask*.tif found in {seq_dir}")

    res_track_path = seq_dir / "res_track.txt"
    res_track = _load_res_track(res_track_path)
    if res_track.size == 0:
        return

    track_infos = _load_track_infos(res_track_path)
    _, stats_cache, bin_masks = _collect_frame_cache(mask_files)
    bud_ids, candidates = _build_global_candidates(
        track_infos=track_infos,
        stats_cache=stats_cache,
        bin_masks=bin_masks,
        refractory_frames=refractory_frames,
        max_dist_factor=max_dist_factor,
        bud_max_area_ratio=bud_max_area_ratio,
        min_bud_area=min_bud_area,
        min_mother_age=min_mother_age,
        min_score=min_score,
        min_track_length=min_track_length,
        first_frames=first_frames,
        w_dist=w_dist,
        w_size=w_size,
        w_motion=w_motion,
        w_contact=w_contact,
        w_neck=w_neck,
        w_angle=w_angle,
        w_track_quality=w_track_quality,
        w_margin=w_margin,
        motion_scale=motion_scale,
        interface_radius=interface_radius,
    )
    assigned = _solve_global_ilp(
        bud_ids=bud_ids,
        candidates=candidates,
        track_infos=track_infos,
        refractory_frames=refractory_frames,
    )

    for bud_id in bud_ids:
        bud_mask = res_track[:, 0] == bud_id
        if bud_mask.any():
            res_track[bud_mask, 3] = 0
    for bud_id, cand in assigned.items():
        bud_mask = res_track[:, 0] == bud_id
        if bud_mask.any():
            res_track[bud_mask, 3] = cand.mother_id

    out_path = res_track_path if inplace else seq_dir / f"res_track{out_suffix}.txt"
    _write_res_track(out_path, res_track)

    best_by_bud = {cand.bud_id: cand for cand in assigned.values()}
    by_bud_candidates: Dict[int, List[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_bud_candidates[cand.bud_id].append(cand)
    summary_path = seq_dir / f"bud_parentage{out_suffix}.csv"
    with summary_path.open("w") as f:
        f.write("bud_id,mother_id,frame,score,dist,size_ratio,motion,contact,neck,angle,track_quality,margin,status,num_candidates\n")
        for bud_id in sorted(bud_ids):
            if bud_id in best_by_bud:
                cand = best_by_bud[bud_id]
                status = "assigned"
            else:
                cand = max(
                    by_bud_candidates.get(bud_id, []),
                    key=lambda item: item.score,
                    default=Candidate(
                        bud_id=bud_id,
                        mother_id=0,
                        frame_idx=track_infos[bud_id].start,
                        score=0.0,
                        dist=0.0,
                        size_ratio=0.0,
                        motion=0.0,
                        contact=0.0,
                        neck=0.0,
                        angle=0.0,
                        track_quality=0.0,
                        margin=0.0,
                    ),
                )
                status = "orphan"
            f.write(
                f"{bud_id},{cand.mother_id},{cand.frame_idx},{cand.score:.4f},"
                f"{cand.dist:.2f},{cand.size_ratio:.4f},{cand.motion:.4f},"
                f"{cand.contact:.4f},{cand.neck:.4f},{cand.angle:.4f},"
                f"{cand.track_quality:.4f},{cand.margin:.4f},{status},"
                f"{len(by_bud_candidates.get(bud_id, []))}\n"
            )


def _assignment_conflicts(
    assigned_parents: Dict[int, int],
    track_infos: Dict[int, TrackInfo],
    refractory_frames: int,
) -> set[int]:
    by_mother: Dict[int, List[int]] = defaultdict(list)
    for bud_id, mother_id in assigned_parents.items():
        if mother_id > 0:
            by_mother[mother_id].append(bud_id)
    conflicting: set[int] = set()
    for bud_ids in by_mother.values():
        ordered = sorted(bud_ids, key=lambda bud_id: track_infos[bud_id].start)
        for left_idx, left_bud in enumerate(ordered):
            left_start = track_infos[left_bud].start
            for right_bud in ordered[left_idx + 1 :]:
                right_start = track_infos[right_bud].start
                if abs(right_start - left_start) >= refractory_frames:
                    break
                conflicting.add(left_bud)
                conflicting.add(right_bud)
    return conflicting


def _assign_hybrid(
    seq_dir: Path,
    refractory_frames: int,
    max_dist_factor: float,
    bud_max_area_ratio: float,
    min_bud_area: int,
    min_mother_age: int,
    min_score: float,
    min_track_length: int,
    first_frames: int,
    proposal_bonus: float,
    proposal_lock_score: float,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_angle: float,
    w_track_quality: float,
    w_margin: float,
    motion_scale: float,
    interface_radius: int,
    out_suffix: str,
    inplace: bool,
) -> None:
    mask_files = _sorted_mask_files(seq_dir)
    if not mask_files:
        raise ValueError(f"No mask*.tif found in {seq_dir}")

    res_track_path = seq_dir / "res_track.txt"
    res_track = _load_res_track(res_track_path)
    if res_track.size == 0:
        return

    track_infos = _load_track_infos(res_track_path)
    _, stats_cache, bin_masks = _collect_frame_cache(mask_files)
    bud_ids, candidates = _build_global_candidates(
        track_infos=track_infos,
        stats_cache=stats_cache,
        bin_masks=bin_masks,
        refractory_frames=refractory_frames,
        max_dist_factor=max_dist_factor,
        bud_max_area_ratio=bud_max_area_ratio,
        min_bud_area=min_bud_area,
        min_mother_age=min_mother_age,
        min_score=min_score,
        min_track_length=min_track_length,
        first_frames=first_frames,
        w_dist=w_dist,
        w_size=w_size,
        w_motion=w_motion,
        w_contact=w_contact,
        w_neck=w_neck,
        w_angle=w_angle,
        w_track_quality=w_track_quality,
        w_margin=w_margin,
        motion_scale=motion_scale,
        interface_radius=interface_radius,
    )

    proposal_scores = _load_parentage_scores(seq_dir, out_suffix)
    proposal_parent_by_bud = {
        bud_id: track_infos[bud_id].parent
        for bud_id in bud_ids
    }
    conflicting_buds = _assignment_conflicts(
        assigned_parents=proposal_parent_by_bud,
        track_infos=track_infos,
        refractory_frames=refractory_frames,
    )

    locked_assignments: Dict[int, Candidate] = {}
    flex_buds: set[int] = set()
    candidates_by_bud: Dict[int, List[Candidate]] = defaultdict(list)
    for cand in candidates:
        if cand.mother_id == proposal_parent_by_bud.get(cand.bud_id, 0):
            cand.score += proposal_bonus
        candidates_by_bud[cand.bud_id].append(cand)

    for bud_id in bud_ids:
        proposed_parent = proposal_parent_by_bud.get(bud_id, 0)
        proposed_score = proposal_scores.get(bud_id, 0.0)
        if proposed_parent <= 0:
            flex_buds.add(bud_id)
            continue
        if bud_id in conflicting_buds:
            flex_buds.add(bud_id)
            continue
        if proposed_score >= proposal_lock_score:
            locked_assignments[bud_id] = Candidate(
                bud_id=bud_id,
                mother_id=proposed_parent,
                frame_idx=track_infos[bud_id].start,
                score=proposed_score,
                dist=0.0,
                size_ratio=0.0,
                motion=0.0,
                contact=0.0,
                neck=0.0,
                angle=0.0,
                track_quality=0.0,
                margin=0.0,
            )
        else:
            flex_buds.add(bud_id)

    filtered_candidates: List[Candidate] = []
    for cand in candidates:
        if cand.bud_id not in flex_buds:
            continue
        blocked = False
        for locked_bud, locked in locked_assignments.items():
            if cand.mother_id != locked.mother_id:
                continue
            if abs(track_infos[cand.bud_id].start - track_infos[locked_bud].start) < refractory_frames:
                blocked = True
                break
        if not blocked:
            filtered_candidates.append(cand)

    optimized = _solve_global_ilp(
        bud_ids=sorted(flex_buds),
        candidates=filtered_candidates,
        track_infos=track_infos,
        refractory_frames=refractory_frames,
    )
    assigned = dict(locked_assignments)
    assigned.update(optimized)

    for bud_id in bud_ids:
        bud_mask = res_track[:, 0] == bud_id
        if bud_mask.any():
            res_track[bud_mask, 3] = 0
    for bud_id, cand in assigned.items():
        bud_mask = res_track[:, 0] == bud_id
        if bud_mask.any():
            res_track[bud_mask, 3] = cand.mother_id

    out_path = res_track_path if inplace else seq_dir / f"res_track{out_suffix}.txt"
    _write_res_track(out_path, res_track)

    by_bud_candidates: Dict[int, List[Candidate]] = defaultdict(list)
    for cand in filtered_candidates:
        by_bud_candidates[cand.bud_id].append(cand)
    summary_path = seq_dir / f"bud_parentage{out_suffix}.csv"
    with summary_path.open("w") as f:
        f.write("bud_id,mother_id,frame,score,dist,size_ratio,motion,contact,neck,angle,track_quality,margin,status,num_candidates\n")
        for bud_id in sorted(bud_ids):
            if bud_id in locked_assignments:
                cand = locked_assignments[bud_id]
                status = "locked"
            elif bud_id in optimized:
                cand = optimized[bud_id]
                status = "optimized"
            else:
                cand = max(
                    by_bud_candidates.get(bud_id, []),
                    key=lambda item: item.score,
                    default=Candidate(
                        bud_id=bud_id,
                        mother_id=0,
                        frame_idx=track_infos[bud_id].start,
                        score=0.0,
                        dist=0.0,
                        size_ratio=0.0,
                        motion=0.0,
                        contact=0.0,
                        neck=0.0,
                        angle=0.0,
                        track_quality=0.0,
                        margin=0.0,
                    ),
                )
                status = "orphan"
            f.write(
                f"{bud_id},{cand.mother_id},{cand.frame_idx},{cand.score:.4f},"
                f"{cand.dist:.2f},{cand.size_ratio:.4f},{cand.motion:.4f},"
                f"{cand.contact:.4f},{cand.neck:.4f},{cand.angle:.4f},"
                f"{cand.track_quality:.4f},{cand.margin:.4f},{status},"
                f"{len(by_bud_candidates.get(bud_id, []))}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Online bud parentage post-processing")
    parser.add_argument("--pred_root", required=True, help="Prediction root (sequence dir or parent)")
    parser.add_argument("--mode", choices=["online", "global", "hybrid"], default="online", help="Assignment mode")
    parser.add_argument("--refractory", type=int, default=8, help="Refractory frames per mother")
    parser.add_argument("--max_dist_factor", type=float, default=2.5, help="Max dist = factor * mother radius")
    parser.add_argument("--bud_max_area_ratio", type=float, default=0.6, help="Max bud/mother area ratio")
    parser.add_argument("--min_bud_area", type=int, default=10, help="Minimum bud area")
    parser.add_argument("--min_mother_age", type=int, default=3, help="Min age (frames) for a mother candidate")
    parser.add_argument("--min_score", type=float, default=0.1, help="Minimum score for assignment")
    parser.add_argument("--min_track_length", type=int, default=2, help="Minimum candidate bud track length for global mode")
    parser.add_argument("--first_frames", type=int, default=4, help="Number of early frames aggregated in global mode")
    parser.add_argument("--proposal_bonus", type=float, default=0.05, help="Extra score added to the current proposal edge in hybrid mode")
    parser.add_argument("--proposal_lock_score", type=float, default=0.60, help="Proposal score above which non-conflicting assignments are locked in hybrid mode")
    parser.add_argument("--w_dist", type=float, default=0.6, help="Weight for distance score")
    parser.add_argument("--w_size", type=float, default=0.3, help="Weight for size score")
    parser.add_argument("--w_motion", type=float, default=0.1, help="Weight for motion score")
    parser.add_argument("--w_contact", type=float, default=0.0, help="Weight for mother-bud contact score")
    parser.add_argument("--w_neck", type=float, default=0.25, help="Weight for neck/interface score in global and hybrid modes")
    parser.add_argument("--w_angle", type=float, default=0.20, help="Weight for angle consistency score in global and hybrid modes")
    parser.add_argument("--w_track_quality", type=float, default=0.10, help="Weight for bud track quality score in global and hybrid modes")
    parser.add_argument("--w_margin", type=float, default=0.15, help="Weight for relative margin against competing mothers in global and hybrid modes")
    parser.add_argument("--motion_scale", type=float, default=2.0, help="Scale for motion penalty")
    parser.add_argument("--interface_radius", type=int, default=4, help="Dilation radius used to measure bud-mother contact")
    parser.add_argument("--out_suffix", type=str, default="_parented", help="Suffix for output files")
    parser.add_argument("--inplace", action="store_true", help="Overwrite res_track.txt")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    for seq_dir in _list_sequence_dirs(pred_root):
        if args.mode == "global":
            _assign_global(
                seq_dir=seq_dir,
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
                w_angle=args.w_angle,
                w_track_quality=args.w_track_quality,
                w_margin=args.w_margin,
                motion_scale=args.motion_scale,
                interface_radius=args.interface_radius,
                out_suffix=args.out_suffix,
                inplace=args.inplace,
            )
        elif args.mode == "hybrid":
            _assign_hybrid(
                seq_dir=seq_dir,
                refractory_frames=args.refractory,
                max_dist_factor=args.max_dist_factor,
                bud_max_area_ratio=args.bud_max_area_ratio,
                min_bud_area=args.min_bud_area,
                min_mother_age=args.min_mother_age,
                min_score=args.min_score,
                min_track_length=args.min_track_length,
                first_frames=args.first_frames,
                proposal_bonus=args.proposal_bonus,
                proposal_lock_score=args.proposal_lock_score,
                w_dist=args.w_dist,
                w_size=args.w_size,
                w_motion=args.w_motion,
                w_contact=args.w_contact,
                w_neck=args.w_neck,
                w_angle=args.w_angle,
                w_track_quality=args.w_track_quality,
                w_margin=args.w_margin,
                motion_scale=args.motion_scale,
                interface_radius=args.interface_radius,
                out_suffix=args.out_suffix,
                inplace=args.inplace,
            )
        else:
            _assign_online(
                seq_dir=seq_dir,
                refractory_frames=args.refractory,
                max_dist_factor=args.max_dist_factor,
                bud_max_area_ratio=args.bud_max_area_ratio,
                min_bud_area=args.min_bud_area,
                min_mother_age=args.min_mother_age,
                min_score=args.min_score,
                w_dist=args.w_dist,
                w_size=args.w_size,
                w_motion=args.w_motion,
                w_contact=args.w_contact,
                motion_scale=args.motion_scale,
                interface_radius=args.interface_radius,
                out_suffix=args.out_suffix,
                inplace=args.inplace,
            )


if __name__ == "__main__":
    main()

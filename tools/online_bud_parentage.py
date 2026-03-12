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
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    mother_age: int
    dist: float
    size_ratio: float
    motion: float
    contact: float
    neck: float
    prebud: float
    angle: float
    maturity: float
    track_quality: float
    margin: float
    lineage: float


@dataclass
class TrackInfo:
    track_id: int
    start: int
    end: int
    parent: int


def _track_age(track_info: TrackInfo, frame_idx: int) -> int:
    return max(0, frame_idx - track_info.start)


def _is_ancestor(track_infos: Dict[int, TrackInfo], ancestor_id: int, descendant_id: int) -> bool:
    current = track_infos.get(descendant_id)
    while current is not None and current.parent > 0:
        if current.parent == ancestor_id:
            return True
        current = track_infos.get(current.parent)
    return False


def _same_lineage_family(track_infos: Dict[int, TrackInfo], left_id: int, right_id: int) -> bool:
    if left_id <= 0 or right_id <= 0 or left_id == right_id:
        return left_id == right_id and left_id > 0
    if _is_ancestor(track_infos, left_id, right_id) or _is_ancestor(track_infos, right_id, left_id):
        return True
    left = track_infos.get(left_id)
    right = track_infos.get(right_id)
    return (
        left is not None
        and right is not None
        and left.parent > 0
        and left.parent == right.parent
    )


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


def _border_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_uint8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1).astype(bool)
    border = mask & (~eroded)
    return border if border.any() else mask


def _border_contact_score(
    bud_mask: np.ndarray,
    mother_mask: np.ndarray,
    interface_radius: int,
) -> float:
    if interface_radius <= 0 or not bud_mask.any() or not mother_mask.any():
        return 0.0
    bud_border = _border_mask(bud_mask)
    border_area = int(bud_border.sum())
    if border_area <= 0:
        return 0.0
    kernel_size = 2 * interface_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mother_border = _border_mask(mother_mask).astype(np.uint8)
    dilated_mother_border = cv2.dilate(mother_border, kernel, iterations=1).astype(bool)
    contact_pixels = int((bud_border & dilated_mother_border).sum())
    return min(1.0, contact_pixels / float(border_area))


def _edge_proximity_score(
    bud_mask: np.ndarray,
    mother_mask: np.ndarray,
    interface_radius: int,
) -> float:
    if not bud_mask.any() or not mother_mask.any():
        return 0.0
    bud_border = _border_mask(bud_mask)
    mother_border = _border_mask(mother_mask)
    dist_to_mother = cv2.distanceTransform((~mother_border).astype(np.uint8), cv2.DIST_L2, 3)
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
    use_border_neck: bool,
) -> float:
    contact = _contact_score(bud_mask, mother_mask, interface_radius)
    proximity = _edge_proximity_score(bud_mask, mother_mask, interface_radius)
    if not use_border_neck:
        return 0.6 * contact + 0.4 * proximity
    border_contact = _border_contact_score(bud_mask, mother_mask, interface_radius)
    return float(np.clip(0.45 * border_contact + 0.35 * contact + 0.20 * proximity, 0.0, 1.0))


def _prebud_score(
    bud_mask: np.ndarray,
    prev_mask: np.ndarray | None,
    prev_mother_mask: np.ndarray | None,
    interface_radius: int,
) -> float:
    if prev_mask is None or prev_mother_mask is None or not bud_mask.any() or not prev_mother_mask.any():
        return 0.0
    bud_area = int(bud_mask.sum())
    if bud_area <= 0:
        return 0.0
    adjacency = _border_contact_score(bud_mask, prev_mother_mask, interface_radius)
    proximity = _edge_proximity_score(bud_mask, prev_mother_mask, interface_radius)
    allowed_region = prev_mother_mask | (prev_mask == 0)
    occupancy_ok = float((bud_mask & allowed_region).sum()) / float(bud_area)
    return float(np.clip(0.45 * adjacency + 0.35 * proximity + 0.20 * occupancy_ok, 0.0, 1.0))


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
    neck_score: float,
    prebud_score: float,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_prebud: float,
) -> float:
    dist_score = max(0.0, 1.0 - (dist / max(dist_max, 1e-6)))
    size_score = max(0.0, 1.0 - size_ratio)
    total_weight = max(1e-6, w_dist + w_size + w_motion + w_contact + w_neck + w_prebud)
    return (
        w_dist * dist_score
        + w_size * size_score
        + w_motion * motion_score
        + w_contact * contact_score
        + w_neck * neck_score
        + w_prebud * prebud_score
    ) / total_weight


def _score_aggregate_pair(
    dist: float,
    dist_max: float,
    size_ratio: float,
    motion_score: float,
    contact_score: float,
    neck_score: float,
    prebud_score: float,
    angle_score: float,
    maturity_score: float,
    track_quality: float,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_prebud: float,
    w_angle: float,
    w_maturity: float,
    w_track_quality: float,
) -> float:
    dist_score = max(0.0, 1.0 - (dist / max(dist_max, 1e-6)))
    size_score = max(0.0, 1.0 - size_ratio)
    total_weight = max(
        1e-6,
        w_dist + w_size + w_motion + w_contact + w_neck + w_prebud + w_angle + w_maturity + w_track_quality,
    )
    return (
        w_dist * dist_score
        + w_size * size_score
        + w_motion * motion_score
        + w_contact * contact_score
        + w_neck * neck_score
        + w_prebud * prebud_score
        + w_angle * angle_score
        + w_maturity * maturity_score
        + w_track_quality * track_quality
    ) / total_weight


def _maturity_score(mother_age: int, min_mother_age: int, preferred_mother_age: int) -> float:
    if preferred_mother_age <= min_mother_age:
        return 1.0 if mother_age >= min_mother_age else 0.0
    return float(
        np.clip(
            (mother_age - min_mother_age) / max(preferred_mother_age - min_mother_age, 1),
            0.0,
            1.0,
        )
    )


def _apply_competition_scores(
    candidates: List[Candidate],
    track_infos: Dict[int, TrackInfo],
    preferred_mother_age: int,
    w_margin: float,
    w_lineage: float,
    lineage_margin: float,
) -> List[Candidate]:
    if not candidates or (w_margin <= 0.0 and w_lineage <= 0.0):
        return candidates
    by_bud: Dict[int, List[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_bud[cand.bud_id].append(cand)

    for bud_candidates in by_bud.values():
        base_scores = {id(cand): cand.score for cand in bud_candidates}
        scores = sorted(base_scores.values(), reverse=True)
        if not scores:
            continue
        best_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else 0.0
        for cand in bud_candidates:
            cand_score = base_scores[id(cand)]
            best_other = second_score if cand_score == best_score else best_score
            cand.margin = cand_score - best_other

            lineage_adjust = 0.0
            if w_lineage > 0.0:
                for other in bud_candidates:
                    if other is cand:
                        continue
                    other_score = base_scores[id(other)]
                    if _is_ancestor(track_infos, other.mother_id, cand.mother_id):
                        if cand.mother_age < preferred_mother_age:
                            immaturity = 1.0 - (cand.mother_age / max(preferred_mother_age, 1))
                            lineage_adjust -= immaturity
                        elif cand_score < other_score + lineage_margin:
                            lineage_adjust -= 0.35
                    elif _is_ancestor(track_infos, cand.mother_id, other.mother_id):
                        if (
                            other.mother_age >= preferred_mother_age
                            and other_score > cand_score + lineage_margin
                        ):
                            dominance = min(
                                1.0,
                                (other_score - cand_score - lineage_margin)
                                / max(1.0 - lineage_margin, 1e-6),
                            )
                            lineage_adjust -= dominance
            cand.lineage = float(np.clip(lineage_adjust, -1.0, 1.0))
            cand.score = float(
                np.clip(cand_score + w_margin * cand.margin + w_lineage * cand.lineage, 0.0, 1.0)
            )
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
    w_neck: float,
    w_prebud: float,
    motion_scale: float,
    interface_radius: int,
    use_border_neck: bool,
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
    prev_mask: np.ndarray | None = None

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
                neck_score = _neck_score(
                    bud_mask=masks[bud_id],
                    mother_mask=masks[mother_id],
                    interface_radius=interface_radius,
                    use_border_neck=use_border_neck,
                )
                prebud_score = _prebud_score(
                    bud_mask=masks[bud_id],
                    prev_mask=prev_mask,
                    prev_mother_mask=None if prev_mask is None else (prev_mask == mother_id),
                    interface_radius=interface_radius,
                )
                score = _score_pair(
                    dist=dist,
                    dist_max=dist_max,
                    size_ratio=size_ratio,
                    motion_score=motion_score,
                    contact_score=contact_score,
                    neck_score=neck_score,
                    prebud_score=prebud_score,
                    w_dist=w_dist,
                    w_size=w_size,
                    w_motion=w_motion,
                    w_contact=w_contact,
                    w_neck=w_neck,
                    w_prebud=w_prebud,
                )
                if score < min_score:
                    continue

                pairs.append(
                    Candidate(
                        bud_id=bud_id,
                        mother_id=mother_id,
                        frame_idx=frame_idx,
                        score=score,
                        mother_age=max(0, frame_idx - first_seen.get(mother_id, frame_idx)),
                        dist=dist,
                        size_ratio=size_ratio,
                        motion=motion_score,
                        contact=contact_score,
                        neck=neck_score,
                        prebud=prebud_score,
                        angle=1.0,
                        maturity=1.0,
                        track_quality=1.0,
                        margin=0.0,
                        lineage=0.0,
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
        prev_mask = mask

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
        f.write("bud_id,mother_id,frame,score,dist,size_ratio,motion,contact,neck,prebud,angle,track_quality,margin\n")
        for cand in assignments:
            f.write(
                f"{cand.bud_id},{cand.mother_id},{cand.frame_idx},"
                f"{cand.score:.4f},{cand.dist:.2f},{cand.size_ratio:.4f},"
                f"{cand.motion:.4f},{cand.contact:.4f},{cand.neck:.4f},"
                f"{cand.prebud:.4f},{cand.angle:.4f},{cand.track_quality:.4f},{cand.margin:.4f}\n"
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
    masks: Dict[int, np.ndarray],
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
    w_prebud: float,
    w_angle: float,
    w_maturity: float,
    w_track_quality: float,
    motion_scale: float,
    interface_radius: int,
    use_border_neck: bool,
    min_mother_age: int,
    preferred_mother_age: int,
) -> Candidate | None:
    bud_track = track_infos[bud_id]
    mother_track = track_infos[mother_id]
    end_frame = min(bud_track.end, bud_track.start + first_frames - 1)
    frame_range = range(bud_track.start, end_frame + 1)
    score_parts: List[Candidate] = []
    direction_vectors: List[Tuple[float, float]] = []
    bud_areas: List[int] = []
    dist_max_values: List[float] = []
    prebud_scores: List[float] = []
    mother_age = _track_age(mother_track, bud_track.start)
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
        neck_score = _neck_score(
            bud_mask=bin_masks[frame_idx][bud_id],
            mother_mask=bin_masks[frame_idx][mother_id],
            interface_radius=interface_radius,
            use_border_neck=use_border_neck,
        )
        prebud_score = _prebud_score(
            bud_mask=bin_masks[frame_idx][bud_id],
            prev_mask=masks.get(frame_idx - 1),
            prev_mother_mask=bin_masks.get(frame_idx - 1, {}).get(mother_id),
            interface_radius=interface_radius,
        )
        prebud_scores.append(prebud_score)
        score_parts.append(
            Candidate(
                bud_id=bud_id,
                mother_id=mother_id,
                frame_idx=frame_idx,
                score=0.0,
                mother_age=mother_age,
                dist=dist,
                size_ratio=size_ratio,
                motion=motion_score,
                contact=contact_score,
                neck=neck_score,
                prebud=prebud_score,
                angle=0.0,
                maturity=0.0,
                track_quality=0.0,
                margin=0.0,
                lineage=0.0,
            )
        )

    if not score_parts:
        return None

    support = len(score_parts)
    angle_score = _angle_consistency(direction_vectors)
    maturity_score = _maturity_score(
        mother_age=mother_age,
        min_mother_age=min_mother_age,
        preferred_mother_age=preferred_mother_age,
    )
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
        prebud_score=sum(prebud_scores) / max(len(prebud_scores), 1),
        angle_score=angle_score,
        maturity_score=maturity_score,
        track_quality=track_quality,
        w_dist=w_dist,
        w_size=w_size,
        w_motion=w_motion,
        w_contact=w_contact,
        w_neck=w_neck,
        w_prebud=w_prebud,
        w_angle=w_angle,
        w_maturity=w_maturity,
        w_track_quality=w_track_quality,
    )
    if avg_score < min_score:
        return None
    return Candidate(
        bud_id=bud_id,
        mother_id=mother_id,
        frame_idx=bud_track.start,
        score=avg_score,
        mother_age=mother_age,
        dist=sum(part.dist for part in score_parts) / support,
        size_ratio=sum(part.size_ratio for part in score_parts) / support,
        motion=sum(part.motion for part in score_parts) / support,
        contact=sum(part.contact for part in score_parts) / support,
        neck=sum(part.neck for part in score_parts) / support,
        prebud=sum(prebud_scores) / max(len(prebud_scores), 1),
        angle=angle_score,
        maturity=maturity_score,
        track_quality=track_quality,
        margin=0.0,
        lineage=0.0,
    )


def _build_global_candidates(
    track_infos: Dict[int, TrackInfo],
    masks: Dict[int, np.ndarray],
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
    w_prebud: float,
    w_angle: float,
    w_maturity: float,
    w_track_quality: float,
    w_margin: float,
    w_lineage: float,
    motion_scale: float,
    interface_radius: int,
    use_border_neck: bool,
    preferred_mother_age: int,
    lineage_margin: float,
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
                masks=masks,
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
                w_prebud=w_prebud,
                w_angle=w_angle,
                w_maturity=w_maturity,
                w_track_quality=w_track_quality,
                motion_scale=motion_scale,
                interface_radius=interface_radius,
                use_border_neck=use_border_neck,
                min_mother_age=min_mother_age,
                preferred_mother_age=preferred_mother_age,
            )
            if candidate is None:
                continue
            candidates.append(candidate)

    return candidate_buds, _apply_competition_scores(
        candidates=candidates,
        track_infos=track_infos,
        preferred_mother_age=preferred_mother_age,
        w_margin=w_margin,
        w_lineage=w_lineage,
        lineage_margin=lineage_margin,
    )


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
    w_prebud: float,
    w_angle: float,
    w_maturity: float,
    w_track_quality: float,
    w_margin: float,
    w_lineage: float,
    motion_scale: float,
    interface_radius: int,
    use_border_neck: bool,
    preferred_mother_age: int,
    lineage_margin: float,
    out_suffix: str,
    inplace: bool,
) -> None:
    from tools.parentage_api import (
        HeuristicScorer,
        ParentageConfig,
        assign_parentage,
        build_candidates,
        build_model_inputs,
        score_candidates,
        write_assignments,
    )

    cfg = ParentageConfig(
        refractory=refractory_frames,
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
        w_prebud=w_prebud,
        w_angle=w_angle,
        w_maturity=w_maturity,
        w_track_quality=w_track_quality,
        w_margin=w_margin,
        w_lineage=w_lineage,
        motion_scale=motion_scale,
        interface_radius=interface_radius,
        use_border_neck=use_border_neck,
        preferred_mother_age=preferred_mother_age,
        lineage_margin=lineage_margin,
        score_threshold=0.0,
    )
    candidate_table = build_candidates(seq_dir=seq_dir, cfg=cfg)
    if candidate_table.res_track.size == 0:
        return
    model_inputs = build_model_inputs(candidate_table=candidate_table, scorer_name="heuristic", cfg=cfg)
    scored_candidates = score_candidates(model_inputs=model_inputs, scorer=HeuristicScorer(), cfg=cfg)
    assignment = assign_parentage(candidate_table=candidate_table, scored_candidates=scored_candidates, cfg=cfg, mode="ilp")

    out_path = seq_dir / "res_track.txt" if inplace else seq_dir / f"res_track{out_suffix}.txt"
    write_assignments(assignment, out_path)

    summary_path = seq_dir / f"bud_parentage{out_suffix}.csv"
    with summary_path.open("w") as f:
        f.write("bud_id,mother_id,frame,score,mother_age,dist,size_ratio,motion,contact,neck,prebud,angle,maturity,track_quality,margin,lineage,status,num_candidates\n")
        for bud_id in sorted(candidate_table.bud_ids):
            assigned_cand = assignment.assigned_by_bud.get(bud_id)
            if assigned_cand is not None:
                cand = assigned_cand
                status = "assigned"
            else:
                best_rows = scored_candidates.scored_by_bud.get(bud_id, [])
                if best_rows:
                    cand = best_rows[0].candidate
                else:
                    cand = Candidate(
                        bud_id=bud_id,
                        mother_id=0,
                        frame_idx=candidate_table.track_infos[bud_id].start,
                        score=0.0,
                        mother_age=0,
                        dist=0.0,
                        size_ratio=0.0,
                        motion=0.0,
                        contact=0.0,
                        neck=0.0,
                        prebud=0.0,
                        angle=0.0,
                        maturity=0.0,
                        track_quality=0.0,
                        margin=0.0,
                        lineage=0.0,
                    )
                status = "orphan"
            f.write(
                f"{bud_id},{cand.mother_id},{cand.frame_idx},{cand.score:.4f},{cand.mother_age},"
                f"{cand.dist:.2f},{cand.size_ratio:.4f},{cand.motion:.4f},"
                f"{cand.contact:.4f},{cand.neck:.4f},{cand.prebud:.4f},{cand.angle:.4f},{cand.maturity:.4f},"
                f"{cand.track_quality:.4f},{cand.margin:.4f},{cand.lineage:.4f},{status},"
                f"{len(scored_candidates.scored_by_bud.get(bud_id, []))}\n"
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
    proposal_lock_margin: float,
    w_dist: float,
    w_size: float,
    w_motion: float,
    w_contact: float,
    w_neck: float,
    w_prebud: float,
    w_angle: float,
    w_maturity: float,
    w_track_quality: float,
    w_margin: float,
    w_lineage: float,
    motion_scale: float,
    interface_radius: int,
    use_border_neck: bool,
    preferred_mother_age: int,
    lineage_margin: float,
    family_unlock_margin: float,
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
    masks, stats_cache, bin_masks = _collect_frame_cache(mask_files)
    bud_ids, candidates = _build_global_candidates(
        track_infos=track_infos,
        masks=masks,
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
        w_prebud=w_prebud,
        w_angle=w_angle,
        w_maturity=w_maturity,
        w_track_quality=w_track_quality,
        w_margin=w_margin,
        w_lineage=w_lineage,
        motion_scale=motion_scale,
        interface_radius=interface_radius,
        use_border_neck=use_border_neck,
        preferred_mother_age=preferred_mother_age,
        lineage_margin=lineage_margin,
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
        bud_candidates = sorted(
            candidates_by_bud.get(bud_id, []),
            key=lambda item: item.score,
            reverse=True,
        )
        proposed_candidate = next(
            (cand for cand in bud_candidates if cand.mother_id == proposed_parent),
            None,
        )
        family_alt_score = None
        for cand in bud_candidates:
            if cand.mother_id == proposed_parent:
                continue
            if _same_lineage_family(track_infos, cand.mother_id, proposed_parent):
                family_alt_score = cand.score
                break
        if proposed_score >= proposal_lock_score:
            if proposed_candidate is not None:
                if family_alt_score is not None and proposed_candidate.score < family_alt_score + family_unlock_margin:
                    flex_buds.add(bud_id)
                    continue
            locked_assignments[bud_id] = Candidate(
                bud_id=bud_id,
                mother_id=proposed_parent,
                frame_idx=track_infos[bud_id].start,
                score=proposed_score,
                mother_age=_track_age(track_infos[proposed_parent], track_infos[bud_id].start)
                if proposed_parent in track_infos
                else 0,
                dist=0.0,
                size_ratio=0.0,
                motion=0.0,
                contact=0.0,
                neck=0.0,
                prebud=0.0,
                angle=0.0,
                maturity=0.0,
                track_quality=0.0,
                margin=0.0,
                lineage=0.0,
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
        f.write("bud_id,mother_id,frame,score,mother_age,dist,size_ratio,motion,contact,neck,prebud,angle,maturity,track_quality,margin,lineage,status,num_candidates\n")
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
                        mother_age=0,
                        dist=0.0,
                        size_ratio=0.0,
                        motion=0.0,
                        contact=0.0,
                        neck=0.0,
                        prebud=0.0,
                        angle=0.0,
                        maturity=0.0,
                        track_quality=0.0,
                        margin=0.0,
                        lineage=0.0,
                    ),
                )
                status = "orphan"
            f.write(
                f"{bud_id},{cand.mother_id},{cand.frame_idx},{cand.score:.4f},{cand.mother_age},"
                f"{cand.dist:.2f},{cand.size_ratio:.4f},{cand.motion:.4f},"
                f"{cand.contact:.4f},{cand.neck:.4f},{cand.prebud:.4f},{cand.angle:.4f},{cand.maturity:.4f},"
                f"{cand.track_quality:.4f},{cand.margin:.4f},{cand.lineage:.4f},{status},"
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
    parser.add_argument("--proposal_lock_margin", type=float, default=0.03, help="Minimum score advantage over the best alternative required to lock a proposal in hybrid mode")
    parser.add_argument("--w_dist", type=float, default=0.6, help="Weight for distance score")
    parser.add_argument("--w_size", type=float, default=0.3, help="Weight for size score")
    parser.add_argument("--w_motion", type=float, default=0.1, help="Weight for motion score")
    parser.add_argument("--w_contact", type=float, default=0.0, help="Weight for mother-bud contact score")
    parser.add_argument("--w_neck", type=float, default=0.25, help="Weight for neck/interface score")
    parser.add_argument("--w_prebud", type=float, default=0.0, help="Weight for pre-bud emergence score measured on frames before birth")
    parser.add_argument("--w_angle", type=float, default=0.20, help="Weight for angle consistency score in global and hybrid modes")
    parser.add_argument("--w_maturity", type=float, default=0.18, help="Weight for mother maturity score in global and hybrid modes")
    parser.add_argument("--w_track_quality", type=float, default=0.10, help="Weight for bud track quality score in global and hybrid modes")
    parser.add_argument("--w_margin", type=float, default=0.15, help="Weight for relative margin against competing mothers in global and hybrid modes")
    parser.add_argument("--w_lineage", type=float, default=0.10, help="Weight for intra-lineage competition adjustment in global and hybrid modes")
    parser.add_argument("--motion_scale", type=float, default=2.0, help="Scale for motion penalty")
    parser.add_argument("--interface_radius", type=int, default=4, help="Dilation radius used to measure bud-mother contact")
    parser.add_argument("--use_border_neck", action="store_true", help="Use border-focused interface score instead of the legacy neck score")
    parser.add_argument("--preferred_mother_age", type=int, default=24, help="Preferred age in frames for a mother to be considered fully mature")
    parser.add_argument("--lineage_margin", type=float, default=0.06, help="Score margin required to override an ancestor/descendant candidate within the same lineage")
    parser.add_argument("--family_unlock_margin", type=float, default=0.05, help="If a same-family alternative is within this margin of the proposal, do not lock and let the optimizer decide")
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
                proposal_lock_margin=args.proposal_lock_margin,
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
                family_unlock_margin=args.family_unlock_margin,
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
                w_neck=args.w_neck,
                w_prebud=args.w_prebud,
                motion_scale=args.motion_scale,
                interface_radius=args.interface_radius,
                use_border_neck=args.use_border_neck,
                out_suffix=args.out_suffix,
                inplace=args.inplace,
            )


if __name__ == "__main__":
    main()

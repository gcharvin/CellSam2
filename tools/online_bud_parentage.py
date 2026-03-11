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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


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

                dist_score = max(0.0, 1.0 - (dist / dist_max))
                size_score = max(0.0, 1.0 - size_ratio)
                motion_score = _motion_score(prev_centroids.get(mother_id), mother, radius, motion_scale)
                contact_score = _contact_score(
                    bud_mask=masks[bud_id],
                    mother_mask=masks[mother_id],
                    interface_radius=interface_radius,
                )
                total_weight = max(1e-6, w_dist + w_size + w_motion + w_contact)
                score = (
                    w_dist * dist_score
                    + w_size * size_score
                    + w_motion * motion_score
                    + w_contact * contact_score
                ) / total_weight
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
        f.write("bud_id,mother_id,frame,score,dist,size_ratio,motion,contact\n")
        for cand in assignments:
            f.write(
                f"{cand.bud_id},{cand.mother_id},{cand.frame_idx},"
                f"{cand.score:.4f},{cand.dist:.2f},{cand.size_ratio:.4f},"
                f"{cand.motion:.4f},{cand.contact:.4f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Online bud parentage post-processing")
    parser.add_argument("--pred_root", required=True, help="Prediction root (sequence dir or parent)")
    parser.add_argument("--refractory", type=int, default=8, help="Refractory frames per mother")
    parser.add_argument("--max_dist_factor", type=float, default=2.5, help="Max dist = factor * mother radius")
    parser.add_argument("--bud_max_area_ratio", type=float, default=0.6, help="Max bud/mother area ratio")
    parser.add_argument("--min_bud_area", type=int, default=10, help="Minimum bud area")
    parser.add_argument("--min_mother_age", type=int, default=3, help="Min age (frames) for a mother candidate")
    parser.add_argument("--min_score", type=float, default=0.1, help="Minimum score for assignment")
    parser.add_argument("--w_dist", type=float, default=0.6, help="Weight for distance score")
    parser.add_argument("--w_size", type=float, default=0.3, help="Weight for size score")
    parser.add_argument("--w_motion", type=float, default=0.1, help="Weight for motion score")
    parser.add_argument("--w_contact", type=float, default=0.0, help="Weight for mother-bud contact score")
    parser.add_argument("--motion_scale", type=float, default=2.0, help="Scale for motion penalty")
    parser.add_argument("--interface_radius", type=int, default=4, help="Dilation radius used to measure bud-mother contact")
    parser.add_argument("--out_suffix", type=str, default="_parented", help="Suffix for output files")
    parser.add_argument("--inplace", action="store_true", help="Overwrite res_track.txt")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    for seq_dir in _list_sequence_dirs(pred_root):
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

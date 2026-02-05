#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import numpy as np
import cv2


def load_mask(path):
    return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)


def compute_iou_matrix(gt_mask, pred_mask, gt_ids, pred_ids):
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    for i, gid in enumerate(gt_ids):
        g = gt_mask == gid
        gsum = g.sum()
        if gsum == 0:
            continue
        for j, pid in enumerate(pred_ids):
            p = pred_mask == pid
            psum = p.sum()
            if psum == 0:
                continue
            inter = np.logical_and(g, p).sum()
            if inter == 0:
                continue
            union = gsum + psum - inter
            iou[i, j] = inter / union
    return iou


def greedy_match(iou, gt_ids, pred_ids, thr):
    matches = {}
    if iou.size == 0:
        for gid in gt_ids:
            matches[int(gid)] = None
        return matches
    pairs = []
    for i in range(iou.shape[0]):
        for j in range(iou.shape[1]):
            if iou[i, j] >= thr:
                pairs.append((iou[i, j], i, j))
    pairs.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    for score, i, j in pairs:
        gid = int(gt_ids[i])
        pid = int(pred_ids[j])
        if gid in used_gt or pid in used_pred:
            continue
        used_gt.add(gid)
        used_pred.add(pid)
        matches[gid] = pid
    for gid in gt_ids:
        if int(gid) not in matches:
            matches[int(gid)] = None
    return matches


def compute_metrics_for_sequence(gt_dir, pred_dir, seq, iou_thr):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    gt_masks = sorted(gt_dir.glob('man_track*.tif'))
    if not gt_masks:
        raise FileNotFoundError(f'No GT masks in {gt_dir}')

    def frame_from_name(p, prefix):
        name = p.stem
        return int(name.replace(prefix, ''))

    gt_by_frame = {frame_from_name(p, 'man_track'): p for p in gt_masks}
    pred_masks = sorted(pred_dir.glob('mask*.tif'))
    pred_by_frame = {frame_from_name(p, 'mask'): p for p in pred_masks}

    frames = sorted(gt_by_frame.keys())

    gt_history = {}
    gt_present = {}

    total_gt_dets = 0
    total_matched = 0

    for f in frames:
        gt_mask = load_mask(gt_by_frame[f])
        pred_path = pred_by_frame.get(f)
        if pred_path is None:
            pred_mask = np.zeros_like(gt_mask)
        else:
            pred_mask = load_mask(pred_path)

        gt_ids = np.unique(gt_mask)
        gt_ids = gt_ids[gt_ids != 0]
        pred_ids = np.unique(pred_mask)
        pred_ids = pred_ids[pred_ids != 0]

        total_gt_dets += len(gt_ids)

        iou = compute_iou_matrix(gt_mask, pred_mask, gt_ids, pred_ids)
        matches = greedy_match(iou, gt_ids, pred_ids, iou_thr)

        for gid in gt_ids:
            gid = int(gid)
            pid = matches.get(gid)
            if gid not in gt_history:
                gt_history[gid] = []
                gt_present[gid] = 0
            gt_history[gid].append(pid)
            if pid is not None:
                total_matched += 1
            gt_present[gid] += 1

    switches = 0
    fragments = 0
    matched_steps = 0
    matched_tracks = 0

    for gid, seq_ids in gt_history.items():
        prev = None
        in_segment = False
        track_switches = 0
        track_fragments = 0
        track_matched = 0

        for pid in seq_ids:
            if pid is None:
                if in_segment:
                    track_fragments += 1
                    in_segment = False
                prev = None
                continue
            track_matched += 1
            if prev is None:
                in_segment = True
            else:
                if pid != prev:
                    track_switches += 1
            prev = pid
        if in_segment:
            track_fragments += 1

        if track_matched > 0:
            matched_tracks += 1
            switches += track_switches
            fragments += max(0, track_fragments - 1)
            matched_steps += track_matched

    coverage = total_matched / max(1, total_gt_dets)
    switch_rate = switches / max(1, matched_steps - matched_tracks)
    fragment_rate = fragments / max(1, matched_tracks)

    return {
        'seq': seq,
        'total_gt_dets': total_gt_dets,
        'total_matched': total_matched,
        'coverage': coverage,
        'id_switches': switches,
        'switch_rate': switch_rate,
        'fragments': fragments,
        'fragment_rate': fragment_rate,
        'matched_steps': matched_steps,
        'matched_tracks': matched_tracks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_root', required=True)
    ap.add_argument('--pred_root', required=True)
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--iou_thr', type=float, default=0.3)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    results = {'per_sequence': {}, 'overall': {}}
    totals = {
        'total_gt_dets': 0,
        'total_matched': 0,
        'id_switches': 0,
        'fragments': 0,
        'matched_steps': 0,
        'matched_tracks': 0,
    }

    for seq in args.seqs:
        gt_dir = Path(args.gt_root) / f'{seq}_GT' / 'TRA'
        pred_dir = Path(args.pred_root) / seq
        metrics = compute_metrics_for_sequence(gt_dir, pred_dir, seq, args.iou_thr)
        results['per_sequence'][seq] = metrics

        for k in totals:
            totals[k] += metrics[k]

    coverage = totals['total_matched'] / max(1, totals['total_gt_dets'])
    switch_rate = totals['id_switches'] / max(1, totals['matched_steps'] - totals['matched_tracks'])
    fragment_rate = totals['fragments'] / max(1, totals['matched_tracks'])

    results['overall'] = {
        'coverage': coverage,
        'id_switches': totals['id_switches'],
        'switch_rate': switch_rate,
        'fragments': totals['fragments'],
        'fragment_rate': fragment_rate,
        'matched_steps': totals['matched_steps'],
        'matched_tracks': totals['matched_tracks'],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

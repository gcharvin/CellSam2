#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def _split_list(values):
    items = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.append(float(part))
    return items


def load_track(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            cell_id, start, end, parent = map(int, parts[:4])
            rows.append((cell_id, start, end, parent))
    return rows


def division_events(rows):
    return [(start, parent, cell_id) for (cell_id, start, end, parent) in rows if parent != 0]


def match_by_start(gt_events, pred_events, tol):
    used_pred = set()
    matches = 0
    for gstart, gparent, gid in gt_events:
        match_idx = None
        for idx, (pstart, pparent, pid) in enumerate(pred_events):
            if idx in used_pred:
                continue
            if abs(pstart - gstart) <= tol:
                match_idx = idx
                break
        if match_idx is not None:
            used_pred.add(match_idx)
            matches += 1
    return matches


def compute_metrics(gt_events, pred_events, tol, w_precision, w_recall):
    matches = match_by_start(gt_events, pred_events, tol)
    gt_n = len(gt_events)
    pred_n = len(pred_events)
    precision = matches / pred_n if pred_n else 0.0
    recall = matches / gt_n if gt_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    score = w_precision * precision + w_recall * recall
    return matches, gt_n, pred_n, precision, recall, f1, score


def run_inference(args, div_thresh, obj_thresh, iou_thresh, out_dir, seqs):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "inference/track_cells.py",
        "--video_path",
        args.video_path,
        "--model_name",
        args.model_name,
        "--checkpoint_num",
        str(args.checkpoint_num),
        "--res_path",
        str(out_dir),
        "--pred_iou_thresh",
        str(iou_thresh),
        "--div_obj_score_thresh",
        str(div_thresh),
        "--obj_score_thresh",
        str(obj_thresh),
    ]
    if args.extra_args:
        cmd.extend(args.extra_args)
    subprocess.run(cmd, check=True)


def seq_ready(out_dir, seqs):
    return all((out_dir / seq / "res_track.txt").exists() for seq in seqs)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep division thresholds and score against GT tracks."
    )
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--gt_root", default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--checkpoint_num", type=int, required=True)
    parser.add_argument("--res_root", required=True)
    parser.add_argument("--seqs", default="12,13,14")
    parser.add_argument("--div_obj_score_thresh", nargs="+", default=["-4,-3,-2,-1,0"])
    parser.add_argument("--obj_score_thresh", nargs="+", default=["0,0.25,0.5"])
    parser.add_argument("--pred_iou_thresh", nargs="+", default=["0.2,0.4,0.6"])
    parser.add_argument("--match_tol", type=int, default=1)
    parser.add_argument("--w_precision", type=float, default=0.5)
    parser.add_argument("--w_recall", type=float, default=0.5)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    div_list = _split_list(args.div_obj_score_thresh)
    obj_list = _split_list(args.obj_score_thresh)
    iou_list = _split_list(args.pred_iou_thresh)

    if args.gt_root is None:
        gt_root = Path(args.video_path)
    else:
        gt_root = Path(args.gt_root)
    res_root = Path(args.res_root)
    res_root.mkdir(parents=True, exist_ok=True)

    results = []
    for div_thresh, obj_thresh, iou_thresh in itertools.product(div_list, obj_list, iou_list):
        name = f"div{div_thresh}_obj{obj_thresh}_iou{iou_thresh}".replace(".", "p").replace("-", "m")
        out_dir = res_root / name
        if not (args.reuse and seq_ready(out_dir, seqs)):
            run_inference(args, div_thresh, obj_thresh, iou_thresh, out_dir, seqs)

        total_gt = total_pred = total_matches = 0
        for seq in seqs:
            gt_path = gt_root / f"{seq}_GT" / "TRA" / "man_track.txt"
            pred_path = out_dir / seq / "res_track.txt"
            gt_events = division_events(load_track(gt_path))
            pred_events = division_events(load_track(pred_path))
            matches, gt_n, pred_n, *_ = compute_metrics(
                gt_events, pred_events, args.match_tol, args.w_precision, args.w_recall
            )
            total_gt += gt_n
            total_pred += pred_n
            total_matches += matches

        precision = total_matches / total_pred if total_pred else 0.0
        recall = total_matches / total_gt if total_gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        score = args.w_precision * precision + args.w_recall * recall

        results.append(
            {
                "div_obj_score_thresh": div_thresh,
                "obj_score_thresh": obj_thresh,
                "pred_iou_thresh": iou_thresh,
                "gt_div": total_gt,
                "pred_div": total_pred,
                "matches": total_matches,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "score": score,
                "out_dir": str(out_dir),
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    out_tsv = res_root / f"sweep_results_tol{args.match_tol}.tsv"
    with out_tsv.open("w", encoding="utf-8") as handle:
        header = [
            "div_obj_score_thresh",
            "obj_score_thresh",
            "pred_iou_thresh",
            "gt_div",
            "pred_div",
            "matches",
            "precision",
            "recall",
            "f1",
            "score",
            "out_dir",
        ]
        handle.write("\t".join(header) + "\n")
        for row in results:
            handle.write(
                "\t".join(
                    [
                        str(row["div_obj_score_thresh"]),
                        str(row["obj_score_thresh"]),
                        str(row["pred_iou_thresh"]),
                        str(row["gt_div"]),
                        str(row["pred_div"]),
                        str(row["matches"]),
                        f'{row["precision"]:.4f}',
                        f'{row["recall"]:.4f}',
                        f'{row["f1"]:.4f}',
                        f'{row["score"]:.4f}',
                        row["out_dir"],
                    ]
                )
                + "\n"
            )

    print(f"Wrote results to {out_tsv}")
    print("Top configs:")
    for row in results[: args.top_k]:
        print(
            f'div={row["div_obj_score_thresh"]} obj={row["obj_score_thresh"]} '
            f'iou={row["pred_iou_thresh"]} score={row["score"]:.4f} '
            f'prec={row["precision"]:.3f} rec={row["recall"]:.3f} f1={row["f1"]:.3f}'
        )


if __name__ == "__main__":
    main()

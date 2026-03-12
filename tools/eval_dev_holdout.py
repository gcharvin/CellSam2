#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_utils import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a non-learned heuristic on train-dev and val-holdout splits."
    )
    parser.add_argument("--experiment", required=True, help="Experiment directory.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing train/CTC and val/CTC.")
    parser.add_argument("--train-pred-root", required=True, help="Prediction root for train-dev split.")
    parser.add_argument("--val-pred-root", required=True, help="Prediction root for val-holdout split.")
    parser.add_argument("--train-videos", default="", help="Comma-separated train video ids. Defaults to all train sequences.")
    parser.add_argument("--val-videos", default="", help="Comma-separated val video ids. Defaults to all val sequences.")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--analysis-frames", type=int, default=5)
    parser.add_argument("--birth-search-window", type=int, default=3)
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    parser.add_argument("--stable-support", type=int, default=3)
    parser.add_argument("--stable-fraction", type=float, default=0.6)
    parser.add_argument("--parent-map-threshold", type=float, default=0.5)
    return parser.parse_args()


def split_videos(ctc_root: Path, explicit: str) -> list[str]:
    if explicit.strip():
        return [item.strip() for item in explicit.split(",") if item.strip()]
    return sorted(
        path.name
        for path in ctc_root.iterdir()
        if path.is_dir() and not path.name.endswith("_GT")
    )


def run_python(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def evaluate_split(
    split_name: str,
    gt_root: Path,
    pred_root: Path,
    videos: list[str],
    args: argparse.Namespace,
    eval_root: Path,
    review_root: Path,
) -> dict:
    split_eval_dir = eval_root / split_name
    split_review_dir = review_root / split_name
    split_eval_dir.mkdir(parents=True, exist_ok=True)
    split_review_dir.mkdir(parents=True, exist_ok=True)

    div_json = split_eval_dir / "division_iou_window.json"
    diag_json = split_eval_dir / "bud_event_diagnostic.json"
    diag_md = split_review_dir / "bud_event_diagnostic.md"
    video_arg = ",".join(videos)

    run_python(
        [
            "tools/eval_division_iou_window.py",
            "--gt-root",
            str(gt_root),
            "--pred-root",
            str(pred_root),
            "--videos",
            video_arg,
            "--window",
            str(args.window),
            "--iou-thresh",
            str(args.iou_thresh),
            "--save-json",
            str(div_json),
        ]
    )
    run_python(
        [
            "tools/diagnose_bud_events.py",
            "--gt-root",
            str(gt_root),
            "--pred-root",
            str(pred_root),
            "--videos",
            video_arg,
            "--analysis-frames",
            str(args.analysis_frames),
            "--birth-search-window",
            str(args.birth_search_window),
            "--overlap-threshold",
            str(args.overlap_threshold),
            "--stable-support",
            str(args.stable_support),
            "--stable-fraction",
            str(args.stable_fraction),
            "--parent-map-threshold",
            str(args.parent_map_threshold),
            "--save-json",
            str(diag_json),
            "--save-markdown",
            str(diag_md),
        ]
    )

    div_metrics = json.loads(div_json.read_text(encoding="utf-8"))
    diag_metrics = json.loads(diag_json.read_text(encoding="utf-8"))
    return {
        "videos": videos,
        "gt_root": str(gt_root),
        "pred_root": str(pred_root),
        "division_iou_window": div_metrics,
        "bud_event_diagnostic": diag_metrics,
    }


def review_markdown(summary: dict) -> str:
    train = summary["splits"]["train_dev"]
    val = summary["splits"]["val_holdout"]
    train_div = train["division_iou_window"]["overall"]
    val_div = val["division_iou_window"]["overall"]
    train_diag = train["bud_event_diagnostic"]["overall"]["categories"]
    val_diag = val["bud_event_diagnostic"]["overall"]["categories"]
    return (
        "# Dev Holdout Summary\n\n"
        "## Protocol\n\n"
        "- `train_dev` is used only for heuristic development and tuning.\n"
        "- `val_holdout` is kept as the comparison split for decisions.\n\n"
        "## Train Dev\n\n"
        f"- Videos: `{', '.join(train['videos'])}`\n"
        f"- `division_iou_window_f1`: `{train_div['f1']}`\n"
        f"- `division_iou_window_precision`: `{train_div['precision']}`\n"
        f"- `division_iou_window_recall`: `{train_div['recall']}`\n"
        f"- `stable_correct_parent`: `{train_diag.get('stable_correct_parent', 0)}`\n"
        f"- `stable_wrong_parent`: `{train_diag.get('stable_wrong_parent', 0)}`\n"
        f"- `stable_orphan`: `{train_diag.get('stable_orphan', 0)}`\n\n"
        "## Val Holdout\n\n"
        f"- Videos: `{', '.join(val['videos'])}`\n"
        f"- `division_iou_window_f1`: `{val_div['f1']}`\n"
        f"- `division_iou_window_precision`: `{val_div['precision']}`\n"
        f"- `division_iou_window_recall`: `{val_div['recall']}`\n"
        f"- `stable_correct_parent`: `{val_diag.get('stable_correct_parent', 0)}`\n"
        f"- `stable_wrong_parent`: `{val_diag.get('stable_wrong_parent', 0)}`\n"
        f"- `stable_orphan`: `{val_diag.get('stable_orphan', 0)}`\n\n"
        "## Reading Rule\n\n"
        "- Keep a heuristic only if gains seen on `train_dev` do not collapse on `val_holdout`.\n"
    )


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    eval_root = experiment_dir / "eval"
    review_root = experiment_dir / "review"

    train_gt_root = dataset_root / "train" / "CTC"
    val_gt_root = dataset_root / "val" / "CTC"
    train_videos = split_videos(train_gt_root, args.train_videos)
    val_videos = split_videos(val_gt_root, args.val_videos)

    summary = {
        "protocol": {
            "train_dev_role": "heuristic development and tuning only",
            "val_holdout_role": "decision split for non-learned post-processing",
        },
        "splits": {
            "train_dev": evaluate_split(
                split_name="train_dev",
                gt_root=train_gt_root,
                pred_root=Path(args.train_pred_root).expanduser().resolve(),
                videos=train_videos,
                args=args,
                eval_root=eval_root,
                review_root=review_root,
            ),
            "val_holdout": evaluate_split(
                split_name="val_holdout",
                gt_root=val_gt_root,
                pred_root=Path(args.val_pred_root).expanduser().resolve(),
                videos=val_videos,
                args=args,
                eval_root=eval_root,
                review_root=review_root,
            ),
        },
    }

    write_json(eval_root / "dev_holdout_summary.json", summary)
    write_text(review_root / "dev_holdout_summary.md", review_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

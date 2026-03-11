#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_utils import load_json, write_json, write_text


def parse_key_value(raw_value: str) -> tuple[str, str]:
    if "=" not in raw_value:
        raise ValueError(f"Expected key=value, got: {raw_value}")
    key, value = raw_value.split("=", 1)
    return key.strip(), value.strip()


def parse_metric_value(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update machine-readable and reviewable experiment summaries."
    )
    parser.add_argument("--experiment", required=True, help="Path to experiment directory.")
    parser.add_argument(
        "--status",
        default=None,
        choices=["planned", "running", "done", "failed", "reviewed"],
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--decision", default="")
    parser.add_argument("--next-step", action="append", default=[])
    parser.add_argument("--goal", default="")
    parser.add_argument("--changes", action="append", default=[])
    parser.add_argument("--interpretation", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment).expanduser().resolve()
    manifest_path = experiment_dir / "manifest.yaml"
    summary_json_path = experiment_dir / "eval" / "summary.json"
    review_md_path = experiment_dir / "review" / "summary.md"

    manifest = load_json(manifest_path)
    summary = load_json(summary_json_path)

    metrics = summary.setdefault("metrics", {})
    for raw_metric in args.metric:
        key, value = parse_key_value(raw_metric)
        metrics[key] = parse_metric_value(value)

    if args.note:
        summary.setdefault("notes", []).extend(args.note)

    if args.checkpoint:
        manifest.setdefault("artifacts", {})["best_checkpoint"] = args.checkpoint
        summary["best_checkpoint"] = args.checkpoint

    if args.status:
        manifest["status"] = args.status
        summary["status"] = args.status

    if args.decision:
        summary["decision"] = args.decision

    if args.next_step:
        summary["next_steps"] = args.next_step

    write_json(manifest_path, manifest)
    write_json(summary_json_path, summary)

    cellsam2 = manifest["repos"]["cellsam2"]
    cellhota = manifest["repos"].get("cellhota", {})
    dataset = manifest.get("dataset", {})
    result_lines = [f"- `{key}`: `{value}`" for key, value in sorted(metrics.items())]
    review_md = (
        "# Experiment Summary\n\n"
        "## Goal\n\n"
        f"- {args.goal or manifest.get('tag', '')}\n\n"
        "## Code\n\n"
        f"- `CellSam2` commit: `{cellsam2.get('commit', '')}`\n"
        f"- `Cell-HOTA` commit: `{cellhota.get('commit', '')}`\n\n"
        "## Data\n\n"
        f"- Dataset: `{dataset.get('name', '')}`\n"
        f"- Path: `{dataset.get('path', '')}`\n\n"
        "## Changes\n\n"
        + ("\n".join(f"- {item}" for item in args.changes) if args.changes else "- Not recorded.\n")
        + "\n\n## Results\n\n"
        + ("\n".join(result_lines) if result_lines else "- No metrics recorded.\n")
        + "\n"
        + f"- Best checkpoint: `{manifest.get('artifacts', {}).get('best_checkpoint', '')}`\n\n"
        + "## Interpretation\n\n"
        + ("\n".join(f"- {item}" for item in args.interpretation) if args.interpretation else "- Not recorded.\n")
        + "\n\n## Decision\n\n"
        + (f"- {args.decision}\n" if args.decision else "- Not recorded.\n")
        + "\n## Next Step\n\n"
        + ("\n".join(f"- {item}" for item in args.next_step) if args.next_step else "- Not recorded.\n")
    )
    write_text(review_md_path, review_md)

    print(summary_json_path)
    print(review_md_path)


if __name__ == "__main__":
    main()

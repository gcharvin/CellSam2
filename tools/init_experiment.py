#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_utils import (
    build_exp_id,
    capture_git_env,
    capture_python_env,
    default_machine,
    default_review_summary,
    detect_experiments_root,
    ensure_layout,
    git_snapshot,
    iso_now,
    write_json,
    write_text,
)


DEFAULT_DATASET_PATH = (
    "/homes/Gilles/Data/DetecDivProjects/anais/bud4/classification/"
    "celltracktr_5/trainingdataset/moma"
)
DEFAULT_CELLHOTA_PATH = "/home/charvin-admin/Documents/github/Cell-HOTA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a structured experiment directory and capture provenance."
    )
    parser.add_argument("--tag", required=True, help="Short experiment tag.")
    parser.add_argument("--dataset", default="moma", help="Dataset name.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--cellsam2-repo", default=str(ROOT))
    parser.add_argument("--cellhota-repo", default=DEFAULT_CELLHOTA_PATH)
    parser.add_argument("--machine", default=default_machine())
    parser.add_argument("--train-cmd", default="")
    parser.add_argument("--inference-cmd", default="")
    parser.add_argument("--evaluation-cmd", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--status",
        default="planned",
        choices=["planned", "running", "done", "failed", "reviewed"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cellsam2_repo = Path(args.cellsam2_repo).expanduser().resolve()
    experiments_root = detect_experiments_root(args.experiments_root)
    exp_id = build_exp_id(cellsam2_repo, args.tag)
    experiment_dir = experiments_root / exp_id
    ensure_layout(experiment_dir)

    cellsam2_git = git_snapshot(cellsam2_repo)

    cellhota_manifest = None
    cellhota_path = Path(args.cellhota_repo).expanduser().resolve()
    if cellhota_path.exists():
        cellhota_git = git_snapshot(cellhota_path)
        cellhota_manifest = {
            "path": str(cellhota_git.path),
            "branch": cellhota_git.branch,
            "commit": cellhota_git.commit,
            "dirty": cellhota_git.dirty,
        }
        write_text(experiment_dir / "env" / "git_cellhota.txt", capture_git_env(cellhota_git))
        write_text(experiment_dir / "env" / "cellhota.diff", cellhota_git.diff)

    manifest = {
        "exp_id": exp_id,
        "tag": args.tag,
        "status": args.status,
        "created_at": iso_now(),
        "machine": args.machine,
        "dataset": {
            "name": args.dataset,
            "path": args.dataset_path,
        },
        "repos": {
            "cellsam2": {
                "path": str(cellsam2_git.path),
                "branch": cellsam2_git.branch,
                "commit": cellsam2_git.commit,
                "dirty": cellsam2_git.dirty,
            },
        },
        "commands": {
            "train": args.train_cmd,
            "inference": args.inference_cmd,
            "evaluation": args.evaluation_cmd,
        },
        "artifacts": {
            "best_checkpoint": "",
            "notes": args.notes,
        },
    }
    if cellhota_manifest is not None:
        manifest["repos"]["cellhota"] = cellhota_manifest

    write_json(experiment_dir / "manifest.yaml", manifest)
    write_text(experiment_dir / "env" / "cellsam2_env.txt", capture_python_env())
    write_text(experiment_dir / "env" / "git_cellsam2.txt", capture_git_env(cellsam2_git))
    write_text(experiment_dir / "env" / "cellsam2.diff", cellsam2_git.diff)
    write_json(experiment_dir / "eval" / "summary.json", {"exp_id": exp_id, "metrics": {}, "notes": []})
    write_text(experiment_dir / "review" / "summary.md", default_review_summary(manifest))

    template_manifest = ROOT / "experiment_templates" / "manifest.yaml"
    if template_manifest.exists():
        shutil.copyfile(template_manifest, experiment_dir / "review" / "manifest_template.yaml")

    print(experiment_dir)


if __name__ == "__main__":
    main()

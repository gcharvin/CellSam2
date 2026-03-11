import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT.parent / "experiments"


@dataclass
class GitSnapshot:
    path: Path
    branch: str
    commit: str
    dirty: bool
    status: str
    remotes: str
    diff: str


def sanitize_tag(tag: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", tag.strip().lower())
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        raise ValueError("tag must contain at least one alphanumeric character")
    return value


def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_machine() -> str:
    return socket.gethostname() or platform.node() or "unknown-host"


def run_cmd(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def safe_run_cmd(args: list[str], cwd: Path | None = None) -> str:
    try:
        return run_cmd(args, cwd=cwd)
    except Exception as exc:
        return f"<command failed: {' '.join(shlex.quote(a) for a in args)}>\n{exc}\n"


def detect_experiments_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    env_root = os.environ.get("CELLSAM2_EXPERIMENTS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DEFAULT_EXPERIMENTS_ROOT.resolve()


def build_exp_id(repo_path: Path, tag: str) -> str:
    commit = safe_run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo_path).strip()
    return f"{now_string()}_{commit}_{sanitize_tag(tag)}"


def git_snapshot(repo_path: Path) -> GitSnapshot:
    branch = safe_run_cmd(["git", "branch", "--show-current"], cwd=repo_path).strip()
    commit = safe_run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_path).strip()
    status = safe_run_cmd(["git", "status", "--short"], cwd=repo_path)
    remotes = safe_run_cmd(["git", "remote", "-v"], cwd=repo_path)
    diff = safe_run_cmd(["git", "diff", "HEAD"], cwd=repo_path)
    return GitSnapshot(
        path=repo_path,
        branch=branch,
        commit=commit,
        dirty=bool(status.strip()),
        status=status,
        remotes=remotes,
        diff=diff,
    )


def ensure_layout(experiment_dir: Path) -> None:
    for rel_path in (
        "env",
        "train/checkpoints",
        "inference/val",
        "eval/cell_hota",
        "review/plots",
    ):
        (experiment_dir / rel_path).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def capture_python_env() -> str:
    python_exe = sys.executable
    version_cmd = "import sys; print(sys.version.replace(chr(10), ' '))"
    python_version = safe_run_cmd([python_exe, "-c", version_cmd]).strip()
    lines = [
        f"created_at={iso_now()}",
        f"python_executable={python_exe}",
        f"python_version={python_version}",
    ]
    pip_freeze = safe_run_cmd([python_exe, "-m", "pip", "freeze"])
    return "\n".join(lines) + "\n\n[pip_freeze]\n" + pip_freeze


def capture_git_env(snapshot: GitSnapshot) -> str:
    return (
        f"path={snapshot.path}\n"
        f"branch={snapshot.branch}\n"
        f"commit={snapshot.commit}\n"
        f"dirty={str(snapshot.dirty).lower()}\n\n"
        f"[status]\n{snapshot.status}\n"
        f"[remotes]\n{snapshot.remotes}"
    )


def default_review_summary(manifest: dict[str, Any]) -> str:
    cellsam2 = manifest["repos"]["cellsam2"]
    cellhota = manifest["repos"].get("cellhota", {})
    dataset = manifest.get("dataset", {})
    checkpoint = manifest.get("artifacts", {}).get("best_checkpoint", "")
    return (
        "# Experiment Summary\n\n"
        "## Goal\n\n"
        f"- {manifest.get('tag', '')}\n\n"
        "## Code\n\n"
        f"- `CellSam2` commit: `{cellsam2.get('commit', '')}`\n"
        f"- `Cell-HOTA` commit: `{cellhota.get('commit', '')}`\n\n"
        "## Data\n\n"
        f"- Dataset: `{dataset.get('name', '')}`\n"
        f"- Path: `{dataset.get('path', '')}`\n\n"
        "## Changes\n\n"
        "- To fill after the run.\n\n"
        "## Results\n\n"
        "- Main metrics: to fill.\n"
        f"- Best checkpoint: `{checkpoint}`\n\n"
        "## Interpretation\n\n"
        "- To fill after the run.\n\n"
        "## Decision\n\n"
        "- To fill after the run.\n\n"
        "## Next Step\n\n"
        "- To fill after the run.\n"
    )

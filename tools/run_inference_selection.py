#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
import os
import re
import sys

# Ensure repo root on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.run_inference_for_runs import (
    ensure_worktree,
    ensure_sam2_logs_link,
    ensure_checkpoints_files,
    patch_tkinter,
    patch_ctc_assert,
)


def checkpoint_num_from_name(name):
    m = re.search(r'checkpoint_(\d+)\.', name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--worktrees_dir', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--checkpoint', required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    wt = ensure_worktree(repo, args.commit, args.worktrees_dir)
    ensure_sam2_logs_link(repo, wt)
    ensure_checkpoints_files(repo, wt)
    patch_tkinter(wt)
    # Some older commits assert on ID list equality; patch it out to avoid aborting.
    patch_ctc_assert(wt)

    ckpt_num = checkpoint_num_from_name(args.checkpoint)
    if ckpt_num is None:
        raise SystemExit('Invalid checkpoint name')

    out_root = Path(args.out_root).resolve()
    out_dir = out_root / args.run / args.commit[:8]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'python',
        str(wt / 'inference' / 'track_cells.py'),
        '--model_name', args.run,
        '--checkpoint_num', str(ckpt_num),
        '--video_path', args.data_root,
        '--res_path', str(out_dir),
    ]
    print('Running', args.run, 'commit', args.commit[:8], 'ckpt', ckpt_num)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(wt)
    subprocess.run(cmd, check=True, cwd=str(wt), env=env)


if __name__ == '__main__':
    main()

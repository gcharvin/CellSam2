#!/usr/bin/env python
import argparse
from pathlib import Path
import subprocess
import re


def should_skip(name):
    lowered = name.lower()
    bad = ['debug','smoke','test','align','loss_alignment','ablation']
    return any(b in lowered for b in bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--commits', nargs='+', required=True)
    ap.add_argument('--min_epoch', type=int, default=10)
    args = ap.parse_args()

    repo = Path(args.repo)
    logs = repo / 'sam2_logs'

    # collect runs with latest checkpoint number and mtime
    runs = []
    for run in logs.iterdir():
        if not run.is_dir():
            continue
        if should_skip(run.name):
            continue
        ckpt_dir = run / 'checkpoints'
        if not ckpt_dir.exists():
            continue
        ckpts = list(ckpt_dir.glob('checkpoint_*.pt')) + list(ckpt_dir.glob('checkpoint_*.pth'))
        if not ckpts:
            continue
        latest = max(ckpts, key=lambda p: p.stat().st_mtime)
        m = re.search(r'checkpoint_(\d+)\.', latest.name)
        if not m:
            continue
        epoch = int(m.group(1))
        runs.append((run.name, latest, latest.stat().st_mtime, epoch))

    # map to each commit: choose first run after commit time, with epoch>=min_epoch, by earliest mtime
    for commit in args.commits:
        ts = int(subprocess.check_output(['git','-C',str(repo),'show','-s','--format=%ct',commit]).decode().strip())
        cand = [r for r in runs if r[2] >= ts and r[3] >= args.min_epoch]
        cand = sorted(cand, key=lambda x: x[2])
        if not cand:
            print(commit, 'NO_RUN')
            continue
        name, ckpt, mtime, epoch = cand[0]
        print(commit, name, ckpt.name, epoch)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import subprocess


def should_skip(name):
    lowered = name.lower()
    bad = ['debug', 'smoke', 'test', 'align', 'loss_alignment']
    return any(b in lowered for b in bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sam2_logs', required=True)
    ap.add_argument('--gt_root', required=True)
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--iou_thr', type=float, default=0.3)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    logs = Path(args.sam2_logs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for run in sorted(logs.iterdir()):
        if not run.is_dir():
            continue
        if should_skip(run.name):
            continue
        pred_root = run / 'results' / 'val' / 'CTC'
        if not pred_root.exists():
            continue
        # require masks for all seqs
        ok = True
        for s in args.seqs:
            if not (pred_root / s).exists():
                ok = False
                break
        if not ok:
            continue

        out_path = out_dir / f'{run.name}.json'
        cmd = [
            'python',
            str(Path(__file__).parent / 'tracklet_persistence.py'),
            '--gt_root', args.gt_root,
            '--pred_root', str(pred_root),
            '--seqs', *args.seqs,
            '--iou_thr', str(args.iou_thr),
            '--out', str(out_path),
        ]
        subprocess.run(cmd, check=True)
        data = json.loads(out_path.read_text())
        overall = data['overall']
        results.append({
            'run': run.name,
            'coverage': overall['coverage'],
            'switch_rate': overall['switch_rate'],
            'fragments': overall['fragments'],
        })

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(results, indent=2))

    # also write a simple CSV
    csv_path = out_dir / 'summary.csv'
    with open(csv_path, 'w') as f:
        f.write('run,coverage,switch_rate,fragments\n')
        for r in sorted(results, key=lambda x: (x['switch_rate'], -x['coverage'])):
            f.write(f"{r['run']},{r['coverage']:.6f},{r['switch_rate']:.6f},{r['fragments']}\n")


if __name__ == '__main__':
    main()

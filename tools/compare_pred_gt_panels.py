#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import cv2


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_mask(path):
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)


def load_tracks(res_track_path):
    if not res_track_path.exists():
        return {}
    rows = []
    with res_track_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append([int(p) for p in parts[:4]])
    tracks = {}
    for cell_id, start, end, parent in rows:
        tracks[cell_id] = {"start": start, "end": end, "parent": parent}
    return tracks


def compute_bud_ids(tracks):
    if not tracks:
        return {}
    has_parent = any(v["parent"] != 0 for v in tracks.values())
    buds = {}
    for cell_id, info in tracks.items():
        if has_parent:
            if info["parent"] != 0:
                buds[cell_id] = info
        else:
            if info["start"] > 0:
                buds[cell_id] = info
    return buds


def color_for_id(cell_id):
    rng = np.random.RandomState(cell_id * 9973 % 2**31)
    return rng.randint(0, 255, size=3).tolist()


def overlay_masks(base_img, mask, alpha=0.35):
    if mask is None:
        return base_img
    overlay = base_img.copy()
    ids = np.unique(mask)
    ids = ids[ids != 0]
    for cell_id in ids:
        color = color_for_id(int(cell_id))
        overlay[mask == cell_id] = color
    return cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)


def highlight_buds(img, mask, buds, frame_idx, color=(0, 0, 255)):
    if mask is None or not buds:
        return img
    out = img.copy()
    for bud_id, info in buds.items():
        if frame_idx < info["start"] or frame_idx > info["start"] + 2:
            continue
        bud_mask = (mask == bud_id)
        if not bud_mask.any():
            continue
        out[bud_mask] = color
        contours, _ = cv2.findContours(
            bud_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
        ys, xs = np.where(bud_mask)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        label = f"B{bud_id}"
        if info["parent"] != 0:
            label += f"->M{info['parent']}"
        cv2.putText(
            out,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def make_video(seq, data_root, gt_root, pred_root, out_path, fps=5):
    data_root = Path(data_root)
    gt_root = Path(gt_root) / f"{seq}_GT" / "TRA"
    pred_root = Path(pred_root) / str(seq)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_tracks = load_tracks(gt_root / "man_track.txt")
    pred_tracks = load_tracks(pred_root / "res_track.txt")
    gt_buds = compute_bud_ids(gt_tracks)
    pred_buds = compute_bud_ids(pred_tracks)

    frames = sorted((data_root / str(seq)).glob("t*.tif"))
    if not frames:
        raise FileNotFoundError(f"No frames found for seq {seq}")

    sample_img = read_image(frames[0])
    h, w = sample_img.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h)
    )

    for frame_path in frames:
        frame_idx = int(frame_path.stem.replace("t", ""))
        img = read_image(frame_path)
        if img is None:
            continue

        gt_mask = load_mask(gt_root / f"man_track{frame_idx:03d}.tif")
        pred_mask = load_mask(pred_root / f"mask{frame_idx:03d}.tif")

        left = overlay_masks(img.copy(), gt_mask)
        left = highlight_buds(left, gt_mask, gt_buds, frame_idx, color=(0, 0, 255))
        right = overlay_masks(img.copy(), pred_mask)
        right = highlight_buds(right, pred_mask, pred_buds, frame_idx, color=(0, 255, 0))

        cv2.putText(
            left,
            "GT",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            right,
            "PRED (current)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        panel = np.concatenate([left, right], axis=1)
        writer.write(panel)

    writer.release()

    summary = {
        "seq": seq,
        "gt_buds": len(gt_buds),
        "pred_buds": len(pred_buds),
        "gt_has_parents": any(v["parent"] != 0 for v in gt_tracks.values()),
        "pred_has_parents": any(v["parent"] != 0 for v in pred_tracks.values()),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    summaries = []
    for seq in args.seqs:
        out_path = out_dir / f"compare_pred_gt_{seq}.mp4"
        summaries.append(
            make_video(
                seq,
                args.data_root,
                args.gt_root,
                args.pred_root,
                out_path,
                fps=args.fps,
            )
        )

    for s in summaries:
        print(
            f"seq {s['seq']}: gt_buds={s['gt_buds']} (parents={s['gt_has_parents']}) "
            f"pred_buds={s['pred_buds']} (parents={s['pred_has_parents']})"
        )


if __name__ == "__main__":
    main()

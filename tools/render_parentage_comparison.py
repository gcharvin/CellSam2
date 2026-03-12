#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side GT/prediction movie with colored tracklets and mother-bud links."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-tra-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=-1)
    return parser.parse_args()


def load_track_rows(path: Path) -> dict[int, tuple[int, int, int]]:
    rows: dict[int, tuple[int, int, int]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return rows
    data = np.loadtxt(path, dtype=np.int32)
    if data.size == 0:
        return rows
    if data.ndim == 1:
        data = data.reshape(1, -1)
    for track_id, start, end, parent in data.tolist():
        rows[int(track_id)] = (int(start), int(end), int(parent))
    return rows


def sorted_tiffs(path: Path, prefix: str) -> list[Path]:
    return sorted(path.glob(f"{prefix}*.tif"))


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    image = image.astype(np.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)


def id_to_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919 + 17)
    rgb = rng.integers(40, 255, size=3, dtype=np.int32)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def overlay_mask(base: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = base.copy()
    ids = [int(v) for v in np.unique(mask) if v > 0]
    for track_id in ids:
        color = np.asarray(id_to_color(track_id), dtype=np.float32)
        region = mask == track_id
        out[region] = np.clip((1.0 - alpha) * out[region] + alpha * color, 0, 255)
    return out.astype(np.uint8)


def centroids(mask: np.ndarray) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    ids = [int(v) for v in np.unique(mask) if v > 0]
    for track_id in ids:
        ys, xs = np.where(mask == track_id)
        if ys.size == 0:
            continue
        out[track_id] = (int(np.round(xs.mean())), int(np.round(ys.mean())))
    return out


def draw_panel(
    image: np.ndarray,
    mask: np.ndarray,
    tracks: dict[int, tuple[int, int, int]],
    title: str,
    frame_idx: int,
    alpha: float,
    line_thickness: int,
    font_scale: float,
) -> np.ndarray:
    panel = overlay_mask(image, mask, alpha)
    c = centroids(mask)

    for track_id, (cx, cy) in c.items():
        cv2.putText(
            panel,
            str(track_id),
            (cx - 8, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for bud_id, (_, _, parent_id) in tracks.items():
        if parent_id <= 0:
            continue
        if bud_id not in c or parent_id not in c:
            continue
        bud_xy = c[bud_id]
        parent_xy = c[parent_id]
        cv2.line(panel, parent_xy, bud_xy, (0, 0, 255), line_thickness, cv2.LINE_AA)

    cv2.putText(
        panel,
        f"{title} | frame {frame_idx:03d}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    gt_tra_dir = Path(args.gt_tra_dir).expanduser().resolve()
    pred_dir = Path(args.pred_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted_tiffs(image_dir, "t")
    gt_mask_files = sorted_tiffs(gt_tra_dir, "man_track")
    pred_mask_files = sorted_tiffs(pred_dir, "mask")

    num_frames = min(len(image_files), len(gt_mask_files), len(pred_mask_files))
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)
    if num_frames == 0:
        raise RuntimeError("No frames found to render.")

    gt_tracks = load_track_rows(gt_tra_dir / "man_track.txt")
    pred_tracks = load_track_rows(pred_dir / "res_track.txt")

    first_image = load_image(image_files[0])
    height, width = first_image.shape[:2]
    spacer = 10
    full_width = width * 2 + spacer
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (full_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    for frame_idx in range(num_frames):
        image = load_image(image_files[frame_idx])
        gt_mask = cv2.imread(str(gt_mask_files[frame_idx]), cv2.IMREAD_UNCHANGED)
        pred_mask = cv2.imread(str(pred_mask_files[frame_idx]), cv2.IMREAD_UNCHANGED)
        if gt_mask is None or pred_mask is None:
            raise FileNotFoundError(f"Missing mask at frame {frame_idx}")

        gt_panel = draw_panel(
            image=image,
            mask=gt_mask,
            tracks=gt_tracks,
            title="Ground Truth",
            frame_idx=frame_idx,
            alpha=args.alpha,
            line_thickness=args.line_thickness,
            font_scale=args.font_scale,
        )
        pred_panel = draw_panel(
            image=image,
            mask=pred_mask,
            tracks=pred_tracks,
            title="Prediction",
            frame_idx=frame_idx,
            alpha=args.alpha,
            line_thickness=args.line_thickness,
            font_scale=args.font_scale,
        )
        canvas = np.zeros((height, full_width, 3), dtype=np.uint8)
        canvas[:, :width] = gt_panel
        canvas[:, width + spacer : width + spacer + width] = pred_panel
        writer.write(canvas)

    writer.release()
    print(output)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sam2.utils.misc import read_image
from training.dataset.vos_segment_loader import CTCSegmentLoader

# Utility to audit asymmetric lineage (mother/bud) in CTC man_track files.

def _load_man_track(path: Path):
    if not path.exists():
        return None
    return np.loadtxt(path, dtype=np.int32)


def _find_bud_events(man_track):
    if man_track is None or len(man_track) == 0:
        return []
    events = []
    for row in man_track:
        obj_id, start_frame, end_frame, parent_id = row.tolist()
        if parent_id > 0:
            events.append(
                {
                    "bud_id": int(obj_id),
                    "mother_id": int(parent_id),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                }
            )
    return events


def _overlay_mask(image, mask, color, alpha=0.45):
    if mask is None:
        return image
    mask_np = (mask.numpy().astype(np.uint8) * int(255 * alpha))
    overlay = Image.new("RGBA", image.size, color + (0,))
    overlay.putalpha(Image.fromarray(mask_np))
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def _draw_label(image, mask, label, color):
    if mask is None:
        return image
    ys, xs = np.where(mask.numpy())
    if len(xs) == 0:
        return image
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    draw.text((x0, max(0, y0 - 12)), label, fill=color)
    return image


def check_asym_lineage(split_dir: Path, max_videos: int, max_events: int):
    errors = []
    checked = 0
    events_out = []

    video_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    for video_dir in video_dirs[:max_videos]:
        man_track_path = split_dir / f"{video_dir.name}_GT/TRA/man_track.txt"
        man_track = _load_man_track(man_track_path)
        events = _find_bud_events(man_track)
        if not events:
            continue

        seg_loader = CTCSegmentLoader(split_dir / f"{video_dir.name}_GT/TRA")
        for ev in events:
            mother_rows = man_track[man_track[:, 0] == ev["mother_id"]]
            if len(mother_rows) == 0:
                errors.append(f"{video_dir.name}: missing mother {ev['mother_id']} for bud {ev['bud_id']}")
                continue
            mother_start, mother_end = mother_rows[0, 1], mother_rows[0, 2]
            if ev["bud_id"] == ev["mother_id"]:
                errors.append(f"{video_dir.name}: bud {ev['bud_id']} equals mother id")
            if mother_end < ev["start_frame"]:
                errors.append(
                    f"{video_dir.name}: mother {ev['mother_id']} ends at {mother_end} before bud {ev['bud_id']} starts at {ev['start_frame']}"
                )

            segments = seg_loader.load(ev["start_frame"])
            if ev["mother_id"] not in segments:
                errors.append(f"{video_dir.name}: mother {ev['mother_id']} missing mask at frame {ev['start_frame']}")
            if ev["bud_id"] not in segments:
                errors.append( f"{video_dir.name}: bud {ev['bud_id']} missing mask at frame {ev['start_frame']}")

            checked += 1
            events_out.append({"video_dir": video_dir, **ev})
            if checked >= max_events:
                return checked, errors, events_out

    return checked, errors, events_out


def make_visualization(event, split_dir: Path, out_dir: Path):
    video_dir = event["video_dir"]
    start_frame = event["start_frame"]
    mother_id = event["mother_id"]
    bud_id = event["bud_id"]
    mask_dir = split_dir / f"{video_dir.name}_GT/TRA"
    seg_loader = CTCSegmentLoader(mask_dir)

    frame_ids = [start_frame]
    if start_frame > 0:
        frame_ids = [start_frame - 1, start_frame]

    rendered = []
    for frame_id in frame_ids:
        img_path = video_dir / f"t{frame_id:03d}.tif"
        image = read_image(img_path)
        segments = seg_loader.load(frame_id)
        mother_mask = segments.get(mother_id)
        bud_mask = segments.get(bud_id)

        image = _overlay_mask(image, mother_mask, (220, 20, 60))
        image = _overlay_mask(image, bud_mask, (0, 180, 0))
        image = _draw_label(image, mother_mask, f"mother {mother_id}", (220, 20, 60))
        image = _draw_label(image, bud_mask, f"bud {bud_id}", (0, 180, 0))

        draw = ImageDraw.Draw(image)
        draw.text((6, 6), f"frame {frame_id}", fill=(255, 255, 255))
        rendered.append(image)

    if len(rendered) == 1:
        out_path = out_dir / f"{video_dir.name}_frame{start_frame:03d}.png"
        rendered[0].save(out_path)
        return [out_path]

    w, h = rendered[0].size
    side_by_side = Image.new("RGBA", (w * 2, h))
    side_by_side.paste(rendered[0], (0, 0))
    side_by_side.paste(rendered[1], (w, 0))
    out_path = out_dir / f"{video_dir.name}_frame{start_frame-1:03d}_{start_frame:03d}.png"
    side_by_side.save(out_path)
    return [out_path]


def main():
    parser = argparse.ArgumentParser(description="Audit asymmetric division logic and visualize a bud event.")
    parser.add_argument("--data-dir",required=True,help="Root dataset dir (e.g., .../trainingdataset/moma)",)
    parser.add_argument("--split", default="train/CTC")
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--out-dir", default="debug/asym_division")
    args = parser.parse_args()

    split_dir = Path(args.data_dir) / args.split
    checked, errors, events = check_asym_lineage(split_dir, args.max_videos, args.max_events)

    print(f"Checked events: {checked}")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:10]:
            print(f"- {err}")
    else:
        print("Errors: 0")

    if not events:
        print("No bud events found for visualization.")
        return

    event_index = min(max(args.event_index, 0), len(events) - 1)
    event = events[event_index]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = make_visualization(event, split_dir, out_dir)

    print("Visualization written:")
    for path in out_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

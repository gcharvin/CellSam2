import re
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from tifffile import imread
from pathlib import Path
from collections import defaultdict
from IPython.display import Image

def check_ids_mask_man_track(train_image_dir: Path, video_id: str):
    """Analyze a video and return its missing track IDs."""

    MASK_DIR = train_image_dir / f"{video_id}_GT" / "SEG"
    MAN_TRACK_FILE_PATH = train_image_dir / f"{video_id}_GT" / "TRA" / "man_track.txt"

    if not MASK_DIR.exists() or not MAN_TRACK_FILE_PATH.exists():
        print(f"⚠️  Missing folders for video {video_id}")
        return None

    # --- Read man_track.txt ---
    track_data = {}
    with open(MAN_TRACK_FILE_PATH, 'r') as f:
        for line in f:
            if line.strip():
                track_id, start, end, parent = map(int, line.split())
                track_data[track_id] = (start, end, parent)

    # --- Read all mask images ---
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".tif")])
    masks = [imread(os.path.join(MASK_DIR, f)) for f in mask_files]

    # Collect all unique object IDs present in masks
    mask_all_ids = np.unique(np.concatenate([np.unique(m) for m in masks]))

    # Compute missing IDs (excluding 0)
    missing = set(mask_all_ids) - {0} - set(track_data.keys())
    missing = {int(x) for x in missing}  # conversion pour retirer les types numpy

    return missing


def main():
    """Main function to parse arguments and run the video ID checking."""
    parser = argparse.ArgumentParser(description='Check video IDs for missing track IDs')
    parser.add_argument('--train_image_dir', type=str, required=True,
                        help='Path to the training image directory')
    
    args = parser.parse_args()
    
    TRAIN_IMAGE_DIR = Path(args.train_image_dir)
    print(f"🔍 Checking videos in: {TRAIN_IMAGE_DIR}")
    
    # Run the analysis
    videos = sorted([
        d.name for d in TRAIN_IMAGE_DIR.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])
    
    for vid in videos:
        missing = check_ids_mask_man_track(train_image_dir=TRAIN_IMAGE_DIR, video_id=vid)
        if missing is None:
            continue
        print(f"📁 Vidéo {vid} → IDs manquants : {missing if missing else 'Aucun'}")


if __name__ == "__main__":
    main()
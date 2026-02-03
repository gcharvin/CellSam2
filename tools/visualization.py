import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from tifffile import imread

# anim = make_gif(
#     image_dir=TRAIN_IMAGE_DIR / "10",
#     mask_dir=TRAIN_IMAGE_DIR / "10_GT" / "SEG",
#     man_track_file_path=TRAIN_IMAGE_DIR / "10_GT" / "TRA",
#     save_path=SAVE_PATH,
#     fps=5,
#     show_parent_cell=True,
#     save_frame=-1
# )

def make_gif(
        image_dir: Path,
        mask_dir: Path,
        man_track_file_path: Path,
        alpha: float =0.4,
        save_path: Path = None,
        fps: int = 5,
        show_parent_cell: bool = True,
        save_frame: int = -1
):
    """
    Animate a sequence of images/frames with colored masks, cell IDs, and bounding boxes.

    Args:
        image_dir (Path): Directory containing the .tif images.
        mask_dir (Path): Directory containing the corresponding masks (.tif).
        man_track_file_path (Path): Path to man_track.txt file containing track_id, start_frame, end_frame, parent_id.
        alpha (float, optional): Transparency of masks. Default to 0.4.
        save_path (Path, optional): Path where the GIF is saved. If None, nothing is saved on disk.
        fps (int, optional): Frames per second for the animation. Default to 5.
        show_parent_cell (bool, optional): Displays a line showing the relationship between a daughter cell and a mother
                                           cell during cell division. Default to True.
        save_frame (int, optional): Save a specific frame as a png. If -1, no frame is saved. Default to -1.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object, which can be displayed or saved as GIF.

    Raises:
        AssertionError: If the number of images does not match the number of masks.
    """
    # Load content from path
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

    assert len(image_files) == len(mask_files), "Number of images does not match number of masks"
    print(f"Number of frames: {len(image_files)}")

    images = [imread(os.path.join(image_dir, f)) for f in image_files]
    masks = [imread(os.path.join(mask_dir, f)) for f in mask_files]
    images = [(img - img.min()) / (img.max() - img.min() + 1e-8) for img in images]

    track_data = {}
    with open(man_track_file_path / f"man_track.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                track_id, start, end, parent = map(int, line.split())
                track_data[track_id] = (start, end, parent)

    # Consistent color palette for all cells
    all_ids = np.unique(np.concatenate([np.unique(m) for m in masks]))
    all_ids = all_ids[all_ids != 0]
    np.random.seed(42)
    cmap = ListedColormap(np.random.rand(len(all_ids), 3))
    id_to_color = {cell_id: cmap(i)[:3] for i, cell_id in enumerate(all_ids)}

    def make_colored_mask(mask):
        """Return an RGBA mask where each cell ID has a unique color with transparency."""
        colored = np.zeros((*mask.shape, 4))
        for cell_id, (r, g, b) in id_to_color.items():
            colored[mask == cell_id] = (r, g, b, alpha)
        return colored

    # Precompute centroids for all track IDs
    centroids = {tid: {} for tid in track_data}
    for frame_idx, mask in enumerate(masks):
        print(frame_idx, mask, mask.shape, np.unique(mask))
        for track_id in np.unique(mask):
            if track_id == 0:
                continue
            yx = np.argwhere(mask == track_id)
            if yx.size > 0:
                y, x = yx.mean(axis=0)
                centroids[track_id][frame_idx] = (x, y)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    img_display = ax.imshow(images[0], cmap="gray")
    mask_display = ax.imshow(make_colored_mask(masks[0]))
    title = ax.set_title("Frame 1")
    texts, lines = [], []

    # Update function for animation
    def draw_frame(frame_idx):
        nonlocal texts, lines
        # Remove old texts and rectangles
        for t in texts: t.remove()
        for r in lines: r.remove()
        texts, lines = [], []

        # Update image and mask
        img_display.set_data(images[frame_idx])
        mask_display.set_data(make_colored_mask(masks[frame_idx]))

        # Add IDs and lineage lines
        for track_id, (start, end, parent) in track_data.items():
            if start <= frame_idx <= end:
                if frame_idx in centroids[track_id]:
                    x, y = centroids[track_id][frame_idx]
                    texts.append(ax.text(x, y, str(track_id), color='white', fontsize=10, ha='center', va='center'))

            # Draw lineage line when a new cell appears
            if show_parent_cell:
                if frame_idx == start and parent in centroids and (start - 1) in centroids[parent]:
                    x1, y1 = centroids[parent][start - 1]
                    x2, y2 = centroids[track_id][start]
                    line = ax.plot([x1, x2], [y1, y2], color='red', linewidth=1.4, alpha=0.8)[0]
                    lines.append(line)

        title.set_text(f"Frame {frame_idx + 1}/{len(images)}")
        return [img_display, mask_display] + texts

    if save_frame != -1:
        if not (0 <= save_frame < len(images)):
            raise ValueError(f"save_frame must be between 0 and {len(images)-1}.")
        draw_frame(save_frame-1)
        plt.savefig(f"{save_frame}_{save_path.stem}.png", bbox_inches="tight", dpi=200)
        print(f"✅ Saved frame {save_frame} to {save_frame}_{save_path.stem}.png")

    anim = FuncAnimation(fig, draw_frame, frames=len(images), interval=1000 / fps, blit=False)
    plt.show()

    if save_path:
        anim.save(save_path, fps=fps)
        print(f"✅ Saved gif to {save_path}")

    return anim
import subprocess
import torch
import cv2
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np



def get_device():
    """Get the device to use for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_video_path() -> Path:
    """Open a GUI dialog for selecting video file or image sequence directory.

    Returns:
        Path: Selected path to video file or image directory

    """
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()

    # Ask user if they want to select a video or a folder of images
    selection_window = tk.Toplevel(root)
    selection_window.title("Select Input Type")
    selection_window.geometry("300x150")
    selection_window.resizable(False, False)

    selected_option = tk.StringVar(value="video")

    tk.Label(selection_window, text="Choose input type:").pack(pady=10)
    tk.Radiobutton(selection_window, text="Video file", variable=selected_option, value="video").pack(anchor=tk.W, padx=20)
    tk.Radiobutton(selection_window,text="Folder of images",variable=selected_option,value="images",).pack(anchor=tk.W, padx=20)

    path_result = [None]  # Use list to store result from callback

    def on_confirm():
        option = selected_option.get()
        if option == "video":
            path_result[0] = filedialog.askopenfilename(
                title="Select a video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            )
        else:  # images
            path_result[0] = filedialog.askdirectory(title="Select folder containing image sequence")
        selection_window.destroy()

    tk.Button(selection_window, text="Confirm", command=on_confirm).pack(pady=20)

    # Wait for the window to be closed
    selection_window.wait_window()

    if not path_result[0]:
        print("No input selected. Exiting...")
        exit()

    return Path(path_result[0])


def get_result_path(base_dir: Path,model_name: str,input_path: Path,dir_name: str,res_path: Path = None,) -> Path:
    """Generate the result path based on input path structure.

    Args:
        base_dir: Base directory (usually __file__.parents[1])
        model_name: Name of the model being used
        input_path: Input path being processed
        dir_name: Name of directory being processed
        res_path: Result path to save to

    Returns:
        Path: Result directory path

    """
    if res_path is not None:
        return Path(res_path) / dir_name

    # Start with common base path
    result_path = base_dir / "sam2_logs" / model_name / "results"

    # Add split directory if in train/val/test
    if "test" in input_path.parts:
        result_path = result_path / "test"
    elif "train" in input_path.parts:
        result_path = result_path / "train"
    elif "val" in input_path.parts:
        result_path = result_path / "val"

    # Add CTC directory if in CTC dataset
    if "CTC" in input_path.parts:
        result_path = result_path / "CTC"

    # Add final directory name
    return result_path / dir_name


def has_tif_files(path):
    """Check if the directory contains .tif files."""
    path = Path(path)
    return any(f.suffix.lower() == ".tif" for f in path.glob("*.[tT][iI][fF]"))


def get_tif_directories(base_path):
    """Get all directories containing .tif files."""
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise ValueError(f"{base_path} is not a directory")

    # If the base directory has .tif files, return just that
    if has_tif_files(base_path):
        return [base_path]

    # Otherwise, look for subdirectories with .tif files
    tif_dirs = []
    for subdir in base_path.iterdir():
        if subdir.is_dir() and has_tif_files(subdir):
            tif_dirs.append(subdir)

    if not tif_dirs:
        raise ValueError(f"No directories containing .tif files found in {base_path}")

    return sorted(tif_dirs)


def display_masks(image, masks):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis("off")
    plt.show()


def show_anns(anns, borders=True, mask_alpha=0.1):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4,) )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [mask_alpha]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

            # Add IoU prediction text
            # Get centroid of the largest contour to place text
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Add text with IoU value
                plt.text(
                    cx,
                    cy,
                    f"IoU: {ann['predicted_iou']:.2f}\nObj Score: {ann['obj_score']:.2f}\nStability: {ann['stability_score']:.2f}",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5),
                )

    ax.imshow(img)



def create_colored_frame(img, mask, colors, alpha=0.3,id_scale=0.3):
    """Create a colored frame with overlay from an image and a mask."""
    overlay = np.zeros_like(img)
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids != 0]  # Exclude background (0)

    centroids = {}
    for cell_id in cell_ids:
        # Vérifiez que cell_id est dans les limites du tableau colors
        if cell_id >= len(colors):
            # Si cell_id dépasse la taille de colors, redimensionnez colors
            new_colors = np.random.randint(0, 255, (cell_id + 1, 3))
            new_colors[:len(colors)] = colors
            colors = new_colors

        mask_binary = mask == cell_id
        overlay[mask_binary] = colors[cell_id]

        y_coords, x_coords = np.where(mask_binary)
        if len(y_coords) == 0:
            continue

        centroid_y = int(np.mean(y_coords))
        centroid_x = int(np.mean(x_coords))
        centroids[int(cell_id)] = (centroid_x, centroid_y)

    # Appliquer l'overlay à l'image
    colored_frame = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    # Dessiner les IDs directement sur l'image finale
    for cell_id in centroids:
        centroid_x, centroid_y = centroids[cell_id]
        cv2.putText(
            colored_frame,
            str(cell_id),
            (centroid_x - 5, centroid_y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            id_scale,  # Font scale
            (0, 0, 255),  # Rouge en BGR
            1,  # Line thickness
            cv2.LINE_AA,
        )

    return colored_frame, centroids


def draw_division_lines(frame, centroids, parent_map):
    """Draw division lines between parent and child cells."""
    for child_id, parent_id in parent_map.items():
        child_centroid = centroids.get(child_id)
        parent_centroid = centroids.get(parent_id)
        if child_centroid is None or parent_centroid is None:
            continue
        cv2.line(
            frame,
            child_centroid,
            parent_centroid,
            (0, 0, 0),  # Black color
            1,
        )

def add_frame_number(frame, frame_idx):
    """Add frame number to the top of the frame."""
    cv2.putText(
        frame,
        f"{frame_idx:03}",
        (0, 15),  # Position in top-left
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,  # Font scale
        (255, 255, 255),  # White color
        1,  # Line thickness
        cv2.LINE_AA,
    )

def save_video(frames, output_path, fps=4.0):
    """Save frames as a video."""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()


def combined_pred_gt_videos(summary_pred_path : Path):

    pred_mp4_path = summary_pred_path / "pred_track_video.mp4"
    gt_mp4_path = summary_pred_path / "gt_track_video.mp4"
    output_mp4 = summary_pred_path / "combined_pred_gt_video.mp4"
    # Vérification globale
    if not pred_mp4_path.exists() or not gt_mp4_path.exists():
        print("Erreur : Un ou plusieurs fichiers nécessaires sont manquants.")

    # Commande ffmpeg pour concaténer les vidéos horizontalement
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file without asking
        '-i', str(pred_mp4_path),
        '-i', str(gt_mp4_path),
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',  # Concaténer horizontalement
        '-map', '[v]',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-loglevel', 'quiet',
        str(output_mp4)
    ]


    # Exécuter la commande
    subprocess.run(cmd, check=True)



def aggregate_video_metrics(all_metrics_by_video):
    """Aggregate metrics across all videos."""
    total = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "parentless_pred": 0,
        "time_errors": [],
        "iou_mother_scores": [],
        "iou_bud_scores": [],
        "eval_details": [],  # Renommée
    }

    for metrics in all_metrics_by_video:
        total["tp"] += metrics["metrics"]["tp"]
        total["fp"] += metrics["metrics"]["fp"]
        total["fn"] += metrics["metrics"]["fn"]
        total["parentless_pred"] += metrics["metrics"]["parentless_pred"]

        # Ajouter les erreurs temporelles et les scores IoU
        if "avg_time_error" in metrics["metrics"]:
            total["time_errors"].append(metrics["metrics"]["avg_time_error"])
        if "avg_iou_mother" in metrics["metrics"]:
            total["iou_mother_scores"].append(metrics["metrics"]["avg_iou_mother"])
        if "avg_iou_bud" in metrics["metrics"]:
            total["iou_bud_scores"].append(metrics["metrics"]["avg_iou_bud"])

        # Ajout des triplets (gt_video_dir, pred_video_dir, params)
        if "gt_video_dir" in metrics and "pred_video_dir" in metrics and "params" in metrics:
            total["eval_details"].append({  # Utilisation de la nouvelle clé
                "gt_video_dir": str(metrics["gt_video_dir"]),
                "pred_video_dir": str(metrics["pred_video_dir"]),
                "params": metrics["params"],
            })

    # Calcul des métriques globales
    precision = total["tp"] / (total["tp"] + total["fp"] + 1e-12)
    recall = total["tp"] / (total["tp"] + total["fn"] + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12) if (precision + recall) > 0 else 0.0

    avg_time_error = np.mean(total["time_errors"]) if total["time_errors"] else 0.0
    std_time_error = np.std(total["time_errors"]) if total["time_errors"] else 0.0

    avg_iou_mother = np.mean(total["iou_mother_scores"]) if total["iou_mother_scores"] else 0.0
    std_iou_mother = np.std(total["iou_mother_scores"]) if total["iou_mother_scores"] else 0.0

    avg_iou_bud = np.mean(total["iou_bud_scores"]) if total["iou_bud_scores"] else 0.0
    std_iou_bud = np.std(total["iou_bud_scores"]) if total["iou_bud_scores"] else 0.0

    total_metrics = {
        "tp": total["tp"],
        "fp": total["fp"],
        "fn": total["fn"],
        "parentless_pred": total["parentless_pred"],
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "avg_time_error": round(avg_time_error, 3),
        "std_time_error": round(std_time_error, 3),
        "avg_iou_mother": round(avg_iou_mother, 3),
        "std_iou_mother": round(std_iou_mother, 3),
        "avg_iou_bud": round(avg_iou_bud, 3),
        "std_iou_bud": round(std_iou_bud, 3),
        "eval_details": total["eval_details"],  # Nouvelle clé
    }

    return total_metrics


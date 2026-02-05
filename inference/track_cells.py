# Standard library imports
import argparse
import sys
from pathlib import Path

# Third-party imports
import hydra
from cell_tracker import SAM2AutomaticCellTracker
from hydra.core.global_hydra import GlobalHydra
from inference_utils import get_device, get_result_path, get_tif_directories, get_video_path, combined_pred_gt_videos

# Local imports
from sam2.build_sam import build_sam2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cell tracking with SAM2")
    # Defaults are relaxed to reduce FN divisions in asym/budding inference.
    parser.add_argument("--video_path",type=str,default=None, help="Path to video file or image sequence directory",)
    parser.add_argument("--res_path", type=str, default=None, help="Path to save tracking results")
    parser.add_argument("--data_version", type=str, default=None, help="tag for datadir")
    parser.add_argument("--model_name",type=str,default="SAM2-tracking-LoRA-heatmap",help="Name of the model to use",)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7,help="Non-maximum suppression threshold for bounding boxes",)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.3,help="IoU threshold for predictions",)
    parser.add_argument("--obj_score_thresh",type=float,default=0.0,help="Object score threshold (logit)",)
    parser.add_argument("--div_obj_score_thresh",type=float,default=-4.0,help="Division score threshold (logit)",)
    parser.add_argument("--min_mask_area",type=int,default=10,help="Minimum area for keeping masks",)
    parser.add_argument("--segment",action="store_true",help="Whether to perform segmentation (default: False)",)
    parser.add_argument("--use_heatmap", type=bool, default=True, help="Whether to use heatmap")
    parser.add_argument("--checkpoint_num", type=int, default=None, help="Checkpoint number to use")
    parser.add_argument("--fps", type=int, default=4, help="fps for video gt and pred")

    return parser.parse_args()


def setup_hydra(model_name: str):
    """Setup Hydra configuration.

    Args:
        model_name: Name of the model to use

    """
    config_path = f"../sam2_logs/{model_name}"

    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    # Global initialization
    hydra.initialize(version_base=None, config_path=config_path)
    return f"../sam2_logs/{model_name}", "config_resolved"


def process_directory(
    tracker: SAM2AutomaticCellTracker,
    dir_path: Path,
    base_dir: Path,
    model_name: str,
    input_path: Path,
    res_path: Path,
    total_dirs: int,
    idx: int,
    fps: int,
):
    """Process a single directory with the cell tracker.

    Args:
        tracker: The cell tracker instance
        dir_path: Directory to process
        base_dir: Base directory for results
        model_name: Name of the model
        input_path: Input path being processed
        res_path: Path to save results
        idx: Index of current directory
        total_dirs: Total number of directories

    """
    print(f"Processing directory {idx + 1}/{total_dirs}: {dir_path}")
    # Generate result path
    result_path = get_result_path(
        base_dir=base_dir,
        model_name=model_name,
        input_path=input_path,
        dir_name=dir_path.stem,
        res_path=res_path,
    )
    result_summary = result_path / "summary"
    result_summary.mkdir(parents=True, exist_ok=True)

    # Track cells in the video or image sequence
    tracker.predict(
        dir_path,
        result_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        max_frame_num_to_track=None,
        fps=fps,
    )
    combined_pred_gt_videos(result_summary)

    print(f"Finished processing: {dir_path}")


def main():
    """Main function for cell tracking."""
    # Parse arguments and setup paths
    args = parse_args()
    model_name = args.model_name
    config_path, config_name = setup_hydra(model_name)

    # Setup model and device
    device = get_device()
    if args.checkpoint_num is None:
        sam2_checkpoint = f"sam2_logs/{model_name}/checkpoints/checkpoint.pt"
    else:
        sam2_checkpoint = f"sam2_logs/{model_name}/checkpoints/checkpoint_{args.checkpoint_num}.pt"
    sam2_model = build_sam2(config_name, sam2_checkpoint, device=device)

    # Create the cell tracker
    tracker = SAM2AutomaticCellTracker(
        sam2_model,
        pred_iou_thresh=args.pred_iou_thresh,
        obj_score_thresh=args.obj_score_thresh,
        div_obj_score_thresh=args.div_obj_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        min_mask_area=args.min_mask_area,
        segment=args.segment,
        use_heatmap=args.use_heatmap,
    )

    # Get input path
    video_path = Path(args.video_path) if args.video_path else get_video_path()

    try:
        # Get directories containing .tif files
        directories = get_tif_directories(video_path)

        # Process each directory
        base_dir = Path(__file__).parents[1]
        for idx, dir_path in enumerate(directories):
            res_pred_path = Path(args.res_path) / f"model:{model_name}" / f"data_vers:{args.data_version}"

            process_directory(
                tracker=tracker,
                dir_path=dir_path,
                base_dir=base_dir,
                model_name=model_name,
                input_path=video_path,
                res_path=res_pred_path,
                total_dirs=len(directories),
                idx=idx,
                fps=args.fps
            )


    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    main()

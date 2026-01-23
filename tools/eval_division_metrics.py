#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from training.utils.checkpoint_utils import load_state_dict_into_model
from training.utils.train_utils import register_omegaconf_resolvers

# Offline evaluator for division head metrics on a CTC split.

def average_precision(y_true, y_score):
    if y_true.size == 0:
        return float("nan")
    pos_count = y_true.sum()
    if pos_count == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_true = y_true[order]
    cum_tp = np.cumsum(y_true)
    cum_fp = np.cumsum(1 - y_true)
    precision = cum_tp / (cum_tp + cum_fp + 1e-12)
    ap = precision[y_true == 1].mean()
    return float(ap)


def summarize_scores(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate division prediction metrics on a CTC split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",default="sam2/configs/sam2.1_training/sam2.1_ctc_finetune.yaml",)
    parser.add_argument("--split", default="val/CTC")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-json", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    register_omegaconf_resolvers()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.data_dir = args.data_dir
    cfg.scratch.batch_size = args.batch_size
    cfg.scratch.num_frames = args.num_frames
    cfg.trainer.data.val.dataset.video_dataset.train_dir = f"{args.data_dir}/{args.split}"

    model = instantiate(cfg.trainer.model, _convert_="all")
    if cfg.trainer.use_lora:
        lora = instantiate(cfg.trainer.lora, model=model, _convert_="all")
        model = lora.model

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    load_state_dict_into_model(ckpt["model"], model, strict=True, ignore_missing_keys=None, ignore_unexpected_keys=None,)

    model.to(device)
    model.eval()

    val_loader = instantiate(cfg.trainer.data.val, _convert_="all")

    all_scores = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch)
            frame_idxs = [t for t in range(batch.num_frames) if not batch.no_inputs[t]]
            for out, t in zip(outputs, frame_idxs):
                obj_ids = batch.metadata.unique_objects_identifier[t][batch.is_real[t]][:, 1]
                is_cell = obj_ids > 0
                if not is_cell.any():
                    continue
                targets = batch.cell_divides[t][batch.is_real[t]][is_cell]
                div_logits = out["multistep_div_score_logits"][-1].squeeze(1)
                scores = div_logits[is_cell].sigmoid()
                all_scores.append(scores.detach().cpu())
                all_targets.append(targets.detach().cpu())

            num_batches += 1
            if args.max_batches and num_batches >= args.max_batches:
                break

    if not all_scores:
        print("No division targets found for evaluation.")
        return

    y_score = torch.cat(all_scores, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy().astype(np.int32)

    summary = summarize_scores(y_true, y_score, args.threshold)
    ap = average_precision(y_true, y_score)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    stats = {
        "num_samples": int(y_true.size),
        "num_pos": int(y_true.sum()),
        "num_neg": int((y_true == 0).sum()),
        "threshold": args.threshold,
        "average_precision": ap,
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "tp": summary["tp"],
        "fp": summary["fp"],
        "fn": summary["fn"],
        "tn": summary["tn"],
        "pos_score_mean": float(pos_scores.mean()) if pos_scores.size else float("nan"),
        "neg_score_mean": float(neg_scores.mean()) if neg_scores.size else float("nan"),
    }

    print(json.dumps(stats, indent=2))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

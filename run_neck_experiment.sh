#!/usr/bin/env bash
# Run pairwise-temporal-sam2-neck-blend060-v1 experiment
# Replicates the reference blend060 experiment with 3 additional SAM2 neck features.
#
# Usage: bash run_neck_experiment.sh
# Master log: /tmp/neck_exp_master.log
#
set -euo pipefail

PYTHON=/home/charvin-admin/.conda/envs/cellsam2/bin/python
REPO=/home/charvin-admin/Documents/cellSAM2/CellSam2
EXPERIMENTS=/home/charvin-admin/Documents/cellSAM2/experiments
DATASET=/homes/Gilles/Data/DetecDivProjects/anais/bud4/classification/celltracktr_5/trainingdataset/moma
CELLHOTA=/home/charvin-admin/Documents/github/Cell-HOTA

RAW_TRAIN=$EXPERIMENTS/20260312_1130_854f313_hybrid-cost-v1-postproc-dev-holdout/inference/raw_train_dev
RAW_VAL=$EXPERIMENTS/20260312_1130_854f313_hybrid-cost-v1-postproc-dev-holdout/inference/raw_val_holdout

GT_TRAIN=$DATASET/train/CTC
GT_VAL=$DATASET/val/CTC
IMAGE_TRAIN=$DATASET/train/CTC
IMAGE_VAL=$DATASET/val/CTC

SAM2_MODEL=CellSam2-asym-no-gate-mothercentric-12ep-div3-20260205
ALPHA=0.6
TAG=pairwise-temporal-sam2-neck-blend060-v1

cd "$REPO"

# --- Init experiment ---
echo "[init] Initializing experiment $TAG..."
$PYTHON tools/init_experiment.py \
  --tag "$TAG" \
  --dataset moma \
  --dataset-path "$DATASET" \
  --cellhota-repo "$CELLHOTA" \
  --status running \
  --notes "Same as pairwise-temporal-sam2-blend060-v1 but with 3 additional SAM2 neck-region features: sam2_neck_cosine_mean, sam2_neck_cosine_min, sam2_neck_valid_fraction. Pool interface region instead of whole object. FEATURE_NAMES 23->26."

EXP_ID=$($PYTHON - <<'PY'
import sys
sys.path.insert(0, "/home/charvin-admin/Documents/cellSAM2/CellSam2")
from tools.experiment_utils import build_exp_id
from pathlib import Path
repo = Path("/home/charvin-admin/Documents/cellSAM2/CellSam2")
print(build_exp_id(repo, "pairwise-temporal-sam2-neck-blend060-v1"))
PY
)
EXP_DIR=$EXPERIMENTS/$EXP_ID
echo "[init] Experiment dir: $EXP_DIR"

TRAIN_OUT=$EXP_DIR/inference/train_dev_learned
VAL_OUT=$EXP_DIR/inference/val_holdout_learned
mkdir -p "$TRAIN_OUT" "$VAL_OUT" "$EXP_DIR/eval"

COMMON_ARGS=(
  --objective pairwise
  --sam2-model-name "$SAM2_MODEL"
  --learned-score-alpha "$ALPHA"
  --interface-radius 4
  --first-frames 4
  --w-dist 0.6 --w-size 0.3 --w-motion 0.1 --w-contact 0.0
  --w-neck 0.25 --w-angle 0.20 --w-track-quality 0.10 --w-margin 0.15
  --refractory 8
  --min-bud-area 10 --min-mother-age 3 --min-track-length 2
  --score-threshold 0.5
)

# --- Train on train_dev, apply to train_dev ---
echo "[1/3] Training + apply on train_dev..."
$PYTHON tools/learned_bud_rerank.py \
  --train-gt-root "$GT_TRAIN" \
  --train-pred-root "$RAW_TRAIN" \
  --train-image-root "$IMAGE_TRAIN" \
  --apply-gt-root "$GT_TRAIN" \
  --apply-pred-root "$RAW_TRAIN" \
  --apply-image-root "$IMAGE_TRAIN" \
  --train-videos 01,02,03,04,05,06,07,08,09,10,11 \
  --apply-videos 01,02,03,04,05,06,07,08,09,10,11 \
  --output-root "$TRAIN_OUT" \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "$EXP_DIR/inference/train_dev_learned.stdout.log"

# --- Train on train_dev, apply to val_holdout ---
echo "[2/3] Training + apply on val_holdout..."
$PYTHON tools/learned_bud_rerank.py \
  --train-gt-root "$GT_TRAIN" \
  --train-pred-root "$RAW_TRAIN" \
  --train-image-root "$IMAGE_TRAIN" \
  --apply-gt-root "$GT_VAL" \
  --apply-pred-root "$RAW_VAL" \
  --apply-image-root "$IMAGE_VAL" \
  --train-videos 01,02,03,04,05,06,07,08,09,10,11 \
  --apply-videos 12,13,14 \
  --output-root "$VAL_OUT" \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "$EXP_DIR/inference/val_holdout_learned.stdout.log"

# --- Evaluate train_dev + val_holdout ---
echo "[3/3] Evaluating..."
$PYTHON tools/eval_dev_holdout.py \
  --experiment "$EXP_DIR" \
  --dataset-root "$DATASET" \
  --train-pred-root "$TRAIN_OUT/predictions" \
  --val-pred-root "$VAL_OUT/predictions" \
  --train-videos 01,02,03,04,05,06,07,08,09,10,11 \
  --val-videos 12,13,14 \
  2>&1 | tee "$EXP_DIR/eval/eval.stdout.log"

echo "[done] Experiment complete: $EXP_DIR"

#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="/opt/miniconda3/bin/conda"
DATA_DIR="/homes/Gilles/Data/DetecDivProjects/anais/bud4/classification/celltracktr_5/trainingdataset/moma/"
SUMMARY="sam2_logs/div_target_sweep_summary.tsv"

mkdir -p sam2_logs

echo -e "run_id\ttrain_div_last\tval_div_last" > "$SUMMARY"

run_training() {
  local run_id="$1"
  local prev_window="$2"
  local include_current="$3"
  local include_bud="$4"
  local soft="$5"
  local soft_scale="$6"
  local soft_cap="$7"

  local log_dir="sam2_logs/${run_id}"
  mkdir -p "$log_dir"

  $CONDA_BIN run -n cellsam2 python train_ctc.py \
    launcher.experiment_log_dir="$log_dir" \
    scratch.dataset_name=moma \
    dataset.data_dir="$DATA_DIR" \
    scratch.batch_size=1 \
    scratch.num_epochs=10 \
    trainer.data.train.collate_fn.division_target_prev_window="$prev_window" \
    trainer.data.train.collate_fn.division_target_include_current="$include_current" \
    trainer.data.train.collate_fn.division_target_include_bud="$include_bud" \
    trainer.data.train.collate_fn.division_target_soft="$soft" \
    trainer.data.train.collate_fn.division_target_soft_scale="$soft_scale" \
    trainer.data.train.collate_fn.division_target_soft_cap="$soft_cap" \
    trainer.data.val.collate_fn.division_target_prev_window="$prev_window" \
    trainer.data.val.collate_fn.division_target_include_current="$include_current" \
    trainer.data.val.collate_fn.division_target_include_bud="$include_bud" \
    trainer.data.val.collate_fn.division_target_soft="$soft" \
    trainer.data.val.collate_fn.division_target_soft_scale="$soft_scale" \
    trainer.data.val.collate_fn.division_target_soft_cap="$soft_cap" \
    > "$log_dir/train.log" 2>&1

  python - <<PY
import re
from pathlib import Path

log_path = Path("$log_dir") / "logs" / "log.txt"
train_div = None
val_div = None

if log_path.exists():
    with log_path.open("r") as f:
        for line in f:
            if "Losses/train_all_loss_div" in line and "Losses/train_all_loss_mask" in line:
                train_div = line.strip()
            if "Losses/val_all_loss_div" in line and "Losses/val_all_loss_mask" in line:
                val_div = line.strip()

    def extract_div(line, key):
        if not line:
            return None
        start = line.find("{")
        end = line.rfind("}")
        if start == -1 or end == -1:
            return None
        dict_str = line[start+1:end]
        for item in dict_str.split(","):
            if key in item:
                return item.split(":", 1)[1].strip()
        return None

    train_div_val = extract_div(train_div, "Losses/train_all_loss_div")
    val_div_val = extract_div(val_div, "Losses/val_all_loss_div")
else:
    train_div_val = None
    val_div_val = None

print(f"{train_div_val}\t{val_div_val}")
PY
}

RUNS=(
  "divtarget_prev_only 1 false false false 1.0 1.0"
  "divtarget_prev_current 1 true false false 1.0 1.0"
  "divtarget_prev_current_bud 1 true true false 1.0 1.0"
  "divtarget_prev_current_bud_soft 1 true true true 1.0 1.0"
)

for run in "${RUNS[@]}"; do
  set -- $run
  run_id="$1"
  prev_window="$2"
  include_current="$3"
  include_bud="$4"
  soft="$5"
  soft_scale="$6"
  soft_cap="$7"

  divs=$(run_training "$run_id" "$prev_window" "$include_current" "$include_bud" "$soft" "$soft_scale" "$soft_cap")
  echo -e "${run_id}\t${divs}" >> "$SUMMARY"
  echo "Finished ${run_id}"
  echo "---"
  echo "${divs}"
  echo "---"
  sleep 5
 done

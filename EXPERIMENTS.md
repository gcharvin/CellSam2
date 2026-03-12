# Experiments

This repo uses a structured experiment layout so training, inference, and evaluation remain reproducible and reviewable.

## Scope

The rule is simple:

- heavy artifacts live outside git;
- metadata and decisions are stored in a stable structure;
- every run is tied to exact `CellSam2` and `Cell-HOTA` commits.

## Default Server Paths

- `CellSam2` repo:
  - `/home/charvin-admin/Documents/cellSAM2/CellSam2`
- `Cell-HOTA` repo:
  - `/home/charvin-admin/Documents/github/Cell-HOTA`
- experiments root:
  - `/home/charvin-admin/Documents/cellSAM2/experiments`
- current dataset root:
  - `/homes/Gilles/Data/DetecDivProjects/anais/bud4/classification/celltracktr_5/trainingdataset/moma`

## Experiment ID

Each experiment uses:

```text
YYYYMMDD_HHMM_<cellsam2_commit_short>_<tag>
```

Example:

```text
20260311_1540_b139340_online-bud-v1
```

## Required Layout

```text
<experiments_root>/<exp_id>/
  manifest.yaml
  env/
    cellsam2_env.txt
    git_cellsam2.txt
    cellsam2.diff
    git_cellhota.txt
    cellhota.diff
  train/
    config_resolved.yaml
    stdout.log
    checkpoints/
  inference/
    inference_args.yaml
    stdout.log
    val/
      12/
      13/
      14/
  eval/
    summary.json
    division_iou_window.json
    cell_hota/
  review/
    summary.md
    plots/
```

`manifest.yaml` is stored as JSON-compatible YAML so it can be read without external dependencies.

## Minimal Workflow

Initialize a run:

```bash
python tools/init_experiment.py \
  --tag online-bud-v1 \
  --dataset moma \
  --cellhota-repo /home/charvin-admin/Documents/github/Cell-HOTA
```

Update the summary after the run:

```bash
python tools/update_experiment_summary.py \
  --experiment /home/charvin-admin/Documents/cellSAM2/experiments/20260311_1540_b139340_online-bud-v1 \
  --status reviewed \
  --checkpoint train/checkpoints/checkpoint_12.pt \
  --metric div_iou_window_f1=0.649 \
  --metric div_iou_window_precision=0.526 \
  --metric div_iou_window_recall=0.847 \
  --note "Online bud parentage improves recall but hurts seq14 precision." \
  --decision "Keep as baseline and add a contact/interface score."
```

## Review Rules

- Never commit checkpoints or prediction trees.
- Always capture repo diffs if the tree is dirty.
- Every run must have:
  - machine-readable summary in `eval/summary.json`
  - short interpretation in `review/summary.md`

## Current Recommendation

For current yeast tracking work:

- initialize every new run with `tools/init_experiment.py`
- keep `CellSam2` and `Cell-HOTA` commits in the manifest
- compare runs primarily with:
  - `tools/eval_division_iou_window.py`
  - `Cell-HOTA`

## Train Dev / Val Holdout

For non-learned post-processing and heuristics, use:

- `train/CTC` as `train_dev`
  - purpose:
    - diagnose failure modes
    - tune heuristic parameters
    - reject clearly unstable ideas early
- `val/CTC` as `val_holdout`
  - purpose:
    - decide whether a heuristic is kept
    - estimate whether a gain survives outside the development split

This is intentionally not framed as standard ML model selection. The point is to avoid overfitting heuristic rules to the current holdout videos `12,13,14`.

Recommended workflow:

```bash
python tools/eval_dev_holdout.py \
  --experiment /home/charvin-admin/Documents/cellSAM2/experiments/<exp_id> \
  --dataset-root /homes/Gilles/Data/DetecDivProjects/anais/bud4/classification/celltracktr_5/trainingdataset/moma \
  --train-pred-root /path/to/train/predictions \
  --val-pred-root /path/to/val/predictions
```

Outputs:

- `eval/train_dev/division_iou_window.json`
- `eval/train_dev/bud_event_diagnostic.json`
- `eval/val_holdout/division_iou_window.json`
- `eval/val_holdout/bud_event_diagnostic.json`
- `eval/dev_holdout_summary.json`
- `review/dev_holdout_summary.md`

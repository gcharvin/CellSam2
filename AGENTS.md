# AGENTS.md

## Experiment Rules

These rules apply to any agent running training, inference, or evaluation in this repo.

### Goals

- Keep heavy artifacts out of git.
- Keep metadata, decisions, and summaries reviewable.
- Keep every run traceable to exact code revisions for `CellSam2` and `Cell-HOTA`.

### Canonical Layout

- Store experiment artifacts under an experiments root outside the git repo when possible.
- Default server path:
  - `/home/charvin-admin/Documents/cellSAM2/experiments`
- Each run must live under:
  - `<experiments_root>/<exp_id>/`

### Experiment ID

- Use:
  - `YYYYMMDD_HHMM_<cellsam2_commit_short>_<tag>`
- `tag` must be short, lowercase, and hyphenated.

### Mandatory Files Per Experiment

- `manifest.yaml`
- `env/cellsam2_env.txt`
- `env/git_cellsam2.txt`
- `env/cellsam2.diff`
- `review/summary.md`
- `eval/summary.json`

If `Cell-HOTA` is involved, also write:

- `env/git_cellhota.txt`
- `env/cellhota.diff`

### Required Directory Skeleton

```text
<exp_id>/
  manifest.yaml
  env/
  train/
    checkpoints/
  inference/
    val/
  eval/
    cell_hota/
  review/
    plots/
```

### Required Process

1. Before launching a run, initialize the experiment directory with:
   - `python tools/init_experiment.py ...`
2. Run training/inference/evaluation while writing outputs into that experiment directory.
3. After results are available, update the summary with:
   - `python tools/update_experiment_summary.py ...`
4. If the repo is dirty, do not hide it:
   - preserve the diff in `env/*.diff`
   - record dirty state in `manifest.yaml`

### Reviewability Rules

- Do not commit checkpoints, masks, videos, or large result trees.
- Do commit:
  - scripts
  - configs
  - templates
  - documentation
  - experiment summaries or indexes if they remain small
- Every reviewed run must have a short Markdown interpretation in `review/summary.md`.

### Cell-HOTA Rules

- Prefer the local clone on the server:
  - `/home/charvin-admin/Documents/github/Cell-HOTA`
- Always record the exact `Cell-HOTA` commit used for evaluation.

### Server Conventions

- Preferred repo:
  - `/home/charvin-admin/Documents/cellSAM2/CellSam2`
- Preferred conda env:
  - `cellsam2`
- Preferred dataset root for current yeast work:
  - `/homes/Gilles/Data/DetecDivProjects/anais/bud4/classification/celltracktr_5/trainingdataset/moma`

### Split Evaluation Rules

- For non-learned post-processing, use:
  - `train/CTC` as `train_dev`
  - `val/CTC` as `val_holdout`
- `train_dev` may be used to tune heuristic rules and thresholds.
- `val_holdout` must remain the decision split.
- Do not claim an improvement from `train_dev` alone.
- Prefer the helper:
  - `python tools/eval_dev_holdout.py ...`

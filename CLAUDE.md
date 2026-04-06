# Claude Handoff: CellSam2 Yeast Bud Parentage

This repository is a refactor/extension of CellSam2 for budding yeast parentage.
The current work branch is:

```bash
codex/yeast-bud-parentage
```

Important: before editing, always run:

```bash
git status --short
git branch --show-current
```

At the time this note was written, the local working tree had user/Claude-side
uncommitted changes in `tools/learned_bud_rerank.py` adding SAM2 neck features,
plus local `.claude/` and `CON` untracked files. Do not overwrite these without
review.

## Remote Server

SSH alias:

```bash
ssh detecdiv-server
```

Server repo:

```bash
/home/charvin-admin/Documents/cellSAM2/CellSam2
```

Server branch:

```bash
codex/yeast-bud-parentage
```

Server conda environment:

```bash
conda activate cellsam2
```

Dataset:

```bash
/homes/Gilles/Data/DetecDivProjects/anais/bud4/classification/celltracktr_5/trainingdataset/moma
```

Split:

```bash
train/CTC  # videos 01-11, used as train_dev for post-processing development
val/CTC    # videos 12-14, used as val_holdout
```

Experiment root:

```bash
/home/charvin-admin/Documents/cellSAM2/experiments
```

Typical sync from local to server:

```bash
# local
git push origin codex/yeast-bud-parentage

# server
ssh detecdiv-server
cd /home/charvin-admin/Documents/cellSAM2/CellSam2
git checkout codex/yeast-bud-parentage
git pull --ff-only
conda activate cellsam2
```

## Current State

The best current model is:

```text
pairwise-temporal-sam2-blend060-v1
```

Reference experiment:

```bash
/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1455_ce88ca0_pairwise-temporal-sam2-blend060-v1
```

Scores on `val_holdout`:

```text
division_iou_window F1 = 0.662420
stable_correct_parent = 41
stable_wrong_parent = 9
stable_orphan = 2
Cell-HOTA HOTA = 42.493
Cell-HOTA DivA = 10.276
Cell-HOTA DivRe = 23.283
Cell-HOTA DivPr = 14.46
```

Baseline heuristic reference:

```bash
/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1130_854f313_hybrid-cost-v1-postproc-dev-holdout
```

Baseline scores:

```text
division_iou_window F1 = 0.662338
Cell-HOTA DivA = 8.8339
```

Interpretation:

- `pairwise + SAM2 + blend` is the best current compromise.
- The improvement over heuristic F1 is tiny, but Cell-HOTA `DivA` and bud-centric parentage are better.
- Richer context models did not beat it on the current dataset.
- The remaining hard cases are mostly local ranking mistakes among very close candidates in one dense cavity, not long-range lineage mistakes.
- Expert review suggests some metric errors are biologically acceptable: timing shifts, tiny buds, hallucinated/disputable buds, and ambiguous cases.

## Key Notes

Read these first:

```bash
NOTE_MODELES_PARENTAGE_AGENT.md
NOTE_HYBRID_PARENTAGE_STATUS.md
NOTE_SYNTHESE_TRACKING_LEVURE.md
EXPERIMENTS.md
AGENTS.md
```

`NOTE_MODELES_PARENTAGE_AGENT.md` is the main handoff note for models, inputs,
training, strengths, limitations, scores, and function-level implementation map.

## Shared Parentage API

The standardized API is in:

```bash
tools/parentage_api.py
```

Core functions:

```python
build_candidates(...)
build_model_inputs(...)
score_candidates(...)
assign_parentage(...)
write_assignments(...)
```

Core data/scorer classes:

```python
ParentageConfig
CandidateTable
ModelInputs
ScoredCandidates
AssignmentResult
HeuristicScorer
LinearReranker
TransformerReranker
ContextReranker
```

Loaders for learned artifacts:

```python
load_linear_reranker(...)
load_transformer_reranker(...)
load_context_reranker(...)
```

Current integration:

- `tools/learned_bud_rerank.py` uses the shared API in its apply paths.
- `tools/train_bud_context_ranker.py` uses the shared API in its apply path.
- `tools/online_bud_parentage.py` uses the shared API for `mode=global`.
- `mode=online` and `mode=hybrid` still contain custom logic and were not fully migrated.

## Model Implementations

Heuristic global:

```bash
tools/online_bud_parentage.py
```

Main functions:

```python
_build_global_candidates
_score_pair
_score_aggregate_pair
_solve_global_ilp
_assign_global
_assign_hybrid
```

Learned pairwise / pairwise + SAM2 / transformer listwise:

```bash
tools/learned_bud_rerank.py
```

Main functions/classes:

```python
build_training_set
fit_pairwise_model
apply_model_to_root
SAM2EmbeddingExtractor
compute_sam2_features
features_from_candidate
build_listwise_training_samples
ListwiseTransformerRanker
fit_transformer_listwise_model
apply_transformer_model_to_root
```

Contextual dataset:

```bash
tools/build_bud_context_dataset.py
```

Main functions:

```python
compute_pair_frame_features
summarize_candidate_frames
build_sample
write_npz
```

Context ranker:

```bash
tools/train_bud_context_ranker.py
```

Main functions/classes:

```python
ContextRanker
split_train_val_indices
train_model
evaluate_model
apply_model
```

Comparison video renderer:

```bash
tools/render_parentage_comparison.py
```

Useful rendered videos:

```bash
/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1455_ce88ca0_pairwise-temporal-sam2-blend060-v1/review/video13_gt_vs_pairwise_sam2.mp4
/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1721_0488f12_context-ranker-v2-dev-holdout/review/video13_gt_vs_context_v2.mp4
```

Local copies may exist in:

```powershell
C:\Users\charvin\Downloads\video13_gt_vs_pairwise_sam2.mp4
C:\Users\charvin\Downloads\video13_gt_vs_context_v2.mp4
```

## Models Tested

1. `hybrid-cost-v1`
   - No learning.
   - Heuristic candidate scoring + ILP.
   - Robust, interpretable, but limited in dense local ambiguities.

2. `pairwise-temporal-v1`
   - Learned pairwise ranking on heuristic + temporal features.
   - Better on train_dev, weaker on val_holdout.

3. `pairwise-temporal-sam2-v1`
   - Pairwise ranking plus SAM2 object embeddings.
   - Better parentage signal, but embeddings are object-level averages.

4. `pairwise-temporal-sam2-blend060-v1`
   - Best current model.
   - Final score = `0.6 * learned_score + 0.4 * heuristic_score`.
   - Uses ILP after scoring.

5. `transformer_listwise`
   - Listwise transformer on aggregated candidate features.
   - Did not generalize well.

6. `context_ranker_v2/v3/v4`
   - Framewise + relative candidate dataset.
   - v3 adds internal validation and early stopping.
   - v4 adds light candidate attention.
   - Still underperforms the pairwise + SAM2 baseline on holdout.

## Error Analysis Summary

The dominant remaining issue is not "which distant family is correct".
It is local ranking among highly correlated nearby candidates in the same
dense spatial structure:

```text
mother vs recent daughter
mother vs immediate ancestor
mother vs close sister/neighbor
```

For `val_holdout`, remaining `wrong_parent` relations:

```text
pairwise + SAM2:
  ancestor_of_true = 4
  descendant_of_true = 3
  sibling_of_true = 1
  other_branch = 1

context_v2:
  ancestor_of_true = 5
  descendant_of_true = 5
  sibling_of_true = 2
  other_branch = 1
```

User/expert visual feedback on video 13:

- Pairwise + SAM2 looks globally very good.
- Some disagreements are due to tiny buds detected late, or hallucinated/disputable buds.
- Some cases are nearly impossible to decide visually.
- The lower part of the cavity is especially important for the application.

## What Not To Do Next

- Do not keep adding ad hoc rules tuned to video 13 only.
- Do not push a larger transformer without more data or a better input representation.
- Do not optimize only `division_iou_window F1` without expert review; it penalizes timing and ambiguous cases that may not matter biologically.

## Plausible Next Steps

Most useful next steps:

1. Preserve and inspect current local uncommitted changes in `tools/learned_bud_rerank.py`.
   They appear to add SAM2 neck-region features:

   ```text
   sam2_neck_cosine_mean
   sam2_neck_cosine_min
   sam2_neck_valid_fraction
   ```

2. If continuing learned scoring, evaluate these neck-region SAM2 features using the existing `train_dev / val_holdout` protocol.

3. Add a targeted expert-centric evaluation:

   ```text
   bottom-of-cavity focused
   tolerant to bud timing offsets
   separate categories: real parentage error / detection timing / hallucination / ambiguous
   ```

4. If developing a stronger model, focus on local candidate ranking in dense structures, not generic context.

5. If refactoring, finish migrating `mode=hybrid` into `parentage_api.py` while preserving its lock/proposal behavior.

## Safety Rules

- Do not run `git reset --hard` unless explicitly approved.
- Do not remove the local uncommitted changes without checking them.
- Prefer server-side training/evaluation under `/home/charvin-admin/Documents/cellSAM2/experiments/<exp_id>`.
- Keep heavy artifacts out of git.
- Commit code and notes; store checkpoints, masks, videos, and evaluation artifacts under the experiment root.

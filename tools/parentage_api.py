#!/usr/bin/env python3
"""
Programmatic API for interchangeable bud->mother parentage scorers.

This module standardizes the sequence-level pipeline into five stages:
- build_candidates(...)
- build_model_inputs(...)
- score_candidates(...)
- assign_parentage(...)
- write_assignments(...)

The underlying scorers may consume different inputs, but they expose the
same output contract: one score per candidate edge `bud -> mother`.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence

import numpy as np
import torch

from tools import learned_bud_rerank as lbr
from tools import online_bud_parentage as obp


ScorerName = Literal[
    "heuristic",
    "pairwise",
    "pairwise_sam2",
    "pairwise_sam2_blend",
    "transformer_listwise",
    "context",
]


@dataclass
class ParentageConfig:
    refractory: int = 8
    max_dist_factor: float = 2.5
    bud_max_area_ratio: float = 0.6
    min_bud_area: int = 10
    min_mother_age: int = 3
    min_score: float = 0.12
    min_track_length: int = 2
    first_frames: int = 4
    w_dist: float = 0.6
    w_size: float = 0.3
    w_motion: float = 0.1
    w_contact: float = 0.1
    w_neck: float = 0.25
    w_prebud: float = 0.0
    w_angle: float = 0.20
    w_maturity: float = 0.0
    w_track_quality: float = 0.10
    w_margin: float = 0.15
    w_lineage: float = 0.0
    motion_scale: float = 2.0
    interface_radius: int = 4
    use_border_neck: bool = False
    preferred_mother_age: int = 24
    lineage_margin: float = 0.06
    score_threshold: float = 0.5
    learned_score_alpha: float = 1.0
    window_pre: int = 1
    window_post: int = 3
    device: str = ""

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            refractory=self.refractory,
            max_dist_factor=self.max_dist_factor,
            bud_max_area_ratio=self.bud_max_area_ratio,
            min_bud_area=self.min_bud_area,
            min_mother_age=self.min_mother_age,
            min_score=self.min_score,
            min_track_length=self.min_track_length,
            first_frames=self.first_frames,
            w_dist=self.w_dist,
            w_size=self.w_size,
            w_motion=self.w_motion,
            w_contact=self.w_contact,
            w_neck=self.w_neck,
            w_prebud=self.w_prebud,
            w_angle=self.w_angle,
            w_maturity=self.w_maturity,
            w_track_quality=self.w_track_quality,
            w_margin=self.w_margin,
            w_lineage=self.w_lineage,
            motion_scale=self.motion_scale,
            interface_radius=self.interface_radius,
            use_border_neck=self.use_border_neck,
            preferred_mother_age=self.preferred_mother_age,
            lineage_margin=self.lineage_margin,
            score_threshold=self.score_threshold,
            learned_score_alpha=self.learned_score_alpha,
            window_pre=self.window_pre,
            window_post=self.window_post,
            device=self.device,
        )


@dataclass
class CandidateTable:
    seq_dir: Path
    image_dir: Path | None
    track_infos: Dict[int, obp.TrackInfo]
    bud_ids: List[int]
    candidates_by_bud: Dict[int, List[obp.Candidate]]
    masks: Dict[int, np.ndarray]
    stats_cache: Dict[int, Dict[int, obp.ObjStats]]
    bin_masks: Dict[int, Dict[int, np.ndarray]]
    feature_cache: Dict[tuple[int, int], dict[str, float]]
    res_track: np.ndarray
    video_id: str


@dataclass
class ModelInputs:
    scorer_name: ScorerName
    candidate_table: CandidateTable
    payload_by_bud: Dict[int, dict[str, Any]]
    feature_names: Sequence[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateScore:
    bud_id: int
    mother_id: int
    heuristic_score: float
    model_score: float
    score: float
    candidate: obp.Candidate


@dataclass
class ScoredCandidates:
    scorer_name: ScorerName
    candidate_table: CandidateTable
    scored_by_bud: Dict[int, List[CandidateScore]]


@dataclass
class AssignmentResult:
    mode: Literal["ilp", "greedy"]
    scorer_name: ScorerName
    candidate_table: CandidateTable
    scored_candidates: ScoredCandidates
    assigned_by_bud: Dict[int, obp.Candidate]
    updated_res_track: np.ndarray


@dataclass
class HeuristicScorer:
    name: str = "heuristic"


@dataclass
class LinearReranker:
    mean: np.ndarray
    std: np.ndarray
    theta: np.ndarray
    alpha: float = 1.0
    name: str = "pairwise"


@dataclass
class TransformerReranker:
    mean: np.ndarray
    std: np.ndarray
    model: torch.nn.Module
    device: str = ""
    alpha: float = 1.0
    name: str = "transformer_listwise"


@dataclass
class ContextReranker:
    checkpoint: dict[str, Any]
    device: str = ""
    alpha: float = 0.6
    name: str = "context"


def _require_context_dataset_module():
    from tools import build_bud_context_dataset as context_ds

    return context_ds


def _require_context_ranker_module():
    from tools import train_bud_context_ranker as context_ranker

    return context_ranker


def _normalize_scorer_name(name: str) -> ScorerName:
    normalized = name.lower().strip()
    aliases = {
        "heuristic": "heuristic",
        "pairwise": "pairwise",
        "pairwise_sam2": "pairwise_sam2",
        "pairwise-sam2": "pairwise_sam2",
        "pairwise_sam2_blend": "pairwise_sam2_blend",
        "pairwise-temporal-sam2-blend060-v1": "pairwise_sam2_blend",
        "transformer": "transformer_listwise",
        "transformer_listwise": "transformer_listwise",
        "context": "context",
        "contextual": "context",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported scorer name: {name}")
    return aliases[normalized]


def load_linear_reranker(model_path: str | Path, alpha: float = 1.0, name: str = "pairwise") -> LinearReranker:
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    return LinearReranker(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        std=np.asarray(payload["std"], dtype=np.float64),
        theta=np.asarray(payload["theta"], dtype=np.float64),
        alpha=float(alpha),
        name=name,
    )


def load_transformer_reranker(model_path: str | Path, alpha: float = 1.0, device: str = "") -> TransformerReranker:
    payload = torch.load(Path(model_path), map_location="cpu")
    config = payload["model_config"]
    model = lbr.ListwiseTransformerRanker(
        input_dim=int(config["input_dim"]),
        model_dim=int(config["model_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return TransformerReranker(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        std=np.asarray(payload["std"], dtype=np.float64),
        model=model,
        device=device,
        alpha=float(alpha),
        name=str(payload.get("objective", "transformer_listwise")),
    )


def load_context_reranker(model_path: str | Path, alpha: float = 0.6, device: str = "") -> ContextReranker:
    payload = torch.load(Path(model_path), map_location="cpu")
    return ContextReranker(checkpoint=payload, device=device, alpha=float(alpha))


def build_candidates(seq_dir: str | Path, cfg: ParentageConfig, image_dir: str | Path | None = None) -> CandidateTable:
    seq_path = Path(seq_dir).expanduser().resolve()
    seq_context = lbr.build_seq_context(seq_path, cfg.to_namespace())
    return CandidateTable(
        seq_dir=seq_path,
        image_dir=None if image_dir is None else Path(image_dir).expanduser().resolve(),
        track_infos=seq_context["track_infos"],
        bud_ids=list(seq_context["bud_ids"]),
        candidates_by_bud=seq_context["candidates_by_bud"],
        masks=seq_context["masks"],
        stats_cache=seq_context["stats_cache"],
        bin_masks=seq_context["bin_masks"],
        feature_cache=seq_context["feature_cache"],
        res_track=obp._load_res_track(seq_path / "res_track.txt"),
        video_id=seq_path.name,
    )


def _build_pairwise_payload(
    candidate_table: CandidateTable,
    cfg: ParentageConfig,
    embedding_extractor: lbr.SAM2EmbeddingExtractor | None = None,
) -> ModelInputs:
    payload_by_bud: Dict[int, dict[str, Any]] = {}
    seq_context = {
        "seq_dir": candidate_table.seq_dir,
        "track_infos": candidate_table.track_infos,
        "bud_ids": candidate_table.bud_ids,
        "candidates_by_bud": candidate_table.candidates_by_bud,
        "masks": candidate_table.masks,
        "stats_cache": candidate_table.stats_cache,
        "bin_masks": candidate_table.bin_masks,
        "feature_cache": candidate_table.feature_cache,
    }
    for bud_id, cand_list in candidate_table.candidates_by_bud.items():
        proposal_parent = candidate_table.track_infos[bud_id].parent
        feature_rows = []
        extra_by_pair: Dict[tuple[int, int], dict[str, float]] = {}
        for cand in cand_list:
            extra = lbr.get_extra_features(cand, seq_context, candidate_table.image_dir, embedding_extractor, cfg.to_namespace())
            extra_by_pair[(cand.bud_id, cand.mother_id)] = extra
            feature_rows.append(lbr.features_from_candidate(cand, proposal_parent, extra))
        payload_by_bud[bud_id] = {
            "candidates": list(cand_list),
            "proposal_parent": proposal_parent,
            "feature_rows": np.stack(feature_rows, axis=0) if feature_rows else np.zeros((0, len(lbr.FEATURE_NAMES)), dtype=np.float64),
            "extra_by_pair": extra_by_pair,
        }
    return ModelInputs(
        scorer_name="pairwise",
        candidate_table=candidate_table,
        payload_by_bud=payload_by_bud,
        feature_names=lbr.FEATURE_NAMES,
    )


def _build_context_payload(candidate_table: CandidateTable, cfg: ParentageConfig) -> ModelInputs:
    context_ds = _require_context_dataset_module()
    payload_by_bud: Dict[int, dict[str, Any]] = {}
    frame_offsets = list(range(-cfg.window_pre, cfg.window_post + 1))
    frame_ids_by_bud = {
        bud_id: [candidate_table.track_infos[bud_id].start + off for off in frame_offsets]
        for bud_id in candidate_table.bud_ids
    }
    for bud_id, cand_list in candidate_table.candidates_by_bud.items():
        proposal_parent = candidate_table.track_infos[bud_id].parent
        candidate_ids = [cand.mother_id for cand in cand_list]
        bud_track = candidate_table.track_infos[bud_id]
        first_stats = candidate_table.stats_cache.get(bud_track.start, {}).get(bud_id)
        last_stats = candidate_table.stats_cache.get(bud_track.end, {}).get(bud_id)
        bud_global = np.asarray(
            [
                float(bud_track.end - bud_track.start + 1),
                float(len(cand_list)),
                float(0 if first_stats is None else first_stats.area),
                float(0 if last_stats is None else last_stats.area),
                float((0 if last_stats is None else last_stats.area) - (0 if first_stats is None else first_stats.area)),
            ],
            dtype=np.float32,
        )
        pair_rows = []
        pair_valid = []
        cand_global_rows = []
        cand_valid = []
        for cand in cand_list:
            frame_features = []
            valid_flags = []
            for frame_idx in frame_ids_by_bud[bud_id]:
                feats, valid = context_ds.compute_pair_frame_features(
                    frame_idx=frame_idx,
                    candidate=cand,
                    candidate_ids=candidate_ids,
                    seq_context={
                        "stats_cache": candidate_table.stats_cache,
                        "bin_masks": candidate_table.bin_masks,
                        "track_infos": candidate_table.track_infos,
                    },
                    interface_radius=cfg.interface_radius,
                    use_border_neck=cfg.use_border_neck,
                )
                frame_features.append(feats)
                valid_flags.append(valid)
            summary = context_ds.summarize_candidate_frames(frame_features, valid_flags)
            candidate_global = np.asarray(
                [
                    float(cand.score),
                    float(cand.dist),
                    float(cand.size_ratio),
                    float(cand.motion),
                    float(cand.contact),
                    float(cand.neck),
                    float(cand.prebud),
                    float(cand.angle),
                    float(cand.maturity),
                    float(cand.track_quality),
                    float(cand.margin),
                    float(cand.mother_age),
                    1.0 if cand.mother_id == proposal_parent else 0.0,
                    float(cand.lineage),
                    float(summary["alive_fraction"]),
                    float(summary["best_dist_fraction"]),
                    float(summary["best_contact_fraction"]),
                    float(summary["mean_dist_norm"]),
                    float(summary["min_dist_norm"]),
                    float(summary["mean_contact"]),
                    float(summary["max_contact"]),
                    float(summary["mean_neck"]),
                    float(summary["max_neck"]),
                ],
                dtype=np.float32,
            )
            pair_rows.append(frame_features)
            pair_valid.append(valid_flags)
            cand_global_rows.append(candidate_global)
            cand_valid.append(True)
        payload_by_bud[bud_id] = {
            "candidates": list(cand_list),
            "frame_offsets": frame_offsets,
            "frame_ids": frame_ids_by_bud[bud_id],
            "x_pair": np.asarray(pair_rows, dtype=np.float32),
            "x_pair_valid": np.asarray(pair_valid, dtype=np.bool_),
            "x_candidate_global": np.asarray(cand_global_rows, dtype=np.float32),
            "x_candidate_valid": np.asarray(cand_valid, dtype=np.bool_),
            "x_bud_global": bud_global,
        }
    return ModelInputs(
        scorer_name="context",
        candidate_table=candidate_table,
        payload_by_bud=payload_by_bud,
        feature_names=tuple(context_ds.PAIR_FEATURE_NAMES),
        metadata={
            "pair_feature_names": list(context_ds.PAIR_FEATURE_NAMES),
            "candidate_global_feature_names": list(context_ds.CANDIDATE_GLOBAL_FEATURE_NAMES),
            "bud_global_feature_names": list(context_ds.BUD_GLOBAL_FEATURE_NAMES),
        },
    )


def build_model_inputs(
    candidate_table: CandidateTable,
    scorer_name: str,
    cfg: ParentageConfig,
    embedding_extractor: lbr.SAM2EmbeddingExtractor | None = None,
) -> ModelInputs:
    scorer = _normalize_scorer_name(scorer_name)
    if scorer == "heuristic":
        payload_by_bud = {
            bud_id: {"candidates": list(cand_list)}
            for bud_id, cand_list in candidate_table.candidates_by_bud.items()
        }
        return ModelInputs(
            scorer_name=scorer,
            candidate_table=candidate_table,
            payload_by_bud=payload_by_bud,
            feature_names=(),
        )
    if scorer in {"pairwise", "pairwise_sam2", "pairwise_sam2_blend", "transformer_listwise"}:
        inputs = _build_pairwise_payload(candidate_table, cfg, embedding_extractor=embedding_extractor)
        inputs.scorer_name = scorer
        return inputs
    if scorer == "context":
        return _build_context_payload(candidate_table, cfg)
    raise AssertionError(f"Unhandled scorer {scorer}")


def _softmax_scores(logits: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    masked = np.where(valid_mask, logits, -1e9)
    masked = masked - np.max(masked)
    exp = np.exp(masked) * valid_mask
    denom = float(exp.sum()) if float(exp.sum()) > 0 else 1.0
    return exp / denom


def score_candidates(
    model_inputs: ModelInputs,
    scorer: HeuristicScorer | LinearReranker | TransformerReranker | ContextReranker,
    cfg: ParentageConfig,
) -> ScoredCandidates:
    scored_by_bud: Dict[int, List[CandidateScore]] = {}
    if isinstance(scorer, TransformerReranker):
        device_name = scorer.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        scorer.model = scorer.model.to(device)
        scorer.model.eval()

    for bud_id, payload in model_inputs.payload_by_bud.items():
        candidates = payload["candidates"]
        if isinstance(scorer, HeuristicScorer):
            probs = np.asarray([float(cand.score) for cand in candidates], dtype=np.float64)
            heuristic_scores = probs.copy()
        elif isinstance(scorer, LinearReranker):
            feature_rows = payload["feature_rows"]
            heuristic_scores = np.asarray([float(cand.score) for cand in candidates], dtype=np.float64)
            probs = lbr.predict_prob(feature_rows, scorer.mean, scorer.std, scorer.theta)
        elif isinstance(scorer, TransformerReranker):
            feature_rows = payload["feature_rows"]
            heuristic_scores = np.asarray([float(cand.score) for cand in candidates], dtype=np.float64)
            probs = lbr.predict_transformer_scores(feature_rows, scorer.mean, scorer.std, scorer.model, torch.device(scorer.device or ("cuda" if torch.cuda.is_available() else "cpu")))
        elif isinstance(scorer, ContextReranker):
            context_ranker = _require_context_ranker_module()
            device_name = scorer.device or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_name)
            x_pair = torch.from_numpy(payload["x_pair"][None, ...]).to(device)
            x_pair_valid = torch.from_numpy(payload["x_pair_valid"][None, ...]).to(device).bool()
            x_cand = torch.from_numpy(payload["x_candidate_global"][None, ...]).to(device)
            x_cand_valid = torch.from_numpy(payload["x_candidate_valid"][None, ...]).to(device).bool()
            x_bud = torch.from_numpy(payload["x_bud_global"][None, ...]).to(device)
            model = context_ranker.ContextRanker(
                pair_dim=int(payload["x_pair"].shape[-1]),
                cand_dim=int(payload["x_candidate_global"].shape[-1]),
                bud_dim=int(payload["x_bud_global"].shape[-1]),
                hidden_dim=int(scorer.checkpoint["hidden_dim"]),
                dropout=float(scorer.checkpoint["dropout"]),
                model_type=str(scorer.checkpoint.get("model_type", "baseline")),
                candidate_heads=int(scorer.checkpoint.get("candidate_heads", 4)),
            ).to(device)
            model.load_state_dict(scorer.checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                logits = model(x_pair, x_pair_valid, x_cand, x_cand_valid, x_bud)[0].detach().cpu().numpy()
            heuristic_scores = np.asarray([float(cand.score) for cand in candidates], dtype=np.float64)
            probs = _softmax_scores(logits, payload["x_candidate_valid"])
        else:
            raise TypeError(f"Unsupported scorer type: {type(scorer)!r}")

        alpha = getattr(scorer, "alpha", 1.0)
        rows = []
        for cand, heuristic_score, model_score in zip(candidates, heuristic_scores.tolist(), probs.tolist()):
            final_score = float(np.clip(alpha * model_score + (1.0 - alpha) * heuristic_score, 0.0, 1.0))
            rows.append(
                CandidateScore(
                    bud_id=int(cand.bud_id),
                    mother_id=int(cand.mother_id),
                    heuristic_score=float(heuristic_score),
                    model_score=float(model_score),
                    score=final_score,
                    candidate=replace(cand, score=final_score),
                )
            )
        rows.sort(key=lambda item: item.score, reverse=True)
        scored_by_bud[bud_id] = rows

    return ScoredCandidates(
        scorer_name=model_inputs.scorer_name,
        candidate_table=model_inputs.candidate_table,
        scored_by_bud=scored_by_bud,
    )


def assign_parentage(
    candidate_table: CandidateTable,
    scored_candidates: ScoredCandidates,
    cfg: ParentageConfig,
    mode: Literal["ilp", "greedy"] = "ilp",
) -> AssignmentResult:
    flat_candidates = [
        score_row.candidate
        for bud_id in candidate_table.bud_ids
        for score_row in scored_candidates.scored_by_bud.get(bud_id, [])
    ]
    if mode == "ilp":
        assigned = obp._solve_global_ilp(
            bud_ids=candidate_table.bud_ids,
            candidates=flat_candidates,
            track_infos=candidate_table.track_infos,
            refractory_frames=cfg.refractory,
        )
    else:
        assigned = obp._solve_global_greedy(
            bud_ids=candidate_table.bud_ids,
            candidates=flat_candidates,
            track_infos=candidate_table.track_infos,
            refractory_frames=cfg.refractory,
        )

    res_track = deepcopy(candidate_table.res_track)
    for bud_id, cand in assigned.items():
        if cand.score < cfg.score_threshold:
            continue
        mask = res_track[:, 0] == bud_id
        if mask.any():
            res_track[mask, 3] = cand.mother_id

    return AssignmentResult(
        mode=mode,
        scorer_name=scored_candidates.scorer_name,
        candidate_table=candidate_table,
        scored_candidates=scored_candidates,
        assigned_by_bud=assigned,
        updated_res_track=res_track,
    )


def write_assignments(
    assignment: AssignmentResult,
    output_res_track_path: str | Path,
    output_csv_path: str | Path | None = None,
) -> None:
    output_res_track = Path(output_res_track_path).expanduser().resolve()
    output_res_track.parent.mkdir(parents=True, exist_ok=True)
    obp._write_res_track(output_res_track, assignment.updated_res_track)

    if output_csv_path is None:
        return

    output_csv = Path(output_csv_path).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bud_id",
                "mother_id",
                "heuristic_score",
                "model_score",
                "final_score",
                "selected",
                "selected_parent",
                "scorer_name",
            ]
        )
        for bud_id in assignment.candidate_table.bud_ids:
            selected = assignment.assigned_by_bud.get(bud_id)
            selected_parent = -1 if selected is None else int(selected.mother_id)
            for row in assignment.scored_candidates.scored_by_bud.get(bud_id, []):
                writer.writerow(
                    [
                        bud_id,
                        row.mother_id,
                        f"{row.heuristic_score:.6f}",
                        f"{row.model_score:.6f}",
                        f"{row.score:.6f}",
                        int(selected is not None and row.mother_id == selected_parent),
                        selected_parent,
                        assignment.scorer_name,
                    ]
                )


__all__ = [
    "AssignmentResult",
    "CandidateScore",
    "CandidateTable",
    "ContextReranker",
    "HeuristicScorer",
    "LinearReranker",
    "ModelInputs",
    "ParentageConfig",
    "ScoredCandidates",
    "TransformerReranker",
    "assign_parentage",
    "build_candidates",
    "build_model_inputs",
    "load_context_reranker",
    "load_linear_reranker",
    "load_transformer_reranker",
    "score_candidates",
    "write_assignments",
]

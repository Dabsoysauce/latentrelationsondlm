"""Small CPU-only artifact integration runner for every corrected experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import canonical_hash, scientific_configuration
from ..config import RELATION_NAMES, RunConfig
from ..paper_protocol import (
    PaperLockSet,
    choose_selection_winners,
    write_resolved_selection_locks,
    write_selection_bundle,
)
from .shared import write_frames

FAKE_HEADS = tuple((layer, head) for layer in range(2) for head in range(3))


def _selection(cfg: RunConfig, run_dir: Path, manifest_hashes):
    scores = []
    for relation_index, relation in enumerate(RELATION_NAMES):
        winner = relation_index % len(FAKE_HEADS)
        for head_index, (layer, head) in enumerate(FAKE_HEADS):
            correct = 20 - abs(head_index - winner)
            scores.append(
                {
                    "relation": relation,
                    "layer": layer,
                    "head": head,
                    "accuracy": correct / 20,
                    "n_total": 20,
                    "n_correct": correct,
                }
            )
    scores = pd.DataFrame(scores)
    winners = choose_selection_winners(scores)
    scores.to_csv(run_dir / "selection_all_head_scores.csv", index=False)
    winners.to_csv(run_dir / "selection_winners.csv", index=False)
    locks = write_selection_bundle(
        run_dir / "selection-locks",
        winners,
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        tokenizer_revision=cfg.model.tokenizer_revision,
        dataset_id=cfg.dataset.id,
        selection_manifest_hash=manifest_hashes["select"],
        config_hash=canonical_hash(scientific_configuration(cfg.to_dict())),
        code_hash="fake",
        created_at="2000-01-01T00:00:00+00:00",
    )
    rows = []
    for relation, lock in locks.locks.items():
        for index in range(8):
            rows.append(
                {
                    "sentence_id": f"test-{relation}-{index}",
                    "instance_id": f"test-{relation}-{index}:0",
                    "relation": relation,
                    "seed": 42,
                    "timestep": 63,
                    "normalized_progress": 1.0,
                    "visibility": "at_least_one_revealed",
                    "layer": lock.layer,
                    "head": lock.head,
                    "predicted_source_subtoken": 2,
                    "receiver_span": [2, 3],
                    "correct": int(index < 6),
                }
            )
    raw = pd.DataFrame(rows)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    metrics = raw.groupby(["relation", "layer", "head"], as_index=False).agg(
        numerator=("correct", "sum"), denominator=("correct", "size"), accuracy=("correct", "mean")
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    metrics.assign(seed=42).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    write_resolved_selection_locks(run_dir, locks)
    return {
        "fake_cpu_only": True,
        "fake_validation_runner": cfg.experiment.type,
        "development_used": False,
        "permutations_used": False,
        "holm_correction_used": False,
        "relations_locked": list(RELATION_NAMES),
        "selection_lock_dir": str(run_dir / "selection-locks"),
    }


def _generic(cfg: RunConfig, run_dir: Path, source_locks: PaperLockSet | None):
    rows = []
    time_resolved = len(cfg.experiment.normalized_progress) == 64
    progress_points = (
        [(timestep, timestep / 63) for timestep in range(64)]
        if time_resolved
        else [
            (round(progress * 63), progress)
            for progress in cfg.experiment.normalized_progress
        ]
    )
    for seed in cfg.experiment.seeds:
        for timestep, normalized_progress in progress_points:
            if cfg.experiment.type in {
                "relation_head_receiver_prediction_over_diffusion_time",
                "attention_entropy",
                "attention_heatmaps_and_trajectories",
                "multilingual_relation_head_transfer",
            } and timestep in {0, 63} and seed != 42:
                continue
            relations = RELATION_NAMES if source_locks is not None else ("not_applicable",)
            for relation in relations:
                lock = source_locks.resolve(relation) if source_locks is not None else None
                rows.append(
                    {
                        "sentence_id": f"fake-{relation}",
                        "instance_id": f"fake-{relation}:0",
                        "relation": relation,
                        "seed": seed,
                        "timestep": timestep,
                        "normalized_progress": normalized_progress,
                        "visibility": (
                            "both_masked" if timestep < 32 else "at_least_one_revealed"
                        ),
                        "layer": lock.layer if lock else 1,
                        "head": lock.head if lock else 0,
                        "relative_label": "middle",
                        "actual_layer_index": 1,
                        "total_model_layers": 2,
                        "word_index": 1,
                        "label": "NOUN",
                        "prediction": "NOUN",
                        "feature_kind": "residual",
                        "correct": int((seed + timestep) % 3 != 0),
                        "entropy_normalized": (timestep + 1) / 64,
                        "entropy": (timestep + 1) / 64,
                        "relation_attention_mass": 0.5,
                        "top1": int(timestep > 31),
                        "top5": int(timestep > 15),
                        "rank": max(1, 64 - timestep),
                        "mrr": 1.0 / max(1, 64 - timestep),
                        "target_token_id": 1,
                        "prediction_offset": 0,
                        "found_time": 10,
                        "unmask_time": 12,
                        "lead_steps": 2,
                        "token_class": "content_word",
                        "control_kind": "selected_relation_head",
                        "target_logit_support": 0.1,
                        "target_rank": 2,
                        "vocabulary_percentile": 0.9,
                        "target_logit_change": -0.1,
                        "target_probability_change": -0.01,
                        "target_rank_change": 1,
                        "predicted_source_subtoken": 2,
                    }
                )
    raw = pd.DataFrame(rows)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    group = ["seed", "relation", "timestep", "layer", "head"]
    per_seed = raw.groupby(group, as_index=False).agg(
        accuracy=("correct", "mean"),
        entropy=("entropy_normalized", "mean"),
        top1=("top1", "mean"),
        denominator=("correct", "size"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(group[1:], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        seed_std=("accuracy", "std"),
        denominator=("denominator", "sum"),
        n_seeds=("seed", "nunique"),
    ).to_csv(run_dir / "metrics.csv", index=False)
    if source_locks is not None:
        write_resolved_selection_locks(run_dir, source_locks)
    if cfg.experiment.type in {
        "final_token_prediction_by_layer",
        "prediction_before_unmasking_timing_analysis",
    }:
        pd.DataFrame(
            [
                {
                    "prompt_id": "fake",
                    "seed": 42,
                    "pre_forward_ids": [[0, 0] for _ in range(64)],
                    "argmax_ids": [[1, 1] for _ in range(64)],
                    "final_ids": [1, 1],
                }
            ]
        ).to_parquet(run_dir / "native_trajectories.parquet", index=False)
    return {
        "fake_cpu_only": True,
        "fake_validation_runner": cfg.experiment.type,
        "development_used": False,
        "timesteps": [timestep for timestep, _progress in progress_points],
        "seeds": list(cfg.experiment.seeds),
    }


def run(
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict[str, str],
    source_locks: PaperLockSet | None,
    **_unused: Any,
) -> dict[str, Any]:
    if cfg.experiment.type == "relation_head_receiver_prediction":
        return _selection(cfg, run_dir, manifest_hashes)
    required = {
        "relation_head_receiver_prediction_over_diffusion_time",
        "direct_logit_attribution",
        "matched_relation_head_ablation",
        "attention_heatmaps_and_trajectories",
        "multilingual_relation_head_transfer",
    }
    if cfg.experiment.type in required and source_locks is None:
        raise ValueError(f"{cfg.experiment.type} requires a paper selection lock")
    return _generic(cfg, run_dir, source_locks)

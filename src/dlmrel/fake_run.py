"""Deterministic CPU end-to-end run for artifact and protocol verification."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .artifacts import ArtifactError, atomic_json, final_artifact_hashes
from .config import RELATION_NAMES, RunConfig
from .experiments.shared import aggregate_head_scores, write_frames
from .experiments.time_curve import aggregate_curve
from .head_search_recovery import complete_cpu_finalization, write_test_all_head_evidence
from .models.fake import FakeAdapter
from .relation_selection import (
    derive_relation_selection_bundle,
    filter_relation_locked_rows,
    install_primary_aliases,
    load_relation_locks,
    write_resolved_lock_manifest,
)

FAKE_PERMUTATIONS = 10
FAKE_HEADS = tuple((layer, head) for layer in range(2) for head in range(3))


def run_fake(cfg: RunConfig, run_dir: Path) -> None:
    """Exercise six locks, serialization, permutation checkpoints, and finalization on CPU."""
    if cfg.experiment.type == "time_curve":
        _run_fake_time_curve(cfg, run_dir)
        return
    if cfg.experiment.type == "attention_entropy":
        _run_fake_entropy(cfg, run_dir)
        return
    if cfg.experiment.type == "logit_lens":
        _run_fake_logit_lens(cfg, run_dir)
        return
    if cfg.experiment.type == "pos_probe":
        _run_fake_pos_probe(cfg, run_dir)
        return
    if cfg.track == "external_treebank_transfer":
        if not cfg.runtime.selection_lock:
            raise ArtifactError("external transfer requires an EWT relation-lock source")
        locks = load_relation_locks(cfg.runtime.selection_lock, cfg)
        test_all = _all_head_rows(cfg, "test")
    else:
        select_rows = _all_head_rows(cfg, "select")
        dev_rows = _all_head_rows(cfg, "dev")
        aggregate_head_scores(select_rows).to_csv(
            run_dir / "select_all_head_scores.csv", index=False
        )
        aggregate_head_scores(dev_rows).to_csv(
            run_dir / "dev_all_head_scores.csv", index=False
        )
        select_rows.to_parquet(run_dir / "select_instances.parquet", index=False)
        dev_rows.to_parquet(run_dir / "dev_instances.parquet", index=False)
        bundle = derive_relation_selection_bundle(
            run_dir,
            run_dir / "relation-selection",
            require_complete=False,
            allow_source_output=True,
            allow_existing=True,
        )
        install_primary_aliases(run_dir, bundle)
        locks = load_relation_locks(bundle.output_dir, cfg)
        test_all = _all_head_rows(cfg, "test")
        write_test_all_head_evidence(run_dir, cfg, test_all)

        selected = filter_relation_locked_rows(test_all, locks)
        write_frames(run_dir, raw=selected, exclusions=pd.DataFrame())
        complete_cpu_finalization(
            cfg,
            run_dir,
            n_permutations=FAKE_PERMUTATIONS,
            checkpoint_interval=5,
        )
        return

    selected = filter_relation_locked_rows(test_all, locks)
    write_resolved_lock_manifest(run_dir, locks)
    write_frames(run_dir, raw=selected, exclusions=pd.DataFrame())
    per_seed = selected.groupby(
        ["seed", "treebank", "relation", "layer", "head"], as_index=False
    ).agg(accuracy=("correct", "mean"), n_instances=("correct", "size"))
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(["treebank", "relation", "layer", "head"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        seed_std=("accuracy", "std"),
        n_seeds=("seed", "nunique"),
    ).to_csv(run_dir / "metrics.csv", index=False)
    _finalize_fake_run(
        cfg,
        run_dir,
        {
            "n_instances": int(selected["instance_id"].nunique()),
            "relations": sorted(selected["relation"].unique()),
            "relation_locks_applied": len(locks.locks),
        },
    )


def _finalize_fake_run(cfg: RunConfig, run_dir: Path, details: dict) -> None:
    atomic_json(
        run_dir / "summary.json",
        {
            "schema_version": "dlmrel-run-v1",
            "completion_status": "complete",
            "capabilities": asdict(FakeAdapter.capabilities),
            "fake_cpu_only": True,
            **details,
        },
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "completion_status": "complete",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "model_revision": cfg.model.revision,
            "tokenizer_revision": cfg.model.tokenizer_revision,
            "remote_code_revision": cfg.model.remote_code_revision,
            "final_artifact_hashes": final_artifact_hashes(run_dir),
        }
    )
    atomic_json(metadata_path, metadata)


def _run_fake_time_curve(cfg: RunConfig, run_dir: Path) -> None:
    if not cfg.runtime.selection_lock:
        raise ArtifactError("fake time curve requires an EWT relation-lock source")
    locks = load_relation_locks(cfg.runtime.selection_lock, cfg)
    frames = []
    visibility = (
        "both_masked",
        "attender_visible_only",
        "receiver_visible_only",
        "both_visible",
    )
    base = filter_relation_locked_rows(_all_head_rows(cfg, "test"), locks)
    for progress_index, progress in enumerate(cfg.experiment.normalized_progress):
        frame = base.copy()
        frame["normalized_progress"] = progress
        frame["timestep"] = round(progress * (cfg.experiment.steps - 1))
        frame["visibility"] = visibility[progress_index % len(visibility)]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    write_resolved_lock_manifest(run_dir, locks)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    per_seed, metrics = aggregate_curve(raw)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    _finalize_fake_run(cfg, run_dir, {"n_rows": len(raw), "fake_validation_runner": "time_curve"})


def _run_fake_entropy(cfg: RunConfig, run_dir: Path) -> None:
    rows = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            for sentence_index in range(2):
                for layer, head in FAKE_HEADS:
                    rows.append(
                        {
                            "sentence_id": f"test-{sentence_index}",
                            "treebank": cfg.dataset.treebank,
                            "seed": seed,
                            "timestep": round(progress * (cfg.experiment.steps - 1)),
                            "normalized_progress": progress,
                            "layer": layer,
                            "head": head,
                            "entropy": float(layer + head + 1) / 10,
                            "entropy_normalized": float(layer + head + 1) / 20,
                            "entropy_no_bos": float(layer + 1) / 10,
                            "bos_sink_mass": float(head + 1) / 10,
                            "valid_key_count": 4,
                            "n_masked": int((1 - progress) * 3),
                        }
                    )
    raw = pd.DataFrame(rows)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    group = ["seed", "treebank", "timestep", "normalized_progress", "layer", "head"]
    per_seed = raw.groupby(group, as_index=False).mean(numeric_only=True)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(group[1:], as_index=False).agg(
        entropy_mean=("entropy", "mean"), n_seeds=("seed", "nunique")
    ).to_csv(run_dir / "metrics.csv", index=False)
    _finalize_fake_run(cfg, run_dir, {"n_rows": len(raw), "fake_validation_runner": "attention_entropy"})


def _run_fake_logit_lens(cfg: RunConfig, run_dir: Path) -> None:
    rows = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            for depth in range(3):
                for position in range(4):
                    rank = 1 + (depth + position) % 7
                    rows.append(
                        {
                            "sentence_id": "test-0",
                            "treebank": cfg.dataset.treebank,
                            "seed": seed,
                            "timestep": round(progress * (cfg.experiment.steps - 1)),
                            "normalized_progress": progress,
                            "depth": depth,
                            "position": position,
                            "position_state": "visible" if position == 0 else "masked",
                            "target_token_id": position,
                            "top1": int(rank == 1),
                            "top5": int(rank <= 5),
                            "rank": rank,
                            "mrr": 1.0 / rank,
                            "target_logit": float(depth - position),
                        }
                    )
    raw = pd.DataFrame(rows)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    group = [
        "seed",
        "treebank",
        "timestep",
        "normalized_progress",
        "depth",
        "position_state",
    ]
    per_seed = raw.groupby(group, as_index=False).agg(
        top1=("top1", "mean"), top5=("top5", "mean"), mrr=("mrr", "mean"),
        mean_rank=("rank", "mean"), n_positions=("rank", "size")
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(group[1:], as_index=False).agg(
        top1=("top1", "mean"), top5=("top5", "mean"), mrr=("mrr", "mean"),
        mean_rank=("mean_rank", "mean"), n_positions=("n_positions", "sum"),
        n_seeds=("seed", "nunique")
    ).to_csv(run_dir / "metrics.csv", index=False)
    _finalize_fake_run(
        cfg,
        run_dir,
        {"n_rows": len(raw), "final_depth_max_abs_parity_error": 0.0,
         "fake_validation_runner": "logit_lens"},
    )


def _run_fake_pos_probe(cfg: RunConfig, run_dir: Path) -> None:
    rows = []
    metrics = []
    for seed in cfg.experiment.seeds:
        for word_index, gold in enumerate(("DET", "NOUN", "VERB", "NOUN")):
            rows.append(
                {
                    "seed": seed,
                    "sentence_id": "test-0",
                    "word_index": word_index,
                    "form": f"word-{word_index}",
                    "gold_upos": gold,
                    "prediction": gold,
                    "lexical_prediction": gold,
                    "shuffled_prediction": "NOUN",
                    "random_feature_prediction": "NOUN",
                }
            )
        metrics.append(
            {
                "seed": seed,
                "selected_c": 0.01,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "majority_accuracy": 0.5,
                "lexical_accuracy": 1.0,
                "shuffled_accuracy": 0.5,
                "random_feature_accuracy": 0.5,
                "n_test_positions": 4,
                "n_test_sentences": 1,
            }
        )
    raw = pd.DataFrame(rows)
    per_seed = pd.DataFrame(metrics)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    score_columns = [
        "accuracy", "macro_f1", "majority_accuracy", "lexical_accuracy",
        "shuffled_accuracy", "random_feature_accuracy"
    ]
    summary = {"n_seeds": len(cfg.experiment.seeds), "n_test_positions": len(raw)}
    for column in score_columns:
        summary[column] = per_seed[column].mean()
        summary[f"{column}_seed_std"] = per_seed[column].std()
    pd.DataFrame([summary]).to_csv(run_dir / "metrics.csv", index=False)
    _finalize_fake_run(
        cfg,
        run_dir,
        {"n_rows": len(raw), "selected_c_by_seed": {str(seed): 0.01 for seed in cfg.experiment.seeds},
         "fake_validation_runner": "pos_probe"},
    )


def _all_head_rows(cfg: RunConfig, role: str) -> pd.DataFrame:
    rows = []
    progress = 0.0
    timestep = 0
    for relation_index, relation in enumerate(RELATION_NAMES):
        for instance_index in range(10):
            gold = 1 + instance_index % 3
            for seed_index, seed in enumerate(cfg.experiment.seeds):
                observation = seed_index * 10 + instance_index
                for head_index, (layer, head) in enumerate(FAKE_HEADS):
                    if role == "select":
                        correct_limit = 30 - head_index
                    elif role == "dev":
                        correct_limit = 30 if head_index == (relation_index + 1) % 5 else 20 - head_index
                    else:
                        correct_limit = 30 if head_index == (relation_index + 2) % 6 else 12
                    correct = int(observation < max(correct_limit, 0))
                    rows.append(
                        {
                            "sentence_id": f"{role}-{relation}-{instance_index}",
                            "instance_id": f"{role}-{relation}-{instance_index}:0",
                            "sentence": "alpha beta gamma delta",
                            "treebank": cfg.dataset.treebank,
                            "language": cfg.dataset.language,
                            "original_split": {"select": "train", "dev": "dev", "test": "test"}[
                                role
                            ],
                            "role": role,
                            "relation": relation,
                            "seed": seed,
                            "timestep": timestep,
                            "normalized_progress": progress,
                            "visibility": "both_masked",
                            "layer": layer,
                            "head": head,
                            "attender_word_idx": 0,
                            "gold_receiver_word_idx": gold,
                            "receiver_word_idx": gold,
                            "predicted_word_idx": gold if correct else ((gold % 3) + 1),
                            "correct": correct,
                            "attender_span": [1],
                            "receiver_span": [gold + 1],
                            "signed_distance": relation_index + 1,
                            "direction": "right",
                            "coordinated": bool(relation_index % 2),
                            "embedded_clause": bool(relation_index % 3 == 0),
                            "relative_clause": bool(relation_index % 4 == 0),
                            "passive_voice": bool(relation_index % 5 == 0),
                            "punctuation_between": bool(instance_index % 2),
                            "sentence_length_words": 4,
                            "n_candidate_words": 3,
                            "nearest_correct": correct,
                            "uniform_correct": int((observation + head_index) % 3 == 0),
                            "previous_correct": int(gold == 1),
                            "next_correct": int(gold == 1),
                            "oracle_pos_correct": correct,
                            "wrong_same_pos_correct": 0,
                            "gold_attention_mass": 0.75 if correct else 0.25,
                            "matched_attention_mass": 0.25 if correct else 0.75,
                            "matched_gold_greater": bool(correct),
                        }
                    )
    return pd.DataFrame(rows)

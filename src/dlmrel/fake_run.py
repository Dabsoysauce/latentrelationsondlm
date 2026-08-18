"""Deterministic CPU end-to-end run for artifact and protocol verification."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .artifacts import ArtifactError, atomic_json
from .config import RELATION_NAMES, RunConfig
from .experiments.shared import aggregate_head_scores, write_frames
from .models.fake import FakeAdapter
from .permutation import selection_aware_permutation
from .relation_selection import (
    MINIMUM_DENOMINATOR,
    PRIMARY_RELATION,
    RelationLockSet,
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
        _write_fake_permutations(run_dir, cfg, select_rows, dev_rows, test_all, locks)

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
    atomic_json(
        run_dir / "summary.json",
        {
            "schema_version": "dlmrel-run-v1",
            "completion_status": "complete",
            "n_instances": int(selected["instance_id"].nunique()),
            "relations": sorted(selected["relation"].unique()),
            "relation_locks_applied": len(locks.locks),
            "capabilities": asdict(FakeAdapter.capabilities),
            "fake_cpu_only": True,
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
        }
    )
    atomic_json(metadata_path, metadata)


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
                            "sentence_length_words": 4,
                            "n_candidate_words": 3,
                        }
                    )
    return pd.DataFrame(rows)


def _write_fake_permutations(
    run_dir: Path,
    cfg: RunConfig,
    select_rows: pd.DataFrame,
    dev_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    locks: RelationLockSet,
) -> None:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    results = []
    for relation in RELATION_NAMES:
        result = selection_aware_permutation(
            select_rows,
            dev_rows,
            test_rows,
            relation=relation,
            top_k=cfg.experiment.scoring.top_k,
            n_permutations=FAKE_PERMUTATIONS,
            seed=42,
            scientific_config_hash=metadata["scientific_config_hash"],
            minimum_denominator=MINIMUM_DENOMINATOR,
            checkpoint_path=run_dir / "permutation-checkpoints" / f"{relation}.json",
            resume=cfg.runtime.resume,
            progress_interval=0,
            checkpoint_interval=5,
        )
        lock = locks.resolve(relation)
        if (result["observed_selected_layer"], result["observed_selected_head"]) != (
            lock.layer,
            lock.head,
        ):
            raise ArtifactError(
                f"fake permutation protocol disagrees with lock for {relation!r}"
            )
        atomic_json(run_dir / "permutations" / f"{relation}.json", result)
        results.append(
            {
                "relation": relation,
                "observed_test_accuracy": result["observed_test_accuracy"],
                "p_value": result["p_value"],
                "n_permutations": result["n_permutations"],
                "null_definition": result["null_definition"],
            }
        )
        if relation == PRIMARY_RELATION:
            atomic_json(run_dir / "selection_permutation.json", result)
    pd.DataFrame(results).to_csv(run_dir / "selection_permutation_results.csv", index=False)

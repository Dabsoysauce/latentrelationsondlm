"""Select heads on EWT select/dev, test once, and transfer the frozen head."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import (
    ArtifactError,
    SelectionLock,
    atomic_json,
    selection_lock_hash,
)
from ..config import RunConfig
from ..data import load_manifest_examples
from ..relation_selection import derive_relation_selection_bundle, install_primary_aliases
from .shared import (
    aggregate_head_scores,
    locked_metrics,
    per_seed_metrics,
    score_over_seeds,
    structural_slices,
    write_frames,
)


def run(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict[str, str],
    source_lock: SelectionLock | None = None,
) -> dict[str, Any]:
    if cfg.track == "external_treebank_transfer":
        if source_lock is None:
            raise ArtifactError("external transfer requires an EWT selection lock")
        return run_locked_transfer(model, tokenizer, cfg, run_dir, source_lock)
    return run_head_search(model, tokenizer, cfg, run_dir, manifest_hashes)


def run_head_search(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    """Search select, choose among top K on dev, and expose one head on test."""
    select_examples, select_exclusions = load_manifest_examples(cfg, tokenizer, "select")
    dev_examples, dev_exclusions = load_manifest_examples(cfg, tokenizer, "dev")
    test_examples, test_exclusions = load_manifest_examples(cfg, tokenizer, "test")
    checkpoints = run_dir / "checkpoints"
    select_rows = score_over_seeds(
        model, tokenizer, select_examples, cfg, role="select", checkpoint_dir=checkpoints,
        stage="select-all-heads"
    )
    dev_rows = score_over_seeds(
        model, tokenizer, dev_examples, cfg, role="dev", checkpoint_dir=checkpoints,
        stage="dev-all-heads"
    )
    select_scores = aggregate_head_scores(select_rows)
    dev_scores = aggregate_head_scores(dev_rows)
    select_scores.to_csv(run_dir / "select_all_head_scores.csv", index=False)
    dev_scores.to_csv(run_dir / "dev_all_head_scores.csv", index=False)
    select_rows.to_parquet(run_dir / "select_instances.parquet", index=False)
    dev_rows.to_parquet(run_dir / "dev_instances.parquet", index=False)
    relation_bundle = derive_relation_selection_bundle(
        run_dir,
        run_dir / "relation-selection",
        require_complete=False,
        allow_source_output=True,
        allow_existing=True,
    )
    install_primary_aliases(run_dir, relation_bundle)
    lock = relation_bundle.primary_lock
    if lock is None:
        raise ArtifactError("primary object_to_verb relation has insufficient selection evidence")
    fixed_offset = lock.frozen_settings.get("fixed_offset")

    test_rows = score_over_seeds(
        model,
        tokenizer,
        test_examples,
        cfg,
        role="test",
        heads={(lock.layer, lock.head)},
        checkpoint_dir=checkpoints,
        stage="test-locked-head",
    )
    if test_rows[["layer", "head"]].drop_duplicates().shape[0] != 1:
        raise ArtifactError("locked test must expose exactly one head")
    exclusions = pd.concat([select_exclusions, dev_exclusions, test_exclusions], ignore_index=True)
    write_frames(run_dir, raw=test_rows, exclusions=exclusions)
    metrics = locked_metrics(test_rows, fixed_offset)
    permutation_table = pd.read_csv(relation_bundle.output_dir / "permutation_results.csv")
    primary_permutation = permutation_table[
        permutation_table["relation"] == cfg.experiment.scoring.primary_relation
    ].iloc[0]
    permutation = {
        "observed_dev_accuracy": float(primary_permutation["observed_dev_accuracy"]),
        "p_value": float(primary_permutation["raw_p_value"]),
        "n_permutations": int(primary_permutation["n_permutations"]),
        "null_mean": float(primary_permutation["null_mean"]),
        "null_std": float(primary_permutation["null_std"]),
        "family": "primary_separate",
    }
    metrics["selection_permutation_p"] = permutation["p_value"]
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    per_seed_metrics(test_rows).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    structural_slices(test_rows).to_csv(run_dir / "structural_slices.csv", index=False)
    atomic_json(run_dir / "selection_permutation.json", permutation)
    return {
        "selection_lock": asdict(lock),
        "n_select_sentences": len(select_examples),
        "n_dev_sentences": len(dev_examples),
        "n_test_sentences": len(test_examples),
        "n_test_instances": int(test_rows["instance_id"].nunique()),
        "test_heads_exposed": 1,
        "relation_selection_bundle": "relation-selection/relation_selection_bundle.json",
        "secondary_relations": "predefined_selected_not_tested",
    }


def run_locked_transfer(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    source_lock: SelectionLock,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    rows = score_over_seeds(
        model,
        tokenizer,
        examples,
        cfg,
        role="test",
        heads={(source_lock.layer, source_lock.head)},
        normalized_progress=float(source_lock.frozen_settings["selection_progress"]),
        checkpoint_dir=run_dir / "checkpoints",
        stage="external-test-locked-head",
    )
    source_lock.write_once(run_dir / "selection_lock.json")
    write_frames(run_dir, raw=rows, exclusions=exclusions)
    locked_metrics(rows, source_lock.frozen_settings.get("fixed_offset")).to_csv(
        run_dir / "metrics.csv", index=False
    )
    per_seed_metrics(rows).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    structural_slices(rows).to_csv(run_dir / "structural_slices.csv", index=False)
    return {
        "source_selection_dataset": source_lock.dataset_id,
        "source_selection_hash": selection_lock_hash(source_lock),
        "n_test_sentences": len(examples),
        "n_test_instances": int(rows["instance_id"].nunique()),
        "test_heads_exposed": int(rows[["layer", "head"]].drop_duplicates().shape[0]),
    }

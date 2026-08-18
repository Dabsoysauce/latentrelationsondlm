"""Select heads on EWT select/dev, test once, and transfer the frozen head."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import (
    ArtifactError,
    atomic_json,
    selection_source_hash,
)
from ..config import RunConfig
from ..data import load_manifest_examples
from ..permutation import selection_aware_permutation
from ..relation_selection import (
    MINIMUM_DENOMINATOR,
    PRIMARY_RELATION,
    SECONDARY_RELATIONS,
    RelationLockSet,
    derive_relation_selection_bundle,
    filter_relation_locked_rows,
    install_primary_aliases,
    load_relation_locks,
    write_resolved_lock_manifest,
)
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
    source_locks: RelationLockSet | None = None,
) -> dict[str, Any]:
    if cfg.track == "external_treebank_transfer":
        if source_locks is None:
            raise ArtifactError("external transfer requires an EWT selection lock")
        return run_locked_transfer(model, tokenizer, cfg, run_dir, source_locks)
    return run_head_search(model, tokenizer, cfg, run_dir, manifest_hashes)


def run_head_search(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    """Select six relation heads, score the test grid, then apply each matching lock."""
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
    locks = load_relation_locks(relation_bundle.output_dir, cfg)
    fixed_offsets = {
        relation: lock.frozen_settings.get("fixed_offset")
        for relation, lock in locks.locks.items()
    }

    test_all_rows = score_over_seeds(
        model,
        tokenizer,
        test_examples,
        cfg,
        role="test",
        checkpoint_dir=checkpoints,
        stage="test-all-heads-selection-permutation",
    )
    test_rows = filter_relation_locked_rows(test_all_rows, locks)
    _validate_relation_test_rows(test_rows, locks)
    exclusions = pd.concat([select_exclusions, dev_exclusions, test_exclusions], ignore_index=True)
    write_frames(run_dir, raw=test_rows, exclusions=exclusions)
    metrics = locked_metrics(test_rows, fixed_offsets)
    permutation_table, primary_permutation = _run_permutations(
        select_rows,
        dev_rows,
        test_all_rows,
        cfg,
        run_dir,
        locks,
    )
    metrics = metrics.merge(
        permutation_table[["relation", "raw_p_value", "holm_adjusted_p_value"]],
        on="relation",
        how="left",
        validate="many_to_one",
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    per_seed_metrics(test_rows).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    structural_slices(test_rows).to_csv(run_dir / "structural_slices.csv", index=False)
    atomic_json(run_dir / "selection_permutation.json", primary_permutation)
    write_resolved_lock_manifest(run_dir, locks)
    return {
        "selection_lock": asdict(locks.resolve(PRIMARY_RELATION)),
        "n_select_sentences": len(select_examples),
        "n_dev_sentences": len(dev_examples),
        "n_test_sentences": len(test_examples),
        "n_test_instances": int(test_rows["instance_id"].nunique()),
        "test_heads_exposed": len(locks.heads),
        "test_relation_locks_applied": len(locks.locks),
        "relation_selection_bundle": "relation-selection/relation_selection_bundle.json",
        "secondary_relations": "predefined_and_evaluated_with_their_own_frozen_heads",
        "permutation_null": "within_instance_valid_receiver_full_select_dev_test_protocol",
    }


def run_locked_transfer(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    source_locks: RelationLockSet,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    progress_values = {
        float(lock.frozen_settings["selection_progress"])
        for lock in source_locks.locks.values()
    }
    if len(progress_values) != 1:
        raise ArtifactError("relation locks disagree on frozen selection progress")
    rows = score_over_seeds(
        model,
        tokenizer,
        examples,
        cfg,
        role="test",
        heads=source_locks.heads,
        normalized_progress=progress_values.pop(),
        checkpoint_dir=run_dir / "checkpoints",
        stage="external-test-locked-head",
    )
    rows = filter_relation_locked_rows(rows, source_locks)
    _validate_relation_test_rows(rows, source_locks)
    write_resolved_lock_manifest(run_dir, source_locks)
    write_frames(run_dir, raw=rows, exclusions=exclusions)
    offsets = {
        relation: lock.frozen_settings.get("fixed_offset")
        for relation, lock in source_locks.locks.items()
    }
    locked_metrics(rows, offsets).to_csv(
        run_dir / "metrics.csv", index=False
    )
    per_seed_metrics(rows).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    structural_slices(rows).to_csv(run_dir / "structural_slices.csv", index=False)
    return {
        "source_selection_dataset": "ewt",
        "source_selection_hash": selection_source_hash(source_locks.source),
        "source_selection_kind": source_locks.source_kind,
        "n_test_sentences": len(examples),
        "n_test_instances": int(rows["instance_id"].nunique()),
        "test_heads_exposed": int(rows[["layer", "head"]].drop_duplicates().shape[0]),
        "test_relation_locks_applied": len(source_locks.locks),
    }


def _run_permutations(
    select_rows: pd.DataFrame,
    dev_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    cfg: RunConfig,
    run_dir: Path,
    locks: RelationLockSet,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    scientific_hash = metadata["scientific_config_hash"]
    result_dir = run_dir / "permutations"
    checkpoint_dir = run_dir / "permutation-checkpoints"
    rows = []
    results = {}
    for relation in locks.locks:
        result = selection_aware_permutation(
            select_rows,
            dev_rows,
            test_rows,
            relation=relation,
            top_k=cfg.experiment.scoring.top_k,
            n_permutations=1000,
            seed=42,
            scientific_config_hash=scientific_hash,
            minimum_denominator=MINIMUM_DENOMINATOR,
            checkpoint_path=checkpoint_dir / f"{relation}.json",
            resume=cfg.runtime.resume,
        )
        if result["completion_status"] != "complete":
            raise ArtifactError(f"permutation run did not complete for relation {relation!r}")
        lock = locks.resolve(relation)
        if (result["observed_selected_layer"], result["observed_selected_head"]) != (
            lock.layer,
            lock.head,
        ):
            raise ArtifactError(
                f"permutation observed protocol disagrees with frozen lock for {relation!r}"
            )
        result["family"] = (
            "primary_separate" if relation == PRIMARY_RELATION else "five_secondaries"
        )
        atomic_json(result_dir / f"{relation}.json", result)
        results[relation] = result
        rows.append(
            {
                "relation": relation,
                "family": result["family"],
                "observed_test_accuracy": result["observed_test_accuracy"],
                "raw_p_value": result["p_value"],
                "holm_adjusted_p_value": None,
                "n_permutations": result["n_permutations"],
                "null_mean": result["null_mean"],
                "null_std": result["null_std"],
                "null_definition": result["null_definition"],
                "checkpoint": f"permutation-checkpoints/{relation}.json",
            }
        )
    _apply_secondary_holm(rows)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "selection_permutation_results.csv", index=False)
    primary = results[PRIMARY_RELATION]
    primary["holm_adjusted_p_value"] = None
    return table, primary


def _apply_secondary_holm(rows: list[dict[str, Any]]) -> None:
    secondaries = [row for row in rows if row["relation"] in SECONDARY_RELATIONS]
    secondaries.sort(
        key=lambda row: (row["raw_p_value"], SECONDARY_RELATIONS.index(row["relation"]))
    )
    running = 0.0
    family_size = len(SECONDARY_RELATIONS)
    for rank, row in enumerate(secondaries, start=1):
        running = max(
            running,
            min(1.0, (family_size - rank + 1) * float(row["raw_p_value"])),
        )
        row["holm_adjusted_p_value"] = running


def _validate_relation_test_rows(rows: pd.DataFrame, locks: RelationLockSet) -> None:
    if not set(rows.get("relation", pd.Series(dtype=str))).issubset(locks.locks):
        raise ArtifactError("locked test rows contain a relation without a matching lock")
    for relation, group in rows.groupby("relation", observed=True):
        lock = locks.resolve(str(relation))
        heads = set(group[["layer", "head"]].itertuples(index=False, name=None))
        if heads != {(lock.layer, lock.head)}:
            raise ArtifactError(f"test rows do not match the frozen lock for {relation!r}")

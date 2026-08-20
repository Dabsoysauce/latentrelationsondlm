"""Select heads on EWT select/dev, test once, and transfer the frozen head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import (
    ArtifactError,
    selection_source_hash,
)
from ..config import RunConfig
from ..data import load_manifest_examples
from ..head_search_recovery import (
    finalize_head_search_artifacts,
    score_missing_test_grid,
)
from ..relation_selection import (
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
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cfg.track == "external_treebank_transfer":
        if source_locks is None:
            raise ArtifactError("external transfer requires an EWT selection lock")
        return run_locked_transfer(model, tokenizer, cfg, run_dir, source_locks)
    return run_head_search(
        model,
        tokenizer,
        cfg,
        run_dir,
        manifest_hashes,
        model_metadata=model_metadata,
    )


def run_head_search(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    manifest_hashes: dict[str, str],
    *,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select six relation heads, score the test grid, then apply each matching lock."""
    select_examples, select_exclusions = load_manifest_examples(cfg, tokenizer, "select")
    dev_examples, dev_exclusions = load_manifest_examples(cfg, tokenizer, "dev")
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

    _recovery_report, test_rows, test_exclusions = score_missing_test_grid(
        model,
        tokenizer,
        cfg,
        run_dir,
        model_metadata=model_metadata,
        collect_locked_rows=True,
        reuse_existing_locked=False,
    )
    _validate_relation_test_rows(test_rows, locks)
    exclusions = pd.concat([select_exclusions, dev_exclusions, test_exclusions], ignore_index=True)
    write_frames(run_dir, raw=test_rows, exclusions=exclusions)
    del select_rows, dev_rows, test_rows
    return finalize_head_search_artifacts(cfg, run_dir)


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


def _validate_relation_test_rows(rows: pd.DataFrame, locks: RelationLockSet) -> None:
    if not set(rows.get("relation", pd.Series(dtype=str))).issubset(locks.locks):
        raise ArtifactError("locked test rows contain a relation without a matching lock")
    for relation, group in rows.groupby("relation", observed=True):
        lock = locks.resolve(str(relation))
        heads = set(group[["layer", "head"]].itertuples(index=False, name=None))
        if heads != {(lock.layer, lock.head)}:
            raise ArtifactError(f"test rows do not match the frozen lock for {relation!r}")

"""Track one EWT-locked attention head across the frozen masking schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import RunConfig
from ..data import load_manifest_examples
from ..relation_selection import (
    RelationLockSet,
    filter_relation_locked_rows,
    write_resolved_lock_manifest,
)
from .shared import score_over_seeds, write_frames


def aggregate_curve(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain seed-level and pooled summaries without collapsing timesteps."""
    group = [
        "seed",
        "treebank",
        "relation",
        "layer",
        "head",
        "timestep",
        "normalized_progress",
        "visibility",
    ]
    per_seed = raw.groupby(group, as_index=False).agg(
        accuracy=("correct", "mean"), n_instances=("correct", "size")
    )
    pooled = per_seed.groupby(group[1:], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_instances=("n_instances", "sum"),
        n_seeds=("seed", "nunique"),
    )
    return per_seed, pooled


def run(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    source_locks: RelationLockSet,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    frames = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            frame = score_over_seeds(
                model,
                tokenizer,
                examples,
                cfg,
                role="test",
                heads=source_locks.heads,
                normalized_progress=progress,
                checkpoint_dir=run_dir / "checkpoints",
                stage="time-curve",
                seeds=[seed],
            )
            frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw = filter_relation_locked_rows(all_rows, source_locks)
    write_resolved_lock_manifest(run_dir, source_locks)
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    per_seed, metrics = aggregate_curve(raw)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return {
        "n_rows": len(raw),
        "n_sentences": len(examples),
        "test_heads_exposed": len(source_locks.heads),
        "test_relation_locks_applied": len(source_locks.locks),
    }

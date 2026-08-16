"""Track one EWT-locked attention head across the frozen masking schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts import SelectionLock
from ..config import RunConfig
from ..data import load_manifest_examples
from .shared import atomic_parquet, score_attention_heads, write_frames


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
    source_lock: SelectionLock,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    frames = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            checkpoint = run_dir / "checkpoints" / (
                f"time-curve-seed{seed}-p{progress:.6f}-"
                f"l{source_lock.layer}h{source_lock.head}.parquet"
            )
            if checkpoint.exists():
                frame = pd.read_parquet(checkpoint)
            else:
                frame = score_attention_heads(
                    model,
                    tokenizer,
                    examples,
                    cfg,
                    role="test",
                    heads={(source_lock.layer, source_lock.head)},
                    normalized_progress=progress,
                    seed=seed,
                )
                atomic_parquet(checkpoint, frame)
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    source_lock.write_once(run_dir / "selection_lock.json")
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    per_seed, metrics = aggregate_curve(raw)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return {"n_rows": len(raw), "n_sentences": len(examples), "test_heads_exposed": 1}

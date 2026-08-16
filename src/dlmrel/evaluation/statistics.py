"""Statistical helpers that are used by the active protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sentence_clustered_bootstrap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    sentence_col: str = "sentence_id",
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Resample sentences while keeping their relation and seed rows together."""
    if frame.empty:
        return float("nan"), float("nan")
    clusters = [group[value_col].to_numpy(float) for _, group in frame.groupby(sentence_col)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        chosen = rng.integers(0, len(clusters), len(clusters))
        estimates[index] = np.concatenate([clusters[item] for item in chosen]).mean()
    alpha = (1 - ci) / 2
    return tuple(float(value) for value in np.quantile(estimates, [alpha, 1 - alpha]))


def hierarchical_seed_summary(frame: pd.DataFrame, *, value_col: str = "correct") -> pd.DataFrame:
    """Treat seeds as repeated runs rather than independent observations."""
    per_sentence_seed = frame.groupby(["sentence_id", "seed"], as_index=False)[value_col].mean()
    per_seed = per_sentence_seed.groupby("seed", as_index=False)[value_col].mean()
    return pd.DataFrame(
        [
            {
                "mean": float(per_seed[value_col].mean()),
                "std_across_seeds": float(per_seed[value_col].std()),
                "n_seeds": int(per_seed["seed"].nunique()),
                "n_sentences": int(per_sentence_seed["sentence_id"].nunique()),
            }
        ]
    )


def adjust_pvalues(pvalues: list[float], method: str = "holm") -> list[float]:
    """Apply Holm correction, with BH available only as a sensitivity check."""
    values = np.asarray(pvalues, dtype=float)
    count = len(values)
    if method == "holm":
        order = np.argsort(values)
        adjusted = np.empty(count)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (count - rank) * values[index])
            adjusted[index] = min(1.0, running)
        return adjusted.tolist()
    if method in {"bh", "benjamini-hochberg"}:
        order = np.argsort(values)[::-1]
        adjusted = np.empty(count)
        running = 1.0
        for reverse_rank, index in enumerate(order):
            rank = count - reverse_rank
            running = min(running, count * values[index] / rank)
            adjusted[index] = min(1.0, running)
        return adjusted.tolist()
    raise ValueError("method must be holm or benjamini-hochberg")

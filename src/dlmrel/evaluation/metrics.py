"""The fixed-offset null model.

This is the baseline the previous paper lacked, and the reason four of its six
relations did not survive re-analysis. Uniform chance (~6%) is the wrong bar: a
head that always points a fixed number of positions away from itself solves any
relation whose endpoints sit at a near-constant distance, without representing
anything syntactic.

The null is fit exactly like a head is selected -- best offset chosen on the
`select` split, accuracy reported on held-out data -- so the comparison is
protocol-matched.

Empirically, on UD-EWT this baseline reaches 0.75 on adjective->noun and 0.60
on determiner->noun, which is where DiffuGPT-S's secondary "relation heads"
also land. Only object->verb (true offset spread across -1..-4) leaves it far
behind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OffsetNull:
    """A `receiver = attender + k` predictor with k fit on held-in data."""

    relation: str
    k: int
    fit_accuracy: float
    n_fit: int

    def predict(self, anchor: int) -> int:
        return anchor + self.k


def _anchors_and_targets(df: pd.DataFrame, attender_token: str = "last") -> tuple[np.ndarray, list[set[int]]]:
    """Reduce each instance to (attender anchor token, receiver token set).

    The anchor must be the same token the attention row is taken from,
    otherwise the null is not measuring the same thing as the head.
    """
    spans_a = df["attender_span"].apply(_as_list)
    spans_r = df["receiver_span"].apply(_as_list)
    pick = (lambda s: s[-1]) if attender_token == "last" else (lambda s: s[0])
    return np.array([pick(s) for s in spans_a]), [set(s) for s in spans_r]


def _as_list(value) -> list[int]:
    """Span columns survive a CSV round trip as strings; accept either form."""
    if isinstance(value, str):
        import ast

        return list(ast.literal_eval(value))
    return list(value)


def offset_accuracy(df: pd.DataFrame, k: int, attender_token: str = "last") -> float:
    anchors, targets = _anchors_and_targets(df, attender_token)
    if len(anchors) == 0:
        return float("nan")
    return float(np.mean([(a + k) in t for a, t in zip(anchors, targets, strict=False)]))


def fit_offset_null(
    df: pd.DataFrame,
    relation: str,
    offset_range: tuple[int, int] = (-15, 15),
    attender_token: str = "last",
) -> OffsetNull:
    """Choose the single offset that best explains a relation on this split."""
    sub = df[df["relation"] == relation]
    if len(sub) == 0:
        raise ValueError(f"no instances for relation {relation!r}")

    lo, hi = offset_range
    candidates = [k for k in range(lo, hi + 1) if k != 0]
    accuracies = {k: offset_accuracy(sub, k, attender_token) for k in candidates}
    best_k = max(accuracies, key=accuracies.__getitem__)
    return OffsetNull(
        relation=relation,
        k=best_k,
        fit_accuracy=accuracies[best_k],
        n_fit=len(sub),
    )


def bootstrap_ci(
    correct: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI over a boolean correctness vector."""
    correct = np.asarray(correct, dtype=float)
    if correct.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, correct.size, size=(n_boot, correct.size))
    means = correct[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def offset_correctness(df: pd.DataFrame, k: int, attender_token: str = "last") -> np.ndarray:
    anchors, targets = _anchors_and_targets(df, attender_token)
    return np.array([(a + k) in t for a, t in zip(anchors, targets, strict=False)], dtype=float)


def build_null_table(
    select_df: pd.DataFrame,
    test_df: pd.DataFrame,
    relations: list[str],
    offset_range: tuple[int, int] = (-15, 15),
    attender_token: str = "last",
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Fit on `select`, report on `test`, with a bootstrap CI on the estimate."""
    rows = []
    for relation in relations:
        null = fit_offset_null(select_df, relation, offset_range, attender_token)
        sub = test_df[test_df["relation"] == relation]
        correct = offset_correctness(sub, null.k, attender_token)
        lo, hi = bootstrap_ci(correct, n_boot, ci)
        rows.append(
            {
                "relation": relation,
                "k": null.k,
                "null_fit_acc": null.fit_accuracy,
                "n_fit": null.n_fit,
                "null_test_acc": (float(correct.mean()) if correct.size else float("nan")),
                "null_ci_lo": lo,
                "null_ci_hi": hi,
                "n_test": int(correct.size),
            }
        )
    return pd.DataFrame(rows)


def offset_distribution(df: pd.DataFrame, relation: str) -> pd.Series:
    """How concentrated is a relation's true offset?

    A relation whose distance is nearly constant is trivially solvable by a
    positional head; a relation with a spread distribution is not. This is the
    diagnostic that explains *why* object->verb survives the null while
    determiner->noun does not.
    """
    sub = df[df["relation"] == relation]
    return (
        sub["word_distance"].value_counts(normalize=True).sort_index()
        if "word_distance" in sub
        else pd.Series(dtype=float)
    )

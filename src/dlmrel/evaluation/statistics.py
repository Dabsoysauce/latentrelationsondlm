"""Selection diagnostics and head-versus-null comparisons.

Searching every head of a model and reporting the winner is a multiple-
comparisons problem: with 144 heads (DiffuGPT-S) or 1024 (DiffuLLaMA-7B), some
head beats any fixed baseline by luck. Holding out *sentences* does not fix
this, because the same head is selected and reported.

Two diagnostics settle it without a correction factor, and both are cheap:

  `n_heads_above_null`
      How many heads clear the baseline at all. On the previous data,
      object->verb put 29/144 heads above the null while the four weak
      relations put 1-3/144 above it -- the difference between a represented
      relation and a lucky draw.

  `selection_rank_correlation`
      Spearman rho between select-split and test-split accuracy across all
      heads. Near 1.0 means the ranking transfers and the winner was not noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def n_heads_above_null(scores: pd.DataFrame, null_acc: float, column: str) -> int:
    """Count heads whose accuracy exceeds the fixed-offset null."""
    return int((scores[column] > null_acc).sum())


def selection_rank_correlation(scores: pd.DataFrame, select_col: str, test_col: str) -> tuple[float, float]:
    """Spearman rho and p-value between selection and reporting splits."""
    if len(scores) < 3:
        return (float("nan"), float("nan"))
    rho, p = sps.spearmanr(scores[select_col], scores[test_col])
    return float(rho), float(p)


def wilson_ci(n_correct: int, n_total: int, ci: float = 0.95) -> tuple[float, float]:
    """Wilson score interval -- well behaved for proportions near 0 or 1."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1.0 - (1.0 - ci) / 2.0)
    p = n_correct / n_total
    denom = 1.0 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (float(centre - half), float(centre + half))


def build_head_vs_null_table(
    scores: pd.DataFrame,
    null_table: pd.DataFrame,
    select_col: str = "accuracy_select",
    test_col: str = "accuracy_test",
    n_correct_col: str = "n_correct_test",
    n_total_col: str = "n_total_test",
    ci: float = 0.95,
) -> pd.DataFrame:
    """The paper's headline table: best head per relation against the null.

    `verdict` requires the head's interval to clear the *null's upper bound*,
    not the null's point estimate. The null is an estimate too, and comparing
    an interval to a point ignores its uncertainty: on UD-EWT that let subject
    determiner->noun read "survives" on a margin of 0.0009, which is a rounding
    artifact rather than a finding. Non-overlapping intervals is the weaker,
    honest claim the data actually supports.
    """
    nulls = null_table.set_index("relation")
    rows = []

    for relation, group in scores.groupby("relation"):
        if relation not in nulls.index:
            continue
        null_acc = float(nulls.loc[relation, "null_test_acc"])
        # Fall back to the point estimate when the null table predates CIs.
        null_hi = float(nulls.loc[relation].get("null_ci_hi", null_acc))
        k = int(nulls.loc[relation, "k"])

        # Select on the selection split only; report the winner's test score.
        best = group.sort_values(select_col, ascending=False).iloc[0]
        n_c = int(best[n_correct_col])
        n_t = int(best[n_total_col])
        lo, hi = wilson_ci(n_c, n_t, ci)
        rho, _ = selection_rank_correlation(group, select_col, test_col)

        rows.append(
            {
                "relation": relation,
                "layer": int(best["layer"]),
                "head": int(best["head"]),
                "head_select_acc": float(best[select_col]),
                "head_test_acc": float(best[test_col]),
                "head_ci_lo": lo,
                "head_ci_hi": hi,
                "null_k": k,
                "null_test_acc": null_acc,
                "null_ci_hi": null_hi,
                "delta": float(best[test_col]) - null_acc,
                "margin": lo - null_hi,
                "n_heads_above_null": n_heads_above_null(group, null_acc, test_col),
                "n_heads_total": len(group),
                "selection_rho": rho,
                "n_test": n_t,
                "verdict": "survives" if lo > null_hi else "not distinguishable",
            }
        )

    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def positional_selectivity(scores: pd.DataFrame, test_col: str = "accuracy_test") -> pd.DataFrame:
    """Profile every head across all relations to separate the two head types.

    A *positional* head scores high on every short-distance relation and near
    zero on long-distance ones -- it implements a fixed offset. A *relational*
    head scores high on one relation and near zero elsewhere, which no offset
    can produce. On DiffuLLaMA-7B this cleanly separates L18 H10 (positional,
    0.86-0.94 adjacent / 0.14 object->verb) from L3 H11 (relational, 0.88
    object->verb / 0.00-0.02 adjacent).
    """
    wide = scores.pivot_table(index=["layer", "head"], columns="relation", values=test_col)
    adjacent = [c for c in wide.columns if c.endswith("_to_noun")]
    distant = [c for c in wide.columns if c.endswith("_to_verb")]

    out = wide.copy()
    out["adjacent_mean"] = wide[adjacent].mean(axis=1) if adjacent else np.nan
    out["distant_mean"] = wide[distant].mean(axis=1) if distant else np.nan
    out["max_relation"] = wide.max(axis=1)
    out["second_relation"] = wide.apply(
        lambda r: r.nlargest(2).iloc[-1] if r.notna().sum() >= 2 else np.nan, axis=1
    )
    # High when a head serves one relation and nothing else.
    out["selectivity"] = out["max_relation"] - out["second_relation"]
    # Positive when a head prefers adjacency, which is the positional signature.
    out["adjacency_bias"] = out["adjacent_mean"] - out["distant_mean"]
    return out.reset_index()


def stratify_by_distance(
    correctness: pd.DataFrame,
    bins: list[int],
    value_col: str = "correct",
    distance_col: str = "word_distance",
) -> pd.DataFrame:
    """Accuracy as a function of dependency distance.

    This is the experiment that turns the positional/relational distinction
    from an anecdote about two heads into a curve: a fixed-offset head's
    accuracy collapses as soon as the distance leaves its offset, a relational
    head's does not.
    """
    df = correctness.copy()
    df["distance_bin"] = pd.cut(df[distance_col].abs(), bins=[0] + list(bins), include_lowest=True)
    grouped = df.groupby("distance_bin", observed=True)[value_col]
    out = grouped.agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n"})
    return out.reset_index()


def sentence_clustered_bootstrap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    sentence_col: str = "sentence_id",
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Resample sentences, retaining all dependent relation/seed rows."""
    if frame.empty:
        return float("nan"), float("nan")
    clusters = [group[value_col].to_numpy(float) for _, group in frame.groupby(sentence_col)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        chosen = rng.integers(0, len(clusters), len(clusters))
        estimates[index] = np.concatenate([clusters[item] for item in chosen]).mean()
    alpha = (1 - ci) / 2
    return tuple(float(x) for x in np.quantile(estimates, [alpha, 1 - alpha]))


def hierarchical_seed_summary(frame: pd.DataFrame, *, value_col: str = "correct") -> pd.DataFrame:
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

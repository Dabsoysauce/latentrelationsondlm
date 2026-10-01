"""Sentence-clustered paired inference; seeds are repeated measures, not extra N."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .statistics import adjust_pvalues, sentence_clustered_bootstrap

GROUPS = ["head_relation", "relation", "control", "alpha", "window", "start", "stop", "is_anchor"]
METRICS = [
    "governor_exact",
    "dependent_exact",
    "endpoint_exact",
    "governor_logprob",
    "governor_logit",
    "governor_probability",
    "target_accuracy",
    "non_target_accuracy",
    "overall_accuracy",
    "sequence_exact",
]


def paired_summary(raw, *, n_boot=2000, seed=42):
    summaries, seeds = [], []
    for key, group in raw.groupby(GROUPS, dropna=False):
        identity = dict(zip(GROUPS, key, strict=True))
        for metric in METRICS + ["specificity_degradation"]:
            if metric == "specificity_degradation":
                frame = group[["sentence_id", "seed", "primary_anchor", metric]].dropna().copy()
                frame["b"], frame["i"] = 0.0, frame[metric]
            else:
                b, i = f"baseline_{metric}", f"intervention_{metric}"
                frame = (
                    group[["sentence_id", "seed", "primary_anchor", b, i]]
                    .dropna()
                    .rename(columns={b: "b", i: "i"})
                )
            if frame.empty:
                continue
            frame["delta"] = frame.i - frame.b
            low, high = sentence_clustered_bootstrap(frame, value_col="delta", n_boot=n_boot, seed=seed)
            cluster = frame.groupby("sentence_id").delta.mean().to_numpy()
            cluster_sums = frame.groupby("sentence_id").delta.sum().to_numpy()
            rng = np.random.default_rng(seed)
            observed = abs(frame.delta.mean())
            null = np.array(
                [
                    abs((cluster_sums * rng.choice([-1, 1], len(cluster))).sum() / len(frame))
                    for _ in range(n_boot)
                ]
            )
            p = (1 + int((null >= observed - 1e-15).sum())) / (n_boot + 1)
            sd = cluster.std(ddof=1) if len(cluster) > 1 else np.nan
            per_seed = frame.groupby("seed").delta.mean()
            summaries.append(
                {
                    **identity,
                    "metric": metric,
                    "n_sentences": len(cluster),
                    "n_pairs": len(frame),
                    "n_seeds": len(per_seed),
                    "baseline_mean": frame.b.mean(),
                    "intervention_mean": frame.i.mean(),
                    "difference": frame.delta.mean(),
                    "relative_difference": frame.delta.mean() / frame.b.mean()
                    if frame.b.mean() != 0
                    and not identity["control"].startswith("target_minus_")
                    and metric not in {"governor_logit", "governor_logprob", "specificity_degradation"}
                    else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p,
                    "test": "sentence_cluster_sign_flip_two_sided",
                    "effect_dz": cluster.mean() / sd if sd > 0 else np.nan,
                    "seed_delta_mean": per_seed.mean(),
                    "seed_delta_std": per_seed.std(),
                    "baseline_seed_std": frame.groupby("seed").b.mean().std(),
                    "intervention_seed_std": frame.groupby("seed").i.mean().std(),
                }
            )
            for run_seed, current in frame.groupby("seed"):
                result = {
                    **identity,
                    "metric": metric,
                    "seed": run_seed,
                    "n": len(current),
                    "baseline_mean": current.b.mean(),
                    "intervention_mean": current.i.mean(),
                    "difference": current.delta.mean(),
                }
                # Anchor is fixed before outcomes, at most one pair per sentence.
                binary = metric in {"governor_exact", "dependent_exact", "endpoint_exact", "sequence_exact"}
                if identity["is_anchor"] and binary and not identity["control"].startswith("target_minus_"):
                    current = current[current.primary_anchor]
                    if current.sentence_id.duplicated().any():
                        raise ValueError("McNemar requires one anchor per sentence per seed")
                    harmed = int(((current.b == 1) & (current.i == 0)).sum())
                    helped = int(((current.b == 0) & (current.i == 1)).sum())
                    result.update(
                        mcnemar_n=len(current),
                        harmed=harmed,
                        helped=helped,
                        mcnemar_p=binomtest(harmed, harmed + helped).pvalue if harmed + helped else 1.0,
                        test="exact_McNemar_two_sided",
                    )
                seeds.append(result)
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["p_holm"] = adjust_pvalues(summary.p_value.tolist())
    seed_frame = pd.DataFrame(seeds)
    if "mcnemar_p" in seed_frame:
        for _, current in seed_frame[seed_frame.mcnemar_p.notna()].groupby("seed"):
            seed_frame.loc[current.index, "mcnemar_p_holm"] = adjust_pvalues(current.mcnemar_p.tolist())
    return summary, seed_frame


def matched_control_contrasts(raw):
    """Within-example target-head effect minus each control-head effect."""
    keys = [
        "sentence_id",
        "instance_id",
        "anchor_instance_id",
        "seed",
        "head_relation",
        "relation",
        "alpha",
        "window",
        "start",
        "stop",
        "is_anchor",
    ]
    rows = []
    target = raw[raw.control == "target"]
    for control, frame in raw[raw.control != "target"].groupby("control"):
        joined = target.merge(frame, on=keys, suffixes=("_target", "_control"), validate="one_to_one")
        for _, row in joined.iterrows():
            record = {k: row[k] for k in keys}
            record["control"] = f"target_minus_{control}"
            record["primary_anchor"] = row.primary_anchor_target
            for metric in METRICS:
                record[f"baseline_{metric}"] = (
                    row[f"intervention_{metric}_control"] - row[f"baseline_{metric}_control"]
                )
                record[f"intervention_{metric}"] = (
                    row[f"intervention_{metric}_target"] - row[f"baseline_{metric}_target"]
                )
            record["specificity_degradation"] = (
                row.specificity_degradation_target - row.specificity_degradation_control
            )
            # Effect differences are continuous even when the underlying outcome is binary.
            rows.append(record)
    return pd.DataFrame(rows)

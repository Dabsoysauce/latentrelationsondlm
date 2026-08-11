"""Scoring every attention head against gold relations.

Two measurements:

`score_split`
    At the final, fully-revealed frame, does a head's attention row from the
    attender land on the correct receiver? Run over every head and relation.

`masked_state_curve`
    The same question across diffusion time, split by whether the relation's
    endpoints are still masked. This is the claim that structure forms *before*
    tokens resolve, and it is the measurement most in need of care: the
    per-instance correctness vectors are retained so baselines can be computed
    on exactly the instances that entered the average, not on the full split.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import RELATION_NAMES, DiffusionConfig
from .diffusion import attentions_at_time, receiver_predictions
from .relations import Example


def score_split(
    model,
    tokenizer,
    examples: list[Example],
    cfg: DiffusionConfig,
    split_name: str = "",
    log_every: int = 100,
) -> pd.DataFrame:
    """Accuracy of every (layer, head) on every relation, final frame."""
    final_time = cfg.steps - 1
    correct: dict[str, np.ndarray] = {}
    totals: dict[str, int] = defaultdict(int)
    n_layers = n_heads = None

    for i, example in enumerate(examples):
        attentions, state = attentions_at_time(
            model,
            tokenizer,
            example.text,
            diffusion_time=final_time,
            steps=cfg.steps,
            seed=cfg.seed,
            include_bos=cfg.include_bos,
        )

        if n_layers is None:
            n_layers, n_heads = len(attentions), attentions[0].shape[1]
            for name in RELATION_NAMES:
                correct[name] = np.zeros((n_layers, n_heads), dtype=np.float64)

        seq_len = len(state.tokens)
        for inst in example.relations:
            a_span = [t for t in inst.attender_span if t < seq_len]
            r_span = {t for t in inst.receiver_span if t < seq_len}
            if not a_span or not r_span:
                continue

            totals[inst.relation] += 1
            for layer in range(n_layers):
                preds = receiver_predictions(
                    attentions,
                    layer,
                    a_span,
                    cfg.attender_token,
                    cfg.exclude_bos,
                    cfg.exclude_self,
                )
                correct[inst.relation][layer] += np.isin(preds, list(r_span))

        if log_every and (i + 1) % log_every == 0:
            print(f"[scoring] {split_name}: {i + 1}/{len(examples)} sentences")

    rows = []
    for relation in RELATION_NAMES:
        n = totals.get(relation, 0)
        if n == 0 or relation not in correct:
            continue
        acc = correct[relation] / n
        for layer in range(acc.shape[0]):
            for head in range(acc.shape[1]):
                rows.append(
                    {
                        "relation": relation,
                        "layer": layer,
                        "head": head,
                        "accuracy": float(acc[layer, head]),
                        "n_correct": int(correct[relation][layer, head]),
                        "n_total": n,
                    }
                )

    print(f"[scoring] {split_name}: totals {dict(totals)}")
    return pd.DataFrame(rows)


def merge_splits(
    select: pd.DataFrame, test: pd.DataFrame, dev: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join per-split head scores into the wide table the analysis expects."""
    keys = ["relation", "layer", "head"]
    merged = select.merge(test, on=keys, suffixes=("_select", "_test"))
    if dev is not None:
        dev = dev.rename(
            columns={
                "accuracy": "accuracy_dev",
                "n_correct": "n_correct_dev",
                "n_total": "n_total_dev",
            }
        )
        merged = merged.merge(dev, on=keys)
    return merged


def masked_state_curve(
    model,
    tokenizer,
    examples: list[Example],
    heads: dict[str, tuple[int, int]],
    cfg: DiffusionConfig,
    seeds: list[int] | None = None,
    timesteps: list[int] | None = None,
) -> pd.DataFrame:
    """Per-instance correctness across diffusion time for selected heads.

    Returns one row per (relation, seed, timestep, instance) rather than a
    pre-aggregated curve. Keeping the raw rows is what lets the offset null be
    recomputed over *exactly* the instances that contribute to the masked-state
    average -- the mismatch that made the previous 43.4%-versus-31.4%
    comparison approximate.
    """
    seeds = seeds or cfg.seeds
    timesteps = timesteps or list(range(cfg.steps))
    rows = []

    for seed in seeds:
        for ti, t in enumerate(timesteps):
            for si, example in enumerate(examples):
                relevant = [i for i in example.relations if i.relation in heads]
                if not relevant:
                    continue

                attentions, state = attentions_at_time(
                    model,
                    tokenizer,
                    example.text,
                    diffusion_time=t,
                    steps=cfg.steps,
                    seed=seed,
                    include_bos=cfg.include_bos,
                )
                seq_len = len(state.tokens)

                for inst in relevant:
                    layer, head = heads[inst.relation]
                    a_span = [x for x in inst.attender_span if x < seq_len]
                    r_span = {x for x in inst.receiver_span if x < seq_len}
                    if not a_span or not r_span:
                        continue

                    pred = receiver_predictions(
                        attentions,
                        layer,
                        a_span,
                        cfg.attender_token,
                        cfg.exclude_bos,
                        cfg.exclude_self,
                    )[head]

                    endpoints = list(a_span) + list(r_span)
                    both_masked = not any(state.is_visible[p] for p in endpoints)

                    rows.append(
                        {
                            "relation": inst.relation,
                            "layer": layer,
                            "head": head,
                            "seed": seed,
                            "timestep": t,
                            "sentence_idx": si,
                            "correct": int(pred in r_span),
                            "both_endpoints_masked": both_masked,
                            "n_masked": state.n_masked,
                            "word_distance": inst.word_distance,
                            "attender_span": inst.attender_span,
                            "receiver_span": inst.receiver_span,
                        }
                    )

            print(f"[curve] seed {seed}: timestep {ti + 1}/{len(timesteps)}")

    return pd.DataFrame(rows)


def aggregate_curve(raw: pd.DataFrame, min_masked: int = 25) -> pd.DataFrame:
    """Collapse raw correctness rows into masked / unmasked means per seed.

    The `min_masked` gate exists because late in denoising almost nothing is
    masked, so "both endpoints masked" selects a vanishing and unrepresentative
    set of instances. Frames below the threshold are excluded from the masked
    statistic only.
    """
    masked = raw[raw["both_endpoints_masked"] & (raw["n_masked"] >= min_masked)]
    unmasked = raw[~raw["both_endpoints_masked"]]

    per_seed = []
    for label, frame in (("masked", masked), ("unmasked", unmasked)):
        if frame.empty:
            continue
        grouped = (
            frame.groupby(["relation", "seed"])["correct"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped["state"] = label
        per_seed.append(grouped)

    if not per_seed:
        return pd.DataFrame()

    combined = pd.concat(per_seed, ignore_index=True)
    return (
        combined.groupby(["relation", "state"])
        .agg(
            accuracy_mean=("mean", "mean"),
            accuracy_std=("mean", "std"),
            n_instances=("count", "sum"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

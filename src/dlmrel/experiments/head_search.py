from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RELATION_NAMES, Config, DiffusionConfig
from ..data import examples_for_split
from ..diffusion import attentions_at_time, receiver_predictions
from ..evaluation.metrics import build_null_table
from ..evaluation.statistics import build_head_vs_null_table, positional_selectivity
from ..relations import Example


def score_split(
    model,
    tokenizer,
    examples: list[Example],
    cfg: DiffusionConfig,
    split_name: str = "",
    log_every: int = 100,
) -> pd.DataFrame:
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
            print(f"[head_search] {split_name}: {i + 1}/{len(examples)} sentences")

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

    print(f"[head_search] {split_name}: totals {dict(totals)}")
    return pd.DataFrame(rows)


def merge_splits(select: pd.DataFrame, test: pd.DataFrame, dev: pd.DataFrame | None = None) -> pd.DataFrame:
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


def run(model, tokenizer, cfg: Config, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.out_dir)

    frames = {}
    for name in ("select", "dev", "test"):
        examples = examples_for_split(cfg, tokenizer, name)
        frames[name] = score_split(model, tokenizer, examples, cfg.diffusion, name)
        frames[name].to_csv(out / f"head_scores_{name}.csv", index=False)

    merged = merge_splits(frames["select"], frames["test"], frames["dev"])
    merged.to_csv(out / "head_scores_merged.csv", index=False)

    null_path = data_dir / "offset_null.csv"
    if null_path.exists():
        nulls = pd.read_csv(null_path)
    else:
        frame = pd.read_csv(data_dir / "relation_instances.csv")
        nulls = build_null_table(
            frame[frame["split"] == "select"],
            frame[frame["split"] == "test"],
            list(RELATION_NAMES),
            cfg.analysis.offset_range,
            cfg.diffusion.attender_token,
        )

    headline = build_head_vs_null_table(merged, nulls)
    headline.to_csv(out / "head_vs_null.csv", index=False)
    print(headline.to_string(index=False))

    profile = positional_selectivity(merged)
    profile.to_csv(out / "head_profiles.csv", index=False)

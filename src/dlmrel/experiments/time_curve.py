from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config, DiffusionConfig
from ..data import examples_for_split
from ..diffusion import attentions_at_time, receiver_predictions
from ..evaluation.metrics import offset_correctness
from ..relations import Example


def masked_state_curve(
    model,
    tokenizer,
    examples: list[Example],
    heads: dict[str, tuple[int, int]],
    cfg: DiffusionConfig,
    seeds: list[int] | None = None,
    timesteps: list[int] | None = None,
) -> pd.DataFrame:
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

            print(f"[time_curve] seed {seed}: timestep {ti + 1}/{len(timesteps)}")

    return pd.DataFrame(rows)


def aggregate_curve(raw: pd.DataFrame, min_masked: int = 25) -> pd.DataFrame:
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


def run(model, tokenizer, cfg: Config, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.out_dir)

    merged = pd.read_csv(data_dir / "head_search" / "head_scores_merged.csv")
    heads = {
        relation: (int(best["layer"]), int(best["head"]))
        for relation, group in merged.groupby("relation")
        for best in [group.sort_values("accuracy_select", ascending=False).iloc[0]]
    }

    examples = examples_for_split(cfg, tokenizer, "test")

    if cfg.diffusion.timesteps:
        timesteps = cfg.diffusion.timesteps
    else:
        stride = max(1, cfg.diffusion.timestep_stride)
        timesteps = list(range(0, cfg.diffusion.steps, stride))
    if cfg.diffusion.n_curve_sentences is not None:
        examples = examples[: cfg.diffusion.n_curve_sentences]

    raw = masked_state_curve(model, tokenizer, examples, heads, cfg.diffusion, timesteps=timesteps)
    raw.to_csv(out / "curve_raw.csv", index=False)

    agg = aggregate_curve(raw, cfg.diffusion.min_masked_positions)
    agg.to_csv(out / "curve_aggregate.csv", index=False)
    print(agg.to_string(index=False))

    null_path = data_dir / "offset_null.csv"
    if null_path.exists():
        nulls = pd.read_csv(null_path).set_index("relation")
        masked = raw[
            raw["both_endpoints_masked"]
            & (raw["n_masked"] >= cfg.diffusion.min_masked_positions)
        ]
        rows = []
        for relation, group in masked.groupby("relation"):
            k = int(nulls.loc[relation, "k"])
            rows.append(
                {
                    "relation": relation,
                    "head_masked_acc": float(group["correct"].mean()),
                    "null_matched_acc": float(
                        offset_correctness(group, k, cfg.diffusion.attender_token).mean()
                    ),
                    "n": len(group),
                }
            )
        matched = pd.DataFrame(rows)
        matched["delta"] = matched["head_masked_acc"] - matched["null_matched_acc"]
        matched.to_csv(out / "masked_state_vs_null.csv", index=False)
        print(matched.to_string(index=False))

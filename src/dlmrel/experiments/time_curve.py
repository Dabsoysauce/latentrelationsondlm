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


def selected_heads(head_search_dir: Path) -> dict[str, tuple[int, int]]:
    """The per-relation head to trace over diffusion time.

    Prefers the full per-head scores, but falls back to `head_vs_null.csv`,
    which stores the same winner: both take the top row per relation ordered by
    selection-split accuracy. The fallback matters because the merged scores are
    a 1024-row-per-relation intermediate that is not always kept, and re-running
    the search on a 7B model to recover two integers per relation is hours of
    GPU time for a number already on disk.
    """
    merged_path = head_search_dir / "head_scores_merged.csv"
    if merged_path.exists():
        merged = pd.read_csv(merged_path)
        return {
            relation: (int(best["layer"]), int(best["head"]))
            for relation, group in merged.groupby("relation")
            for best in [group.sort_values("accuracy_select", ascending=False).iloc[0]]
        }

    headline_path = head_search_dir / "head_vs_null.csv"
    if not headline_path.exists():
        raise FileNotFoundError(
            f"no head selection in {head_search_dir}; run head_search first "
            "(expected head_scores_merged.csv or head_vs_null.csv)"
        )
    headline = pd.read_csv(headline_path)
    # `head` collides with DataFrame.head, so index by label rather than attribute.
    return {
        str(row["relation"]): (int(row["layer"]), int(row["head"]))
        for _, row in headline.iterrows()
    }


def run(model, tokenizer, cfg: Config, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.out_dir)

    heads = selected_heads(data_dir / "head_search")
    print(f"[time_curve] tracing {len(heads)} heads: {heads}")

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

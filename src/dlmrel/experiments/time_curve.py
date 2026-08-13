from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config, DiffusionConfig
from ..data import examples_for_split
from ..diffusion import attentions_at_time, endpoint_visibility, receiver_predictions
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

                    visibility = endpoint_visibility(state.is_visible, a_span, sorted(r_span))

                    rows.append(
                        {
                            "relation": inst.relation,
                            "layer": layer,
                            "head": head,
                            "seed": seed,
                            "timestep": t,
                            "normalized_progress": t / max(cfg.steps - 1, 1),
                            "sentence_idx": si,
                            "correct": int(pred in r_span),
                            "visibility": visibility,
                            "both_endpoints_masked": visibility == "both_masked",
                            "n_masked": state.n_masked,
                            "word_distance": inst.word_distance,
                            "attender_span": inst.attender_span,
                            "receiver_span": inst.receiver_span,
                        }
                    )

            print(f"[time_curve] seed {seed}: timestep {ti + 1}/{len(timesteps)}")

    return pd.DataFrame(rows)


def aggregate_curve(raw: pd.DataFrame, min_masked: int = 25) -> pd.DataFrame:
    """Retain time and four endpoint states; filter on relation denominators."""
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "visibility" not in frame:
        frame["visibility"] = frame["both_endpoints_masked"].map({True: "both_masked", False: "both_visible"})
    denominator = frame.groupby(["relation", "seed", "timestep", "visibility"], observed=True)[
        "correct"
    ].transform("size")
    frame = frame[(frame["visibility"] != "both_masked") | (denominator >= min_masked)]
    per_seed = (
        frame.groupby(
            ["relation", "layer", "head", "seed", "timestep", "normalized_progress", "visibility"],
            observed=True,
        )["correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    return (
        per_seed.groupby(
            ["relation", "layer", "head", "timestep", "normalized_progress", "visibility"],
            observed=True,
        )
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
        masked = raw[raw["both_endpoints_masked"] & (raw["n_masked"] >= cfg.diffusion.min_masked_positions)]
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

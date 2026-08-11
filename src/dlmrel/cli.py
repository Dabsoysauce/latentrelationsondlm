"""Command line entry points.

The pipeline is split so that everything not requiring a GPU can be built,
inspected and unit-tested first:

    dlmrel data     build splits and extract gold relations   (CPU)
    dlmrel nulls    fit and report the fixed-offset baseline  (CPU)
    dlmrel search   score every head on every relation        (GPU)
    dlmrel curve    masked-state accuracy over diffusion time (GPU)
    dlmrel analyze  head-vs-null tables and diagnostics       (CPU)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import RELATION_NAMES, Config


def _out(cfg: Config) -> Path:
    path = Path(cfg.out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_data(cfg: Config) -> None:
    from transformers import AutoTokenizer

    from .relations import relations_to_records
    from .splits import build_all_splits

    out = _out(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Deliberately the same call the GPU stages use to rebuild their splits. It
    # used to be duplicated here, which meant a change to one path silently did
    # not reach the other.
    splits = build_all_splits(cfg, tokenizer)

    records = []
    for name, examples in splits.items():
        records.extend(relations_to_records(examples, name))
        pd.DataFrame({"split": name, "sentence": [e.text for e in examples]}).to_csv(
            out / f"sentences_{name}.csv", index=False
        )

    frame = pd.DataFrame(records)
    frame.to_csv(out / "relation_instances.csv", index=False)
    cfg.save(out / "config.yaml")

    print(f"\n[data] {len(frame)} relation instances -> {out/'relation_instances.csv'}")
    print(pd.crosstab(frame["relation"], frame["split"]))


def cmd_nulls(cfg: Config) -> None:
    from .nulls import build_null_table, offset_distribution

    out = _out(cfg)
    frame = pd.read_csv(out / "relation_instances.csv")
    select = frame[frame["split"] == "select"]
    test = frame[frame["split"] == "test"]

    table = build_null_table(
        select,
        test,
        list(RELATION_NAMES),
        cfg.analysis.offset_range,
        cfg.diffusion.attender_token,
        cfg.analysis.n_bootstrap,
        cfg.analysis.ci,
    )
    table.to_csv(out / "offset_null.csv", index=False)
    print(table.to_string(index=False))

    print("\n[nulls] true offset distribution per relation (word distance):")
    for relation in RELATION_NAMES:
        dist = offset_distribution(frame, relation)
        if dist.empty:
            continue
        top = ", ".join(f"{int(k):+d}:{v:.0%}" for k, v in dist.nlargest(4).items())
        # A concentrated distribution means the relation is trivially solvable
        # by a fixed offset; a spread one means it is not.
        print(f"  {relation:22s} {top}")


def cmd_search(cfg: Config) -> None:
    from .model import load_model
    from .scoring import merge_splits, score_split
    from .splits import examples_for_split

    out = _out(cfg)
    model, tokenizer, meta = load_model(cfg.model)
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))

    frames = {}
    for name in ("select", "dev", "test"):
        examples = examples_for_split(cfg, tokenizer, name)
        frames[name] = score_split(model, tokenizer, examples, cfg.diffusion, name)
        frames[name].to_csv(out / f"head_scores_{name}.csv", index=False)

    merged = merge_splits(frames["select"], frames["test"], frames["dev"])
    merged.to_csv(out / "head_scores_merged.csv", index=False)
    print(f"[search] merged scores -> {out/'head_scores_merged.csv'}")


def cmd_curve(cfg: Config) -> None:
    from .model import load_model
    from .scoring import aggregate_curve, masked_state_curve
    from .splits import examples_for_split

    out = _out(cfg)
    merged = pd.read_csv(out / "head_scores_merged.csv")
    # Heads are chosen on the selection split only.
    heads = {
        relation: (int(best["layer"]), int(best["head"]))
        for relation, group in merged.groupby("relation")
        for best in [group.sort_values("accuracy_select", ascending=False).iloc[0]]
    }
    print(f"[curve] heads under test: {heads}")

    model, tokenizer, _ = load_model(cfg.model)
    examples = examples_for_split(cfg, tokenizer, "test")

    raw = masked_state_curve(model, tokenizer, examples, heads, cfg.diffusion)
    raw.to_csv(out / "curve_raw.csv", index=False)

    agg = aggregate_curve(raw, cfg.diffusion.min_masked_positions)
    agg.to_csv(out / "curve_aggregate.csv", index=False)
    print(agg.to_string(index=False))


def cmd_analyze(cfg: Config) -> None:
    from .nulls import build_null_table, offset_correctness
    from .stats import build_head_vs_null_table, positional_selectivity

    out = _out(cfg)
    frame = pd.read_csv(out / "relation_instances.csv")
    merged = pd.read_csv(out / "head_scores_merged.csv")

    null_path = out / "offset_null.csv"
    if null_path.exists():
        nulls = pd.read_csv(null_path)
    else:
        nulls = build_null_table(
            frame[frame["split"] == "select"],
            frame[frame["split"] == "test"],
            list(RELATION_NAMES),
            cfg.analysis.offset_range,
            cfg.diffusion.attender_token,
        )

    headline = build_head_vs_null_table(merged, nulls)
    headline.to_csv(out / "head_vs_null.csv", index=False)
    print("\n=== headline: best head per relation vs fixed-offset null ===")
    print(headline.to_string(index=False))

    profile = positional_selectivity(merged)
    profile.to_csv(out / "head_profiles.csv", index=False)
    print("\n=== most relation-selective heads (candidate relational heads) ===")
    print(
        profile.nlargest(5, "selectivity")[
            ["layer", "head", "max_relation", "selectivity", "adjacency_bias"]
        ].to_string(index=False)
    )
    print("\n=== strongest adjacency bias (candidate positional heads) ===")
    print(
        profile.nlargest(5, "adjacency_bias")[
            ["layer", "head", "adjacent_mean", "distant_mean", "adjacency_bias"]
        ].to_string(index=False)
    )

    # Recompute the null on exactly the instances entering the masked-state
    # average, rather than on the whole split.
    curve_path = out / "curve_raw.csv"
    if curve_path.exists():
        raw = pd.read_csv(curve_path)
        masked = raw[
            raw["both_endpoints_masked"]
            & (raw["n_masked"] >= cfg.diffusion.min_masked_positions)
        ]
        rows = []
        for relation, group in masked.groupby("relation"):
            k = int(nulls.set_index("relation").loc[relation, "k"])
            rows.append(
                {
                    "relation": relation,
                    "head_masked_acc": float(group["correct"].mean()),
                    "null_matched_acc": float(
                        offset_correctness(
                            group, k, cfg.diffusion.attender_token
                        ).mean()
                    ),
                    "n": len(group),
                }
            )
        matched = pd.DataFrame(rows)
        matched["delta"] = matched["head_masked_acc"] - matched["null_matched_acc"]
        matched.to_csv(out / "masked_state_vs_null.csv", index=False)
        print("\n=== masked state vs null, matched instances ===")
        print(matched.to_string(index=False))


COMMANDS = {
    "data": cmd_data,
    "nulls": cmd_nulls,
    "search": cmd_search,
    "curve": cmd_curve,
    "analyze": cmd_analyze,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dlmrel", description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="override out_dir")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if args.out:
        cfg.out_dir = args.out
    COMMANDS[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

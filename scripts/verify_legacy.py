"""Reproduce the pilot table in the README from the previous pipeline's outputs.

This is the bridge between the old notebooks and this package: it runs the new
`dlmrel.nulls` and `dlmrel.stats` code over the *old* result CSVs and checks it
recovers the same numbers. If this drifts, the rebuild has changed a definition
somewhere and the comparison to prior results is no longer valid.

Usage:
    python scripts/verify_legacy.py \
        --diffugpt   ~/Downloads/_udseed \
        --diffullama ~/Downloads/relation_head_search_ud_gold_diffullama

Expects each directory to contain the legacy filenames:
    ud_gold_relation_instances.csv
    ud_relation_head_scores_merged.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlmrel.config import RELATION_NAMES
from dlmrel.nulls import build_null_table
from dlmrel.stats import n_heads_above_null, selection_rank_correlation

# Legacy columns are named for a two-split world; map them onto the new names.
LEGACY_RENAME = {
    "accuracy_train": "accuracy_select",
    "n_correct_train": "n_correct_select",
    "n_total_train": "n_total_select",
    "accuracy_eval": "accuracy_test",
    "n_correct_eval": "n_correct_test",
    "n_total_eval": "n_total_test",
}


def report(label: str, directory: Path) -> pd.DataFrame:
    instances = pd.read_csv(directory / "ud_gold_relation_instances.csv")
    scores = pd.read_csv(directory / "ud_relation_head_scores_merged.csv").rename(
        columns=LEGACY_RENAME
    )

    # The legacy split names are train/eval; the null is fit on train and
    # reported on eval, matching how the heads were selected.
    nulls = build_null_table(
        instances[instances["split"] == "train"],
        instances[instances["split"] == "eval"],
        list(RELATION_NAMES),
        offset_range=(-15, 15),
        attender_token="last",
        n_boot=2000,
    ).set_index("relation")

    rows = []
    for relation, group in scores.groupby("relation"):
        null_acc = float(nulls.loc[relation, "null_test_acc"])
        best = group.sort_values("accuracy_select", ascending=False).iloc[0]
        rho, _ = selection_rank_correlation(group, "accuracy_select", "accuracy_test")
        rows.append(
            {
                "relation": relation,
                "k": int(nulls.loc[relation, "k"]),
                "null": round(null_acc, 3),
                "head": round(float(best["accuracy_test"]), 3),
                "delta": round(float(best["accuracy_test"]) - null_acc, 3),
                "above_null": f"{n_heads_above_null(group, null_acc, 'accuracy_test')}"
                f"/{len(group)}",
                "rho": round(rho, 3),
            }
        )

    table = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(f"\n=== {label} ===")
    print(table.to_string(index=False))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffugpt", type=Path)
    parser.add_argument("--diffullama", type=Path)
    args = parser.parse_args()

    if not args.diffugpt and not args.diffullama:
        parser.error("pass at least one of --diffugpt / --diffullama")

    if args.diffugpt:
        report("DiffuGPT-S (12L x 12H)", args.diffugpt)
    if args.diffullama:
        report("DiffuLLaMA-7B (32L x 32H)", args.diffullama)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

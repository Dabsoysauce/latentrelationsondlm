from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path("results")


def _read(model: str, experiment: str, filename: str) -> pd.DataFrame | None:
    p = RESULTS / model / experiment / filename
    return pd.read_csv(p) if p.exists() else None


def compare_head_search(models: list[str]) -> pd.DataFrame:
    cols = ["relation", "layer", "head", "head_test_acc", "null_test_acc", "margin", "verdict"]
    frames = []
    for m in models:
        t = _read(m, "head_search", "head_vs_null.csv")
        if t is None:
            continue
        t = t[[c for c in cols if c in t.columns]].copy()
        t.insert(0, "model", m)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_pos_probe(models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        t = _read(m, "pos_probe", "pos_probe.csv")
        if t is None:
            continue
        best = t.loc[t["accuracy"].idxmax()]
        rows.append(
            {
                "model": m,
                "best_depth": int(best["depth"]),
                "peak_accuracy": float(best["accuracy"]),
                "lexical_baseline": float(best["lexical_baseline"]),
                "delta_vs_lexical": float(best["delta_vs_lexical"]),
            }
        )
    return pd.DataFrame(rows)


def split_overlap(models: list[str]) -> pd.DataFrame:
    sequences = {}
    for m in models:
        p = RESULTS / m / "sentences_test.csv"
        if p.exists():
            sequences[m] = list(pd.read_csv(p)["sentence"])

    names = list(sequences)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sequences[names[i]], sequences[names[j]]
            rows.append(
                {
                    "model_a": names[i],
                    "model_b": names[j],
                    "n_a": len(a),
                    "n_b": len(b),
                    "overlap": len(set(a) & set(b)),
                    "identical_order": a == b,
                }
            )
    return pd.DataFrame(rows)


_COMPARATORS = {
    "head_search": compare_head_search,
    "pos_probe": compare_pos_probe,
}


def compare(models: list[str], experiment: str, out: Path) -> pd.DataFrame:
    if experiment not in _COMPARATORS:
        raise ValueError(f"no comparison defined for {experiment}")
    out.mkdir(parents=True, exist_ok=True)

    table = _COMPARATORS[experiment](models)
    table.to_csv(out / f"{experiment}_comparison.csv", index=False)
    if not table.empty:
        print(table.to_string(index=False))

    overlap = split_overlap(models)
    if not overlap.empty:
        overlap.to_csv(out / "split_overlap.csv", index=False)
        print()
        print(overlap.to_string(index=False))
    return table

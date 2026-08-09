"""Publication figures, generated from the pipeline's own output CSVs.

Nothing here hardcodes a result. Each figure reads the file the corresponding
stage writes, so re-running a stage and re-running this script keeps the paper
in sync with the data:

    fig1_distance_distributions.pdf   <- relation_instances.csv   (dlmrel data)
    fig2_head_vs_null.pdf             <- head_vs_null.csv         (dlmrel analyze)
    fig3_head_profiles.pdf            <- head_profiles.csv        (dlmrel analyze)

Missing inputs are reported and skipped rather than failing, so the script is
usable before the GPU stages have run.

Usage:
    python scripts/make_figures.py \
        --small results/diffugpt-s-ewt --large results/diffullama-ewt \
        --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Two categorical hues, validated for colorblind separation against a white
# page (worst all-pairs OKLab dE 24.7 protan, 33.6 normal vision). Figures are
# single-mode on purpose: they are printed on paper.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#c9c8c4"

# NeurIPS and COLM both set a 5.5in text width.
TEXT_WIDTH = 5.5

RELATION_LABEL = {
    "object_to_verb": "object $\\rightarrow$ verb",
    "subject_to_verb": "subject $\\rightarrow$ verb",
    "object_adj_to_noun": "obj. adj. $\\rightarrow$ noun",
    "subject_adj_to_noun": "subj. adj. $\\rightarrow$ noun",
    "object_det_to_noun": "obj. det. $\\rightarrow$ noun",
    "subject_det_to_noun": "subj. det. $\\rightarrow$ noun",
}
# object->verb is the relation no fixed offset solves, so it is the one the
# reader should be looking at in every figure.
FOCUS = "object_to_verb"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.color": FAINT,
            "grid.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf")


def fig_distance_distributions(instances: Path, out: Path) -> None:
    """Where the receiver sits relative to the attender, per relation.

    This is the figure that justifies the whole design before a model is
    involved: it is computed from the treebank alone. A relation whose distance
    has one dominant value is solvable by a fixed offset; object->verb splits
    its mass across -1 and -2 with a long tail, so no single offset can win.
    """
    if not instances.exists():
        print(f"  skip fig1: no {instances}")
        return
    df = pd.read_csv(instances)
    df = df[df["split"] == "test"]

    order = [r for r in RELATION_LABEL if r in set(df["relation"])]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 2.9), sharey=True)

    for ax, relation in zip(axes.ravel(), order):
        sub = df[df["relation"] == relation]["word_distance"]
        clipped = sub.clip(-8, 8)
        counts = clipped.value_counts(normalize=True).sort_index()
        focus = relation == FOCUS
        colour = ORANGE if focus else BLUE

        ax.bar(counts.index, counts.values, width=0.78, color=colour, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_xlim(-8.8, 8.8)
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

        # State the ceiling a positional head faces, as a share not a claim.
        ax.set_title(
            f"{RELATION_LABEL[relation]}\nmode {counts.max():.0%}, n={len(sub):,}",
            color=ORANGE if focus else INK,
            pad=4,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("share of instances")
    for ax in axes[1, :]:
        ax.set_xlabel("receiver $-$ attender (words)")
    fig.tight_layout(pad=0.4)
    save(fig, out, "fig1_distance_distributions")


def fig_head_vs_null(tables: dict[str, Path], out: Path) -> None:
    """Best head against the fixed-offset baseline, with intervals.

    A dot-and-interval plot rather than bars: these are estimates with
    uncertainty, and bars would imply the distance from zero is meaningful when
    the distance from the baseline is what matters.
    """
    present = {k: v for k, v in tables.items() if v.exists()}
    if not present:
        print("  skip fig2: no head_vs_null.csv (run `dlmrel analyze`)")
        return

    fig, axes = plt.subplots(
        1, len(present), figsize=(TEXT_WIDTH, 2.5), sharex=True, squeeze=False
    )
    for ax, (title, path) in zip(axes[0], present.items()):
        t = pd.read_csv(path).sort_values("delta")
        y = np.arange(len(t))

        for i, row in enumerate(t.itertuples()):
            ax.plot(
                [row.null_test_acc, row.head_test_acc], [i, i],
                color=FAINT, linewidth=1.0, zorder=1,
            )
        ax.hlines(y, t["head_ci_lo"], t["head_ci_hi"],
                  color=BLUE, linewidth=1.6, zorder=2)
        ax.scatter(t["head_test_acc"], y, s=22, color=BLUE, zorder=3,
                   label="best head (95% CI)")
        ax.scatter(t["null_test_acc"], y, s=22, color=ORANGE, marker="D",
                   zorder=3, label="fixed-offset baseline")

        ax.set_yticks(y)
        ax.set_yticklabels([RELATION_LABEL.get(r, r) for r in t["relation"]])
        ax.set_xlim(0, 1)
        ax.set_xlabel("held-out receiver accuracy")
        ax.set_title(title, pad=4)
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    axes[0][0].legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.4)
    save(fig, out, "fig2_head_vs_null")


def fig_head_profiles(profiles: dict[str, Path], out: Path) -> None:
    """Every head as a point: does it serve one relation, or all the near ones?

    Positional heads sit far right (high adjacency bias, low selectivity);
    a relational head sits high (solves one relation and nothing else). The
    claim is about two populations, so every head has to be on the plot.
    """
    present = {k: v for k, v in profiles.items() if v.exists()}
    if not present:
        print("  skip fig3: no head_profiles.csv (run `dlmrel analyze`)")
        return

    fig, axes = plt.subplots(
        1, len(present), figsize=(TEXT_WIDTH, 2.4), squeeze=False
    )
    for ax, (title, path) in zip(axes[0], present.items()):
        p = pd.read_csv(path)
        ax.scatter(p["adjacency_bias"], p["selectivity"], s=6,
                   color=FAINT, linewidth=0, zorder=1)

        relational = p.loc[p["selectivity"].idxmax()]
        positional = p.loc[p["adjacency_bias"].idxmax()]
        for row, colour, note in (
            (relational, ORANGE, "relational"),
            (positional, BLUE, "positional"),
        ):
            ax.scatter(row["adjacency_bias"], row["selectivity"], s=40,
                       color=colour, zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(
                f"L{int(row['layer'])} H{int(row['head'])}\n{note}",
                (row["adjacency_bias"], row["selectivity"]),
                textcoords="offset points", xytext=(6, 4),
                color=colour, fontsize=6.5,
            )

        ax.axhline(0, color=FAINT, linewidth=0.6)
        ax.axvline(0, color=FAINT, linewidth=0.6)
        ax.set_xlabel("adjacency bias")
        ax.set_ylabel("relation selectivity")
        ax.set_title(title, pad=4)
        ax.grid(True)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    save(fig, out, "fig3_head_profiles")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=Path, default=Path("results/diffugpt-s-ewt"))
    ap.add_argument("--large", type=Path, default=Path("results/diffullama-ewt"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()

    set_style()
    small, large, out = args.small, args.large, args.out

    print("figures:")
    # Distances are a property of the treebank, so either run's instances work.
    fig_distance_distributions(large / "relation_instances.csv", out)
    fig_head_vs_null(
        {"DiffuGPT-S (144 heads)": small / "head_vs_null.csv",
         "DiffuLLaMA-7B (1024 heads)": large / "head_vs_null.csv"}, out
    )
    fig_head_profiles(
        {"DiffuGPT-S": small / "head_profiles.csv",
         "DiffuLLaMA-7B": large / "head_profiles.csv"}, out
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

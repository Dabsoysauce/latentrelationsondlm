import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#c9c8c4"
TEXT_WIDTH = 5.5


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
            "grid.color": FAINT,
            "grid.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/dream-7b-ewt"))
    ap.add_argument("--relation", default="object_to_verb")
    ap.add_argument("--title", default="Dream-7B, object $\\rightarrow$ verb (L2 H11)")
    ap.add_argument("--out", type=Path, default=Path("figures-curve"))
    args = ap.parse_args()

    raw = pd.read_csv(args.run / "curve_raw.csv")
    raw = raw[raw["relation"] == args.relation]
    if raw.empty:
        raise SystemExit(f"no rows for {args.relation}")

    nulls = pd.read_csv(args.run / "offset_null.csv").set_index("relation")
    null_acc = float(nulls.loc[args.relation, "null_test_acc"])

    set_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.6))

    overall = raw.groupby("timestep")["correct"].mean()
    ax.plot(overall.index, overall.values, color=INK, linewidth=1.2,
            marker="o", markersize=3, label="all instances", zorder=3)

    for flag, colour, label in (
        (True, ORANGE, "both endpoints masked"),
        (False, BLUE, "at least one endpoint visible"),
    ):
        sub = raw[raw["both_endpoints_masked"] == flag]
        if sub.empty:
            continue
        grouped = sub.groupby("timestep")["correct"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= 30]
        ax.plot(grouped.index, grouped["mean"], color=colour, linewidth=1.2,
                marker="o", markersize=3, label=label, zorder=3)

    ax.axhline(null_acc, color=MUTED, linewidth=0.9, linestyle="--", zorder=2)
    ax.annotate(
        f"fixed-offset baseline ({null_acc:.2f})",
        xy=(raw["timestep"].max(), null_acc),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
        color=MUTED,
    )

    ax.set_xlabel("diffusion timestep (0 = fully masked, 63 = fully revealed)")
    ax.set_ylabel("receiver accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(args.title, pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"fig4_curve_{args.relation}.{ext}")
    plt.close(fig)
    print(f"wrote {args.out}/fig4_curve_{args.relation}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

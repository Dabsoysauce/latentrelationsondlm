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


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf")


def fig_entropy(run: Path, out: Path, label: str) -> None:
    path = run / "attention_entropy.csv"
    if not path.exists():
        print(f"  skip entropy: no {path}")
        return
    t = pd.read_csv(path)
    per_layer = t.groupby("layer")[["entropy_norm", "sink_mass"]].mean()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.4))
    ax.plot(per_layer.index, per_layer["entropy_norm"], color=BLUE,
            marker="o", markersize=3, linewidth=1.2, label="normalised entropy")
    ax.plot(per_layer.index, per_layer["sink_mass"], color=ORANGE,
            marker="o", markersize=3, linewidth=1.2, label="attention mass on position 0")
    ax.scatter(t["layer"], t["entropy_norm"], s=4, color=FAINT, zorder=0)

    ax.set_xlabel("layer")
    ax.set_ylabel("mean over heads")
    ax.set_ylim(0, 1)
    ax.set_title(f"{label}: attention entropy and sink mass by layer", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, ncol=2)
    save(fig, out, "fig5_attention_entropy")


def fig_logit_lens(run: Path, out: Path, label: str) -> None:
    path = run / "logit_lens.csv"
    if not path.exists():
        print(f"  skip logit lens: no {path}")
        return
    t = pd.read_csv(path)
    masked = t[t["position_state"] == "masked"]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.4))
    times = sorted(masked["diffusion_time"].unique())
    shades = [ORANGE, BLUE, INK]
    for i, dt in enumerate(times):
        sub = masked[masked["diffusion_time"] == dt].sort_values("depth")
        ax.plot(sub["depth"], sub["accuracy"], color=shades[i % len(shades)],
                marker="o", markersize=3, linewidth=1.2,
                label=f"t={dt}")

    ax.set_xlabel("layer depth (0 = embeddings)")
    ax.set_ylabel("true-token top-1 accuracy")
    ax.set_title(f"{label}: logit lens at masked positions", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, title="diffusion time")
    save(fig, out, "fig6_logit_lens")


def fig_pos_probe(run: Path, out: Path, label: str) -> None:
    path = run / "pos_probe.csv"
    if not path.exists():
        print(f"  skip pos probe: no {path}")
        return
    t = pd.read_csv(path).sort_values("depth")

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.4))
    ax.plot(t["depth"], t["accuracy"], color=BLUE, marker="o", markersize=3,
            linewidth=1.2, label="linear probe", zorder=3)
    ax.axhline(t["lexical_baseline"].iloc[0], color=ORANGE, linestyle="--",
               linewidth=1.0, label="per-token-type tag baseline")
    ax.axhline(t["majority_baseline"].iloc[0], color=MUTED, linestyle=":",
               linewidth=1.0, label="majority class")

    ax.set_xlabel("layer depth (0 = embeddings)")
    ax.set_ylabel("UPOS accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{label}: POS probe by depth", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "fig7_pos_probe")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    set_style()
    print("probe figures:")
    fig_entropy(args.run, args.out, args.label)
    fig_logit_lens(args.run, args.out, args.label)
    fig_pos_probe(args.run, args.out, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

RELATION_LABEL = {
    "object_to_verb": "object to verb",
    "subject_to_verb": "subject to verb",
    "object_adj_to_noun": "obj. adj. to noun",
    "subject_adj_to_noun": "subj. adj. to noun",
    "object_det_to_noun": "obj. det. to noun",
    "subject_det_to_noun": "subj. det. to noun",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.png")


def fig_receiver_over_time(
    curve_raw: Path, offset_null: Path, out: Path, name: str
) -> None:
    raw = pd.read_csv(curve_raw)
    relations = ["object_to_verb", "subject_to_verb"]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    colors = {"object_to_verb": "C0", "subject_to_verb": "C2"}
    for rel in relations:
        sub = raw[raw["relation"] == rel]
        if sub.empty:
            continue
        allc = sub.groupby("timestep")["correct"].mean()
        ax.plot(
            allc.index, allc.values, color=colors[rel], linewidth=1.6,
            label=f"{RELATION_LABEL[rel]}",
        )
        masked = sub[sub["both_endpoints_masked"]]
        g = masked.groupby("timestep")["correct"].agg(["mean", "count"])
        g = g[g["count"] >= 25]
        ax.plot(
            g.index, g["mean"], color=colors[rel], linewidth=1.4, linestyle="--",
            label=f"{RELATION_LABEL[rel]} (masked only)",
        )

    if offset_null.exists():
        nulls = pd.read_csv(offset_null).set_index("relation")
        rnd = float(nulls.loc["object_to_verb", "null_test_acc"])
        ax.axhline(
            rnd, color="0.4", linestyle=":", linewidth=1.0,
            label=f"fixed-offset null ({rnd:.2f})",
        )

    ax.set_xlabel("Diffusion step")
    ax.set_ylabel("Held-out receiver-prediction accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Relation-head receiver prediction over diffusion time")
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, out, name)


def fig_pos_probe(pos_probe: Path, out: Path, name: str, title: str) -> None:
    t = pd.read_csv(pos_probe).sort_values("depth")

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(
        t["depth"], t["accuracy"], "o-", color="C0", linewidth=1.6,
        markersize=4, label="linear probe",
    )
    if "lexical_baseline" in t:
        ax.axhline(
            t["lexical_baseline"].iloc[0], color="C1", linestyle="--", linewidth=1.2,
            label=f"per-token-type tag ({t['lexical_baseline'].iloc[0]:.2f})",
        )
    if "majority_baseline" in t:
        ax.axhline(
            t["majority_baseline"].iloc[0], color="0.4", linestyle=":", linewidth=1.0,
            label=f"majority class ({t['majority_baseline'].iloc[0]:.2f})",
        )

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Held-out POS accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    _save(fig, out, name)


def fig_logit_lens(logit_lens: Path, out: Path, name: str, title: str) -> None:
    t = pd.read_csv(logit_lens)
    masked = t[t["position_state"] == "masked"]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for i, dt in enumerate(sorted(masked["diffusion_time"].unique())):
        sub = masked[masked["diffusion_time"] == dt].sort_values("depth")
        ax.plot(
            sub["depth"], sub["accuracy"], linewidth=1.6, color=f"C{i}",
            label=f"t = {dt}",
        )

    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Masked-token top-1 accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, out, name)


def fig_attention_entropy(entropy: Path, out: Path, name: str, title: str) -> None:
    t = pd.read_csv(entropy)
    g = t.groupby("layer")[["entropy_norm", "sink_mass"]].mean()

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(
        g.index, g["entropy_norm"], "o-", color="C0", linewidth=1.6,
        markersize=3, label="normalised entropy",
    )
    ax.plot(
        g.index, g["sink_mass"], "s-", color="C1", linewidth=1.6,
        markersize=3, label="attention mass on position 0",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean over heads")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper center", framealpha=0.9, ncol=2)
    _save(fig, out, name)

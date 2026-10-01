"""Export paper-sized paired-effect figures and the exact plotted CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(run_dir):
    root = Path(run_dir)
    frame = pd.read_csv(root / "aggregate.csv")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for relation in frame.head_relation.unique():
        part = frame[(frame.head_relation == relation) & (frame.metric == "governor_exact")]
        primary = part[part.is_anchor]

        def save(fig, name, data, relation=relation):
            prefix = output / f"{relation}-{name}"
            data.to_csv(prefix.with_suffix(".csv"), index=False)
            fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        def errors(ax, data, x):
            y = data.difference.to_numpy()
            # Percentile CIs need not enclose the point estimate in tiny samples.
            ax.vlines(x, data.ci_low, data.ci_high, color="#2166ac", linewidth=1.5)
            ax.plot(x, y, "o", color="#2166ac")
            ax.axhline(0, color="0.5", linewidth=0.7)
            ax.set_ylabel("Governor recovery change\n(intervention − baseline)")

        data = primary[(primary.alpha == 0) & (primary.window == "pre")].sort_values("control")
        if not data.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            errors(ax, data, np.arange(len(data)))
            ax.set_xticks(range(len(data)), data.control, rotation=25, ha="right")
            save(fig, "controls", data)
        data = primary[(primary.control == "target") & (primary.alpha == 0)].copy()
        if not data.empty:
            # Windows are categories, not fabricated measurements of instantaneous effect.
            order = {v: i for i, v in enumerate(["entire", "early_pre", "late_pre", "pre", "post"])}
            data = data.sort_values("window", key=lambda s: s.map(order))
            fig, ax = plt.subplots(figsize=(6, 3))
            errors(ax, data, np.arange(len(data)))
            labels = data.window + " [" + data.start.astype(str) + "," + data.stop.astype(str) + ")"
            ax.set_xticks(range(len(data)), labels, rotation=20)
            ax.set_xlabel("Intervention window (early → late)")
            save(fig, "temporal-windows", data)
            sliding = data[data.window == "entire"].sort_values("start")
            if sliding.start.nunique() > 1:
                fig, ax = plt.subplots(figsize=(5, 3))
                errors(ax, sliding, (sliding.start + sliding.stop - 1) / 2)
                ax.set_xlabel("Suppression interval midpoint (denoising step)")
                save(fig, "temporal-curve", sliding)
        data = primary[(primary.control == "target") & (primary.window == "pre")].sort_values("alpha")
        if not data.empty:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.errorbar(
                data.alpha,
                data.intervention_mean,
                yerr=data.intervention_seed_std.fillna(0),
                fmt="o-",
                color="#2166ac",
                capsize=3,
                label="Intervention (seed SD)",
            )
            ax.axhline(data.baseline_mean.iloc[0], linestyle="--", color="0.4", label="Untouched")
            ax.set(xlabel="Head output scale α", ylabel="Exact governor recovery", ylim=(-0.03, 1.03))
            ax.legend(frameon=False)
            save(fig, "dose", data)
        data = frame[
            (frame.head_relation == relation)
            & frame.is_anchor
            & (frame.control == "target")
            & (frame.window == "pre")
            & (frame.alpha == 0)
            & frame.metric.isin(["target_accuracy", "non_target_accuracy", "overall_accuracy"])
        ]
        if not data.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            errors(ax, data, np.arange(len(data)))
            ax.set_xticks(range(len(data)), data.metric.str.replace("_accuracy", ""))
            ax.set_ylabel("Subtoken accuracy change")
            save(fig, "global-degradation", data)
    # One outcome per pair, selecting non-anchor rows separately would bias the matrix.
    raw = pd.read_json(root / "paired.jsonl", lines=True)
    data = raw[(raw.control == "target") & (raw.window == "pre") & (raw.alpha == 0)].copy()
    if not data.empty:
        data["difference"] = data.intervention_governor_exact - data.baseline_governor_exact
        table = data.groupby(["head_relation", "relation"], as_index=False).difference.mean()
        matrix = table.pivot(index="head_relation", columns="relation", values="difference")
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.55 * len(matrix))))
        bound = max(0.01, np.nanmax(np.abs(matrix.to_numpy())))
        im = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set(xlabel="Evaluated gold relation", ylabel="Intervened head relation")
        fig.colorbar(im, ax=ax, label="Governor recovery change")
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"specificity-matrix.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        table.to_csv(output / "specificity-matrix.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    plot_results(parser.parse_args().run_dir)

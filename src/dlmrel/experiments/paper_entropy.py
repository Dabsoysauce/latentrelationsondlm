"""Complete old Attention Entropy analysis at model-relative depths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import attentions_for_state, teacher_forced_trajectory
from ..paper_protocol import (
    attention_entropy_rows,
    map_relative_depths,
    summarize_entropy_trajectory,
)
from .shared import write_frames


def entropy_trajectory_chunk(model, tokenizer, examples, *, seed: int, depth_rows) -> pd.DataFrame:
    rows = []
    for example in examples:
        states = teacher_forced_trajectory(
            model, tokenizer, example.text, steps=64, seed=seed, include_bos=True
        )
        for timestep, state in enumerate(states):
            if timestep in {0, 63} and seed != 42:
                continue
            attentions = attentions_for_state(model, state)
            for depth in depth_rows:
                layer = int(depth["actual_layer_index"])
                values = attention_entropy_rows(
                    attentions[layer][0].detach().float().cpu().numpy()
                )
                for head, entropy in enumerate(values):
                    rows.append(
                        {
                            "sentence_id": example.sentence_id,
                            "treebank": example.source,
                            "seed": seed,
                            "timestep": timestep,
                            "normalized_progress": timestep / 63,
                            **depth,
                            "layer": layer,
                            "head": head,
                            "entropy_normalized": float(entropy),
                            "n_masked": state.n_masked,
                            "bos_query_excluded": True,
                            "bos_source_retained": True,
                        }
                    )
    return pd.DataFrame(rows)


def _depth_mapping(model, tokenizer, example, cfg: RunConfig):
    states = teacher_forced_trajectory(model, tokenizer, example.text, steps=64, seed=42)
    attentions = attentions_for_state(model, states[0])
    return map_relative_depths(len(attentions), cfg.experiment.settings["relative_depths"])


def _summary_rows(per_seed_curve: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    early = tuple(int(value) for value in settings["early_window"])
    late = tuple(int(value) for value in settings["late_window"])
    threshold = float(settings["flat_threshold"])
    deterministic = per_seed_curve[per_seed_curve["seed"] == 42].set_index(
        ["relative_label", "layer", "head", "timestep"]
    )["entropy_normalized"]
    rows = []
    identity = ["relative_label", "configured_fraction", "layer", "head", "seed"]
    for key, group in per_seed_curve.groupby(identity, observed=True):
        series = group.set_index("timestep")["entropy_normalized"].to_dict()
        label, _fraction, layer, head, seed = key
        for endpoint in (0, 63):
            if endpoint not in series:
                series[endpoint] = float(deterministic.loc[(label, layer, head, endpoint)])
        if set(series) != set(range(64)):
            raise ValueError("entropy summary lacks one or more diffusion steps")
        summary = summarize_entropy_trajectory(
            [series[step] for step in range(64)],
            early_window=early,
            late_window=late,
            flat_threshold=threshold,
        )
        rows.append(
            {
                **dict(zip(identity, key, strict=True)),
                **summary,
                "early_window_start": early[0],
                "early_window_end": early[1],
                "late_window_start": late[0],
                "late_window_end": late[1],
            }
        )
    return pd.DataFrame(rows)


def run(model, tokenizer, cfg: RunConfig, run_dir: Path, **_unused: Any) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    if not examples:
        raise ValueError("Attention Entropy has no valid test examples")
    depths = _depth_mapping(model, tokenizer, examples[0], cfg)
    pd.DataFrame(depths).to_csv(run_dir / "relative_depth_mapping.csv", index=False)
    store = SentenceCheckpointStore(run_dir)
    frames = []
    selected_heads = None
    for seed in cfg.experiment.seeds:
        identity = CheckpointIdentity(
            stage="paper-attention-entropy-trajectory",
            seed=seed,
            normalized_progress=-1.0,
            timestep=-1,
            heads=selected_heads,
        )
        frames.append(
            store.run(
                examples,
                identity,
                lambda chunk, _start, current_seed=seed: entropy_trajectory_chunk(
                    model, tokenizer, chunk, seed=current_seed, depth_rows=depths
                ),
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    curve_keys = [
        "seed",
        "treebank",
        "relative_label",
        "configured_fraction",
        "actual_layer_index",
        "total_model_layers",
        "layer",
        "head",
        "timestep",
        "normalized_progress",
    ]
    per_seed_curve = raw.groupby(curve_keys, as_index=False).agg(
        entropy_normalized=("entropy_normalized", "mean"),
        denominator_sentences=("sentence_id", "nunique"),
    )
    per_seed_curve.to_csv(run_dir / "per_seed_trajectories.csv", index=False)
    trajectory_summary = _summary_rows(per_seed_curve, cfg.experiment.settings)
    trajectory_summary.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metric_keys = [
        "treebank",
        "relative_label",
        "configured_fraction",
        "actual_layer_index",
        "total_model_layers",
        "layer",
        "head",
        "timestep",
        "normalized_progress",
    ]
    metrics = per_seed_curve.groupby(metric_keys, as_index=False).agg(
        entropy_mean=("entropy_normalized", "mean"),
        entropy_std=("entropy_normalized", "std"),
        n_seeds=("seed", "nunique"),
        denominator_sentences=("denominator_sentences", "sum"),
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    direction = trajectory_summary.groupby(
        ["relative_label", "layer", "direction"], as_index=False
    ).agg(n_head_seeds=("head", "size"))
    totals = direction.groupby(["relative_label", "layer"])["n_head_seeds"].transform("sum")
    direction["percentage"] = 100 * direction["n_head_seeds"] / totals
    direction.to_csv(run_dir / "direction_percentages.csv", index=False)
    return {
        "development_used": False,
        "timesteps": list(range(64)),
        "stochastic_seeds": list(cfg.experiment.seeds),
        "deterministic_steps_deduplicated": [0, 63],
        "relative_depths": depths,
        "bos_query_excluded": True,
        "bos_source_retained": True,
        "entropy_normalized_by": "log_sequence_length",
        "early_window": cfg.experiment.settings["early_window"],
        "late_window": cfg.experiment.settings["late_window"],
        "window_provenance": cfg.experiment.settings["window_provenance"],
        "test_sentences": len(examples),
    }

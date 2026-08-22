"""Complete old Attention Entropy analysis at model-relative depths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..artifacts import ArtifactError
from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import attention_batches_for_states, attentions_for_state, teacher_forced_trajectory
from ..paper_protocol import (
    attention_entropy_rows,
    map_relative_depths,
    summarize_entropy_trajectory,
)
from .shared import write_frames

ATTENTION_CACHE_STAGE = "paper-shared-attention-entropy-trajectory"
ATTENTION_CACHE_CHUNK_SIZE = 20


def entropy_trajectory_chunk(
    model,
    tokenizer,
    examples,
    *,
    seed: int,
    depth_rows,
    batch_size: int = 8,
) -> pd.DataFrame:
    rows = []
    for example in examples:
        states = teacher_forced_trajectory(
            model, tokenizer, example.text, steps=64, seed=seed, include_bos=True
        )
        indexed_states = [
            (timestep, state)
            for timestep, state in enumerate(states)
            if seed == 42 or timestep not in {0, 63}
        ]
        for start, current, attentions in attention_batches_for_states(
            model,
            [state for _timestep, state in indexed_states],
            batch_size=batch_size,
        ):
            for batch_index, state in enumerate(current):
                timestep = indexed_states[start + batch_index][0]
                for depth in depth_rows:
                    layer = int(depth["actual_layer_index"])
                    values = attention_entropy_rows(
                        attentions[layer][batch_index].detach().float().cpu().numpy()
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


def _load_attention_cache(
    source: str | Path,
    examples,
    cfg: RunConfig,
    depths,
    manifest_hashes: dict[str, str],
) -> list[pd.DataFrame]:
    source = Path(source)
    required = ("summary.json", "config.resolved.yaml", "manifest_refs.json", "run_metadata.json")
    if any(not (source / name).is_file() for name in required):
        raise ArtifactError("attention cache is not a complete run directory")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    source_cfg = yaml.safe_load((source / "config.resolved.yaml").read_text(encoding="utf-8"))
    source_manifests = json.loads((source / "manifest_refs.json").read_text(encoding="utf-8"))
    source_model = source_cfg.get("model", {})
    source_dataset = source_cfg.get("dataset", {})
    source_experiment = source_cfg.get("experiment", {})
    model_identity = ("id", "revision", "tokenizer_revision", "remote_code_revision")
    if any(source_model.get(key) != getattr(cfg.model, key) for key in model_identity):
        raise ArtifactError("attention cache belongs to a different model identity")
    if source_dataset.get("id") != cfg.dataset.id or source_dataset.get("revision") != cfg.dataset.revision:
        raise ArtifactError("attention cache belongs to a different dataset identity")
    if source_experiment.get("id") != "relation_head_receiver_prediction_over_diffusion_time":
        raise ArtifactError("attention cache source is not the English relation time experiment")
    if source_manifests != manifest_hashes:
        raise ArtifactError("attention cache uses a different test manifest")
    if summary.get("completion_status") != "complete" or not summary.get(
        "attention_entropy_cache_exported"
    ):
        raise ArtifactError("attention cache source did not complete a cache export")
    if summary.get("attention_entropy_cache_relative_depths") != depths:
        raise ArtifactError("attention cache relative depths differ from the entropy protocol")

    store = SentenceCheckpointStore(source, chunk_size=ATTENTION_CACHE_CHUNK_SIZE)

    def missing(_chunk, _start):
        raise ArtifactError("attention cache is missing a required sentence chunk")

    frames = []
    for seed in cfg.experiment.seeds:
        identity = CheckpointIdentity(
            stage=ATTENTION_CACHE_STAGE,
            seed=seed,
            normalized_progress=-1.0,
            timestep=-1,
        )
        frame = store.run(examples, identity, missing)
        required_columns = {
            "sentence_id",
            "seed",
            "timestep",
            "relative_label",
            "configured_fraction",
            "actual_layer_index",
            "total_model_layers",
            "layer",
            "head",
            "entropy_normalized",
        }
        if missing_columns := required_columns - set(frame):
            raise ArtifactError(
                f"attention cache rows are missing columns: {sorted(missing_columns)}"
            )
        expected_timesteps = set(range(64)) if seed == 42 else set(range(1, 63))
        if set(frame["seed"].astype(int)) != {seed} or set(
            frame["timestep"].astype(int)
        ) != expected_timesteps:
            raise ArtifactError("attention cache seed/timestep coverage is incompatible")
        cached_depths = (
            frame[
                [
                    "relative_label",
                    "configured_fraction",
                    "actual_layer_index",
                    "total_model_layers",
                ]
            ]
            .drop_duplicates()
            .to_dict("records")
        )
        if cached_depths != depths:
            raise ArtifactError("attention cache rows use incompatible relative depths")
        if not np.isfinite(frame["entropy_normalized"].astype(float)).all():
            raise ArtifactError("attention cache contains nonfinite entropy values")
        frames.append(frame)
    return frames


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


def run(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict[str, str],
    **_unused: Any,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    if not examples:
        raise ValueError("Attention Entropy has no valid test examples")
    depths = _depth_mapping(model, tokenizer, examples[0], cfg)
    pd.DataFrame(depths).to_csv(run_dir / "relative_depth_mapping.csv", index=False)
    frames = []
    selected_heads = None
    cache_source = cfg.runtime.attention_cache
    if cache_source:
        frames = _load_attention_cache(cache_source, examples, cfg, depths, manifest_hashes)
    else:
        store = SentenceCheckpointStore(run_dir, chunk_size=ATTENTION_CACHE_CHUNK_SIZE)
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
                        model,
                        tokenizer,
                        chunk,
                        seed=current_seed,
                        depth_rows=depths,
                        batch_size=cfg.runtime.timestep_batch_size,
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
        "timestep_microbatch_size": cfg.runtime.timestep_batch_size,
        "vocabulary_logits_computed_for_attention_only_passes": False,
        "attention_cache_reused": bool(cache_source),
        "attention_cache_source": str(Path(cache_source).resolve()) if cache_source else None,
    }

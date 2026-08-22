"""Exact direct logit attribution and matched single-head causal ablation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..artifacts import ArtifactError
from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import state_at_time
from ..models.decomposition import capture_or_ablate_projection, capture_projection_inputs
from ..paper_protocol import PaperLockSet, projection_head_slice, write_resolved_selection_locks
from .shared import instance_metadata, write_frames


def _selection_scores_path(locks: PaperLockSet) -> Path:
    candidates = [
        locks.source.parent / "selection_all_head_scores.csv",
        locks.source / "selection_all_head_scores.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise ArtifactError(
        "matched controls require selection_all_head_scores.csv beside the selection locks"
    )


def matched_low_relation_controls(locks: PaperLockSet) -> dict[str, tuple[int, int]]:
    scores = pd.read_csv(_selection_scores_path(locks))
    controls = {}
    for relation, lock in locks.locks.items():
        candidates = scores[
            (scores["relation"] == relation)
            & (scores["layer"] == lock.layer)
            & (scores["head"] != lock.head)
        ].sort_values(["accuracy", "n_total", "head"], ascending=[True, False, True])
        if candidates.empty:
            candidates = scores[
                (scores["relation"] == relation)
                & ~((scores["layer"] == lock.layer) & (scores["head"] == lock.head))
            ].sort_values(
                ["accuracy", "n_total", "layer", "head"],
                ascending=[True, False, True, True],
            )
        if candidates.empty:
            raise ArtifactError(f"no matched low-relation control for {relation}")
        row = candidates.iloc[0]
        controls[relation] = (int(row.layer), int(row.head))
    return controls


def _rank(logits: torch.Tensor, target: int) -> int:
    return int((logits > logits[target]).sum().item()) + 1


def _target_rows(example, instance, state, tokenizer):
    true = tokenizer.encode(example.text, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        true = [tokenizer.bos_token_id, *true]
    for target_position in instance.receiver_span:
        if target_position >= len(true) or state.is_visible[target_position]:
            continue
        yield target_position, int(true[target_position])


def dla_chunk(model, tokenizer, examples, *, seed: int, timestep: int, locks, controls):
    rows = []
    layers = sorted({lock.layer for lock in locks.locks.values()} | {item[0] for item in controls.values()})
    for example in examples:
        state = state_at_time(model, tokenizer, example.text, timestep, 64, seed, True)
        with capture_projection_inputs(model, layers) as (captures, metadata):
            model.forward_attentions(state.input_ids, output_hidden_states=True)
        for instance in example.relations:
            selected = locks.resolve(instance.relation)
            candidates = (
                ("selected_relation_head", selected.layer, selected.head),
                ("matched_low_relation_head", *controls[instance.relation]),
            )
            for control_kind, layer, head in candidates:
                if len(captures[layer]) != 1:
                    raise RuntimeError("output projection was not captured exactly once")
                captured = captures[layer][0]
                contribution = projection_head_slice(
                    captured,
                    metadata[layer].weight,
                    head=head,
                    number_of_heads=metadata[layer].number_of_heads,
                )
                query = instance.attender_span[-1]
                contribution_logits = model.get_lm_head()(
                    model.get_final_norm()(contribution[:, query : query + 1])
                )[0, 0].float()
                for target_position, target in _target_rows(
                    example, instance, state, tokenizer
                ):
                    rank = _rank(contribution_logits, target)
                    rows.append(
                        {
                            **instance_metadata(example, instance, "test"),
                            "seed": seed,
                            "timestep": timestep,
                            "normalized_progress": timestep / 63,
                            "layer": layer,
                            "head": head,
                            "control_kind": control_kind,
                            "query_position": query,
                            "target_position": target_position,
                            "target_token_id": target,
                            "target_token": tokenizer.decode([target]),
                            "target_upos": instance.receiver_upos,
                            "target_logit_support": float(contribution_logits[target]),
                            "target_rank": rank,
                            "vocabulary_percentile": 1.0
                            - (rank - 1) / max(len(contribution_logits) - 1, 1),
                            "projection_module": metadata[layer].module_path,
                            "projection_input_shape": list(captured.shape),
                            "projection_weight_shape": list(metadata[layer].weight.shape),
                            "number_of_heads": metadata[layer].number_of_heads,
                            "decomposition": "exact_o_proj_input_slice_then_final_norm_unembedding",
                        }
                    )
    return pd.DataFrame(rows)


def run_dla(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    source_locks: PaperLockSet,
    **_unused: Any,
):
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    controls = matched_low_relation_controls(source_locks)
    store = SentenceCheckpointStore(run_dir)
    frames = []
    for seed in cfg.experiment.seeds:
        for timestep in cfg.experiment.settings["diagnostic_timesteps"]:
            heads = source_locks.heads | set(controls.values())
            identity = CheckpointIdentity(
                stage="paper-direct-logit-attribution",
                seed=seed,
                normalized_progress=timestep / 63,
                timestep=timestep,
                heads=tuple(sorted(heads)),
            )
            frames.append(
                store.run(
                    examples,
                    identity,
                    lambda chunk, _start, current_seed=seed, current_timestep=timestep: dla_chunk(
                        model,
                        tokenizer,
                        chunk,
                        seed=current_seed,
                        timestep=current_timestep,
                        locks=source_locks,
                        controls=controls,
                    ),
                )
            )
    raw = pd.concat(frames, ignore_index=True)
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    keys = ["seed", "relation", "target_upos", "control_kind", "layer", "head", "timestep"]
    per_seed = raw.groupby(keys, as_index=False).agg(
        target_logit_support=("target_logit_support", "mean"),
        target_rank=("target_rank", "mean"),
        vocabulary_percentile=("vocabulary_percentile", "mean"),
        raw_denominator=("target_token_id", "size"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(keys[1:], as_index=False).agg(
        target_logit_support_mean=("target_logit_support", "mean"),
        target_logit_support_std=("target_logit_support", "std"),
        target_rank_mean=("target_rank", "mean"),
        vocabulary_percentile_mean=("vocabulary_percentile", "mean"),
        raw_denominator=("raw_denominator", "sum"),
        n_seeds=("seed", "nunique"),
    ).to_csv(run_dir / "metrics.csv", index=False)
    relation_selectivity = per_seed.pivot_table(
        index=["seed", "relation", "timestep"],
        columns="control_kind",
        values="target_logit_support",
    ).reset_index()
    relation_selectivity["selected_minus_control_support"] = (
        relation_selectivity["selected_relation_head"]
        - relation_selectivity["matched_low_relation_head"]
    )
    relation_selectivity.to_csv(run_dir / "relation_type_selectivity.csv", index=False)
    write_resolved_selection_locks(run_dir, source_locks)
    return {
        "development_used": False,
        "exact_decomposition": True,
        "approximation_used": False,
        "diagnostic_timesteps": cfg.experiment.settings["diagnostic_timesteps"],
        "seeds": list(cfg.experiment.seeds),
        "selected_head_controls": "matched_low_relation_heads",
        "test_sentences": len(examples),
    }


def _logit_metrics(logits: torch.Tensor, query: int, target: int):
    values = logits[0, query].float()
    probability = values.softmax(dim=-1)
    rank = _rank(values, target)
    return float(values[target]), float(probability[target]), rank, int(rank == 1)


def _pos_control_pairs() -> list[tuple[str, int, int]]:
    configured = os.environ.get("DLMREL_POS_HEAD_RANKINGS")
    if not configured:
        return []
    source = Path(configured)
    run_dir = source if source.is_dir() else source.parent
    rankings_path = source / "pos_head_rankings.csv" if source.is_dir() else source
    mapping_path = run_dir / "relative_depth_mapping.csv"
    if not rankings_path.is_file() or not mapping_path.is_file():
        raise ArtifactError("DLMREL_POS_HEAD_RANKINGS does not identify a complete POS run")
    rankings = pd.read_csv(rankings_path)
    mapping = pd.read_csv(mapping_path).set_index("relative_label")
    pairs = []
    for label, group in rankings.groupby("relative_label", observed=True):
        averaged = group.groupby("feature_kind", as_index=False)["accuracy_mean"].mean()
        ordered = averaged.sort_values(["accuracy_mean", "feature_kind"], ascending=[False, True])
        layer = int(mapping.loc[label, "actual_layer_index"])
        top = int(str(ordered.iloc[0].feature_kind).removeprefix("head_"))
        low = int(str(ordered.iloc[-1].feature_kind).removeprefix("head_"))
        pairs.extend(
            (
                (f"most_pos_decodable_{label}", layer, top),
                (f"lower_pos_decoding_{label}", layer, low),
            )
        )
    return pairs


def ablation_chunk(model, tokenizer, examples, *, seed, timestep, locks, controls, pos_pairs):
    rows = []
    for example in examples:
        state = state_at_time(model, tokenizer, example.text, timestep, 64, seed, True)
        baseline_logits, _attentions = model.forward_attentions(state.input_ids)
        if baseline_logits is None:
            raise RuntimeError("causal ablation requires final logits")
        for instance in example.relations:
            selected = locks.resolve(instance.relation)
            interventions = [
                ("selected_relation_head", selected.layer, selected.head),
                ("matched_low_relation_head", *controls[instance.relation]),
                *pos_pairs,
            ]
            query = instance.attender_span[-1]
            for control_kind, layer, head in interventions:
                with capture_or_ablate_projection(
                    model, layer, ablate_head=head
                ) as (captured, metadata):
                    ablated_logits, _ablated_attentions = model.forward_attentions(state.input_ids)
                if len(captured) != 1 or ablated_logits is None:
                    raise RuntimeError("single-head intervention did not execute exactly once")
                for target_position, target in _target_rows(
                    example, instance, state, tokenizer
                ):
                    baseline = _logit_metrics(baseline_logits, query, target)
                    ablated = _logit_metrics(ablated_logits, query, target)
                    rows.append(
                        {
                            **instance_metadata(example, instance, "test"),
                            "seed": seed,
                            "timestep": timestep,
                            "normalized_progress": timestep / 63,
                            "layer": layer,
                            "head": head,
                            "control_kind": control_kind,
                            "query_position": query,
                            "target_position": target_position,
                            "target_token_id": target,
                            "target_token": tokenizer.decode([target]),
                            "target_upos": instance.receiver_upos,
                            "baseline_target_logit": baseline[0],
                            "ablated_target_logit": ablated[0],
                            "target_logit_change": ablated[0] - baseline[0],
                            "baseline_target_probability": baseline[1],
                            "ablated_target_probability": ablated[1],
                            "target_probability_change": ablated[1] - baseline[1],
                            "baseline_target_rank": baseline[2],
                            "ablated_target_rank": ablated[2],
                            "target_rank_change": ablated[2] - baseline[2],
                            "baseline_top1": baseline[3],
                            "ablated_top1": ablated[3],
                            "top1_decodability_change": ablated[3] - baseline[3],
                            "token_type_support_change": ablated[0] - baseline[0],
                            "projection_module": metadata.module_path,
                            "intervention": "zero_exact_requested_o_proj_input_head_slice",
                        }
                    )
    return pd.DataFrame(rows)


def run_ablation(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    source_locks: PaperLockSet,
    **_unused: Any,
):
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    controls = matched_low_relation_controls(source_locks)
    pos_pairs = _pos_control_pairs()
    store = SentenceCheckpointStore(run_dir)
    frames = []
    all_heads = source_locks.heads | set(controls.values()) | {
        (layer, head) for _kind, layer, head in pos_pairs
    }
    for seed in cfg.experiment.seeds:
        for timestep in cfg.experiment.settings["diagnostic_timesteps"]:
            identity = CheckpointIdentity(
                stage="paper-matched-relation-head-ablation",
                seed=seed,
                normalized_progress=timestep / 63,
                timestep=timestep,
                heads=tuple(sorted(all_heads)),
            )
            frames.append(
                store.run(
                    examples,
                    identity,
                    lambda chunk, _start, current_seed=seed, current_timestep=timestep: ablation_chunk(
                        model,
                        tokenizer,
                        chunk,
                        seed=current_seed,
                        timestep=current_timestep,
                        locks=source_locks,
                        controls=controls,
                        pos_pairs=pos_pairs,
                    ),
                )
            )
    raw = pd.concat(frames, ignore_index=True)
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    keys = ["seed", "relation", "target_upos", "control_kind", "layer", "head", "timestep"]
    per_seed = raw.groupby(keys, as_index=False).agg(
        target_logit_change=("target_logit_change", "mean"),
        target_probability_change=("target_probability_change", "mean"),
        target_rank_change=("target_rank_change", "mean"),
        top1_decodability_change=("top1_decodability_change", "mean"),
        token_type_support_change=("token_type_support_change", "mean"),
        raw_denominator=("target_token_id", "size"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    aggregations = {
        column: (column, "mean")
        for column in (
            "target_logit_change",
            "target_probability_change",
            "target_rank_change",
            "top1_decodability_change",
            "token_type_support_change",
        )
    }
    per_seed.groupby(keys[1:], as_index=False).agg(
        **aggregations,
        raw_denominator=("raw_denominator", "sum"),
        n_seeds=("seed", "nunique"),
    ).to_csv(run_dir / "metrics.csv", index=False)
    write_resolved_selection_locks(run_dir, source_locks)
    return {
        "development_used": False,
        "causal_intervention": True,
        "intervention": "zero_exact_requested_o_proj_input_head_slice",
        "diagnostic_timesteps": cfg.experiment.settings["diagnostic_timesteps"],
        "seeds": list(cfg.experiment.seeds),
        "pos_head_ablation_status": (
            "completed" if pos_pairs else "blocked_until_exact_stanford_pos_probe_results_are_supplied"
        ),
        "pos_head_rankings_environment": "DLMREL_POS_HEAD_RANKINGS",
        "test_sentences": len(examples),
    }

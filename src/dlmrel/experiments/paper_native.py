"""Native eventual-token depth probes and prediction-before-unmasking timing."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..artifacts import ArtifactError, canonical_hash
from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..models.base import NativeTrajectory
from ..paper_protocol import map_relative_depths, prediction_source_index, timing_record
from .shared import write_frames


@dataclass(frozen=True)
class PromptExample:
    sentence_id: str
    task: str
    prompt: str


def load_prompt_manifest(path: str | Path) -> tuple[list[PromptExample], str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = raw.get("prompts")
    if raw.get("schema_version") != "dlmrel-paper-prompts-v1" or not isinstance(prompts, list):
        raise ArtifactError("native prompt manifest is incompatible")
    examples = [PromptExample(str(row["id"]), str(row["task"]), str(row["prompt"])) for row in prompts]
    if len(examples) != 24 or CounterLike(item.task for item in examples) != {
        "reasoning": 12,
        "creative": 12,
    }:
        raise ArtifactError("paper prompt manifest must contain 12 reasoning and 12 creative prompts")
    return examples, canonical_hash(raw)


def CounterLike(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _trajectory_row(example: PromptExample, seed: int, trajectory: NativeTrajectory) -> dict[str, Any]:
    if len(trajectory.pre_forward_ids) != 64 or len(trajectory.argmax_ids) != 64:
        raise ArtifactError("native adapter did not return all 64 trajectory steps")
    return {
        "sentence_id": example.sentence_id,
        "prompt_id": example.sentence_id,
        "task": example.task,
        "prompt": example.prompt,
        "seed": seed,
        "prefix_length": trajectory.prefix_length,
        "pre_forward_ids": [step.tolist() for step in trajectory.pre_forward_ids],
        "argmax_ids": [step.tolist() for step in trajectory.argmax_ids],
        "final_ids": trajectory.final_ids.tolist(),
        "trajectory_metadata": trajectory.metadata,
    }


def generate_trajectory_chunk(model, examples, *, seed: int, settings: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for example in examples:
        trajectory = model.native_trajectory(
            example.prompt,
            seed=seed,
            steps=64,
            generation_length=int(settings["generation_length"]),
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
        )
        rows.append(_trajectory_row(example, seed, trajectory))
    return pd.DataFrame(rows)


def generate_trajectories(model, cfg: RunConfig, run_dir: Path):
    settings = cfg.experiment.settings
    prompts, prompt_hash = load_prompt_manifest(settings["prompt_manifest"])
    store = SentenceCheckpointStore(run_dir)
    frames = []
    for seed in cfg.experiment.seeds:
        identity = CheckpointIdentity(
            stage=f"{cfg.experiment.id}-native-trajectories-{prompt_hash[:12]}",
            seed=seed,
            normalized_progress=-1.0,
            timestep=-1,
        )
        frames.append(
            store.run(
                prompts,
                identity,
                lambda chunk, _start, current_seed=seed: generate_trajectory_chunk(
                    model, chunk, seed=current_seed, settings=settings
                ),
            )
        )
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories.sort_values(["task", "prompt_id", "seed"], inplace=True, kind="mergesort")
    trajectories.to_parquet(run_dir / "native_trajectories.parquet", index=False)
    return trajectories, prompts, prompt_hash


def _native_depth_mapping(model, input_ids: torch.Tensor, settings):
    _logits, attentions, _hidden = model.forward_attentions(input_ids, output_hidden_states=True)
    return map_relative_depths(len(attentions), settings["relative_depths"])


def final_token_rows(model, row, depths, *, collect_probe_features: bool):
    final_ids = torch.tensor(row.final_ids, dtype=torch.long, device=model.device)
    mask_id = int(model.tokenizer.mask_token_id)
    result, features = [], []
    for timestep, state_ids in enumerate(row.pre_forward_ids):
        state = torch.tensor([state_ids], dtype=torch.long, device=model.device)
        _logits, _attentions, hidden_states = model.forward_attentions(
            state, output_hidden_states=True
        )
        for depth in depths:
            layer = int(depth["actual_layer_index"])
            hidden = hidden_states[min(layer + 1, len(hidden_states) - 1)]
            transformed = model.get_final_norm()(hidden)
            logits = model.get_lm_head()(transformed)[0].float()
            for position in range(int(row.prefix_length), len(state_ids)):
                if int(state_ids[position]) != mask_id:
                    continue
                try:
                    source = prediction_source_index(position, int(model.prediction_offset))
                except ValueError:
                    continue
                target = int(final_ids[position])
                position_logits = logits[source]
                target_value = position_logits[target]
                rank = int((position_logits > target_value).sum().item()) + 1
                evidence = {
                    "sentence_id": row.prompt_id,
                    "prompt_id": row.prompt_id,
                    "task": row.task,
                    "seed": int(row.seed),
                    "timestep": timestep,
                    "normalized_progress": timestep / 63,
                    **depth,
                    "target_position": position,
                    "prediction_source_position": source,
                    "prediction_offset": int(model.prediction_offset),
                    "target_token_id": target,
                    "top1": int(rank == 1),
                    "top5": int(rank <= 5),
                    "rank": rank,
                    "mrr": 1.0 / rank,
                    "target_logit": float(target_value),
                    "position_was_masked": True,
                    "target_is_eventual_generated_token": True,
                }
                result.append(evidence)
                if collect_probe_features:
                    features.append(
                        {
                            **{key: evidence[key] for key in (
                                "sentence_id",
                                "prompt_id",
                                "task",
                                "seed",
                                "timestep",
                                "relative_label",
                                "target_token_id",
                            )},
                            "feature": hidden[0, source].detach().float().cpu().tolist(),
                        }
                    )
    return pd.DataFrame(result), pd.DataFrame(features)


def _trained_probe_metrics(features: pd.DataFrame, settings, *, vocab_size: int, device: str):
    rows = []
    max_examples = int(settings["probe_max_examples"])
    train_ids = {
        f"reasoning_{index}" for index in range(8)
    } | {f"creative_{index}" for index in range(8)}
    for depth, group in features.groupby("relative_label", observed=True):
        train = group[group["prompt_id"].isin(train_ids)].head(max_examples)
        test = group[~group["prompt_id"].isin(train_ids)]
        if train.empty or test.empty:
            raise ArtifactError("trained final-token probe lacks a prompt-held-out split")
        train_x = torch.tensor(np.stack(train["feature"]), dtype=torch.float32, device=device)
        train_y = torch.tensor(train["target_token_id"].to_numpy(), dtype=torch.long, device=device)
        test_x = torch.tensor(np.stack(test["feature"]), dtype=torch.float32, device=device)
        test_y = torch.tensor(test["target_token_id"].to_numpy(), dtype=torch.long, device=device)
        classifier = torch.nn.Linear(train_x.shape[1], vocab_size, device=device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=float(settings["probe_learning_rate"]))
        batch_size = int(settings["probe_batch_size"])
        generator = torch.Generator(device="cpu").manual_seed(42)
        for _epoch in range(int(settings["probe_epochs"])):
            order = torch.randperm(len(train_x), generator=generator).to(train_x.device)
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size]
                loss = torch.nn.functional.cross_entropy(classifier(train_x[selected]), train_y[selected])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            logits = classifier(test_x)
            ranks = (logits > logits.gather(1, test_y[:, None])).sum(dim=1) + 1
        rows.append(
            {
                "relative_label": depth,
                "top1": float((ranks == 1).float().mean()),
                "top5": float((ranks <= 5).float().mean()),
                "mrr": float((1.0 / ranks.float()).mean()),
                "n_train": len(train),
                "n_held_out": len(test),
                "prompt_split": "reasoning_0..7+creative_0..7_train_remaining_held_out",
            }
        )
        del classifier, optimizer, train_x, train_y, test_x, test_y
    return pd.DataFrame(rows)


def run_final_token(model, tokenizer, cfg: RunConfig, run_dir: Path, **_unused: Any):
    trajectories, prompts, prompt_hash = generate_trajectories(model, cfg, run_dir)
    first = torch.tensor(
        [trajectories.iloc[0]["pre_forward_ids"][0]], dtype=torch.long, device=model.device
    )
    depths = _native_depth_mapping(model, first, cfg.experiment.settings)
    pd.DataFrame(depths).to_csv(run_dir / "relative_depth_mapping.csv", index=False)
    evidence_frames, probe_frames = [], []
    collect_probe = bool(cfg.experiment.settings.get("trained_probe"))
    for row in trajectories.itertuples(index=False):
        evidence, features = final_token_rows(
            model, row, depths, collect_probe_features=collect_probe
        )
        evidence_frames.append(evidence)
        if collect_probe:
            probe_frames.append(features)
    raw = pd.concat(evidence_frames, ignore_index=True)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    keys = ["seed", "task", "timestep", "normalized_progress", "relative_label"]
    per_seed = raw.groupby(keys, as_index=False).agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        mrr=("mrr", "mean"),
        mean_rank=("rank", "mean"),
        raw_denominator=("rank", "size"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(keys[1:], as_index=False).agg(
        top1_mean=("top1", "mean"),
        top1_std=("top1", "std"),
        top5_mean=("top5", "mean"),
        mrr_mean=("mrr", "mean"),
        mean_rank=("mean_rank", "mean"),
        raw_denominator=("raw_denominator", "sum"),
        n_seeds=("seed", "nunique"),
    ).to_csv(run_dir / "metrics.csv", index=False)
    if collect_probe:
        probe_features = pd.concat(probe_frames, ignore_index=True)
        probe_features.to_parquet(run_dir / "trained_probe_features.parquet", index=False)
        probe_metrics = _trained_probe_metrics(
            probe_features,
            cfg.experiment.settings,
            vocab_size=model.get_lm_head().out_features,
            device=model.device,
        )
        probe_metrics.to_csv(run_dir / "trained_probe_metrics.csv", index=False)
    return {
        "native_generation": True,
        "teacher_forced_gold_targets": False,
        "target": "eventual_generated_token",
        "masked_positions_only": True,
        "timesteps": list(range(64)),
        "seeds": list(cfg.experiment.seeds),
        "prompt_manifest_hash": prompt_hash,
        "prompt_count": len(prompts),
        "prompt_strata": {"reasoning": 12, "creative": 12},
        "prediction_offset": int(model.prediction_offset),
        "relative_depths": depths,
        "trained_probe": collect_probe,
    }


STOPWORDS = set(
    "the a an and or but if then of to in on at for with as is are was were be been by this "
    "that these those it its he she they we you i".split()
)


def classify_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False).strip()
    if not text:
        return "whitespace"
    if all(character in string.punctuation for character in text):
        return "punctuation"
    if any(character.isdigit() for character in text):
        return "number"
    if text.lower() in STOPWORDS:
        return "function_word"
    return "content_word"


def timing_rows(tokenizer, row, *, mask_token_id: int):
    final_ids = [int(value) for value in row.final_ids]
    argmax = [[int(value) for value in step] for step in row.argmax_ids]
    states = [[int(value) for value in step] for step in row.pre_forward_ids]
    tokens, curve = [], []
    positions = range(int(row.prefix_length), len(final_ids))
    count = len(final_ids) - int(row.prefix_length)
    for timestep in range(64):
        correct = sum(argmax[timestep][position] == final_ids[position] for position in positions)
        masked = sum(states[timestep][position] == mask_token_id for position in positions)
        curve.append(
            {
                "sentence_id": row.prompt_id,
                "prompt_id": row.prompt_id,
                "task": row.task,
                "seed": int(row.seed),
                "timestep": timestep,
                "normalized_progress": timestep / 63,
                "correct_final_token_fraction": correct / count,
                "masked_fraction": masked / count,
                "raw_denominator": count,
            }
        )
    for position in positions:
        timing = timing_record(
            argmax, states, final_ids, mask_token_id=mask_token_id, position=position
        )
        unmask = timing["unmask_time"]
        match_at_unmask = (
            None if unmask is None else argmax[int(unmask)][position] == final_ids[position]
        )
        tokens.append(
            {
                "sentence_id": row.prompt_id,
                "prompt_id": row.prompt_id,
                "task": row.task,
                "seed": int(row.seed),
                "position": position,
                "final_token_id": final_ids[position],
                "final_token": tokenizer.decode([final_ids[position]], skip_special_tokens=False),
                "token_class": classify_token(tokenizer, final_ids[position]),
                "argmax_matches_at_unmask": match_at_unmask,
                **timing,
            }
        )
    return pd.DataFrame(tokens), pd.DataFrame(curve)


def run_timing(model, tokenizer, cfg: RunConfig, run_dir: Path, **_unused: Any):
    trajectories, prompts, prompt_hash = generate_trajectories(model, cfg, run_dir)
    token_frames, curve_frames = [], []
    for row in trajectories.itertuples(index=False):
        tokens, curve = timing_rows(
            tokenizer, row, mask_token_id=int(tokenizer.mask_token_id)
        )
        token_frames.append(tokens)
        curve_frames.append(curve)
    raw = pd.concat(token_frames, ignore_index=True)
    curve = pd.concat(curve_frames, ignore_index=True)
    write_frames(run_dir, raw=raw, exclusions=pd.DataFrame())
    curve.to_csv(run_dir / "refinement_curve.csv", index=False)
    valid = raw.dropna(subset=["found_time_minus_unmask_time"])
    keys = ["seed", "task", "token_class"]
    per_seed = valid.groupby(keys, as_index=False).agg(
        mean_found_time=("found_time", "mean"),
        mean_unmask_time=("unmask_time", "mean"),
        mean_delta=("found_time_minus_unmask_time", "mean"),
        mean_lead_steps=("lead_steps", "mean"),
        predicted_before_unmasking=("predicted_before_unmasking", "mean"),
        predicted_exactly_at_unmasking=("predicted_exactly_at_unmasking", "mean"),
        unmasking_argmax_match_rate=("argmax_matches_at_unmask", "mean"),
        raw_denominator=("position", "size"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(keys[1:], as_index=False).agg(
        mean_delta=("mean_delta", "mean"),
        delta_seed_std=("mean_delta", "std"),
        mean_lead_steps=("mean_lead_steps", "mean"),
        predicted_before_unmasking=("predicted_before_unmasking", "mean"),
        predicted_exactly_at_unmasking=("predicted_exactly_at_unmasking", "mean"),
        unmasking_argmax_match_rate=("unmasking_argmax_match_rate", "mean"),
        raw_denominator=("raw_denominator", "sum"),
        n_seeds=("seed", "nunique"),
    ).to_csv(run_dir / "metrics.csv", index=False)
    histogram = valid.groupby(
        ["task", "found_time_minus_unmask_time"], as_index=False
    ).agg(token_count=("position", "size"))
    histogram.to_csv(run_dir / "timing_histogram.csv", index=False)
    curve.groupby(["seed", "task", "timestep"], as_index=False).agg(
        correct_final_token_fraction=("correct_final_token_fraction", "mean"),
        masked_fraction=("masked_fraction", "mean"),
        raw_denominator=("raw_denominator", "sum"),
    ).to_csv(run_dir / "per_seed_refinement_curve.csv", index=False)
    return {
        "native_generation": True,
        "pre_forward_history": True,
        "timesteps": list(range(64)),
        "seeds": list(cfg.experiment.seeds),
        "prompt_manifest_hash": prompt_hash,
        "prompt_count": len(prompts),
        "prompt_strata": {"reasoning": 12, "creative": 12},
        "temperature": cfg.experiment.settings["temperature"],
        "top_p": cfg.experiment.settings["top_p"],
        "reveal_policy": cfg.experiment.settings["reveal_policy"],
        "prediction_offset": int(model.prediction_offset),
    }

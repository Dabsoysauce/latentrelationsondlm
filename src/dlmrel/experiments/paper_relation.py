"""Corrected old-code relation-head selection, time curves, and transfer."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import ArtifactError, canonical_hash, scientific_configuration, selection_source_hash
from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RELATION_NAMES, RunConfig
from ..controls import receiver_controls
from ..data import load_manifest_examples
from ..diffusion import (
    attentions_at_time,
    attentions_for_state,
    receiver_predictions,
    teacher_forced_trajectory,
)
from ..paper_protocol import (
    PaperLockSet,
    choose_selection_winners,
    load_selection_bundle,
    paper_visibility_group,
    receiver_is_correct,
    write_resolved_selection_locks,
    write_selection_bundle,
)
from .shared import aggregate_head_scores, instance_metadata, write_frames


def _code_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _word_for_source(example, source_index: int) -> int | None:
    for word_index, span in example.word_to_tokens.items():
        if source_index in span:
            return int(word_index)
    return None


def _score_state(
    example,
    attentions,
    is_visible: list[bool],
    *,
    role: str,
    seed: int,
    timestep: int,
    available_heads: set[tuple[int, int]] | None,
    locks: PaperLockSet | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_heads = available_heads or {
        (layer, head)
        for layer, attention in enumerate(attentions)
        for head in range(attention.shape[1])
    }
    candidate_words = sorted(example.word_to_tokens)
    for instance in example.relations:
        relation_heads = all_heads
        if locks is not None:
            lock = locks.resolve(instance.relation)
            relation_heads = {(lock.layer, lock.head)}
        controls = receiver_controls(
            example,
            instance,
            [word for word in candidate_words if word != instance.attender_word_idx],
            seed=seed,
        )
        for layer in sorted({candidate[0] for candidate in relation_heads}):
            predictions = receiver_predictions(
                attentions,
                layer,
                instance.attender_span,
                attender_token="last",
                exclude_bos=True,
                exclude_self=True,
            )
            for head in sorted(
                candidate_head
                for candidate_layer, candidate_head in relation_heads
                if candidate_layer == layer
            ):
                source = int(predictions[head])
                rows.append(
                    {
                        **instance_metadata(example, instance, role),
                        "seed": seed,
                        "timestep": timestep,
                        "normalized_progress": timestep / 63,
                        "visibility": paper_visibility_group(
                            is_visible, instance.attender_span, instance.receiver_span
                        ),
                        "layer": layer,
                        "head": head,
                        "predicted_source_subtoken": source,
                        "predicted_word_idx": _word_for_source(example, source),
                        "gold_receiver_word_idx": instance.receiver_word_idx,
                        "correct": int(receiver_is_correct(source, instance.receiver_span)),
                        "gold_attention_max": float(
                            attentions[layer][0, head, instance.attender_span[-1], instance.receiver_span]
                            .float()
                            .max()
                        ),
                        "receiver_method": "last_attender_subtoken_single_source_argmax",
                        **controls,
                    }
                )
    return rows


def score_fully_visible_chunk(
    model,
    tokenizer,
    examples,
    *,
    role: str,
    heads: set[tuple[int, int]] | None = None,
    locks: PaperLockSet | None = None,
) -> pd.DataFrame:
    rows = []
    for example in examples:
        attentions, state = attentions_at_time(
            model,
            tokenizer,
            example.text,
            diffusion_time=63,
            steps=64,
            seed=42,
            include_bos=True,
        )
        rows.extend(
            _score_state(
                example,
                attentions,
                state.is_visible,
                role=role,
                seed=42,
                timestep=63,
                available_heads=heads,
                locks=locks,
            )
        )
    return pd.DataFrame(rows)


def _test_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    controls = [
        column
        for column in (
            "nearest_correct",
            "uniform_correct",
            "previous_correct",
            "next_correct",
            "oracle_pos_correct",
            "wrong_same_pos_correct",
        )
        if column in rows
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "numerator": ("correct", "sum"),
        "denominator": ("correct", "size"),
        "accuracy": ("correct", "mean"),
    }
    aggregations.update({f"{column}_accuracy": (column, "mean") for column in controls})
    return rows.groupby(["relation", "layer", "head"], as_index=False).agg(**aggregations)


def run_selection_and_test(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict[str, str],
    **_unused: Any,
) -> dict[str, Any]:
    """Publish six selection-only locks before opening held-out test data."""
    if set(manifest_hashes) != {"select", "test"}:
        raise ArtifactError("corrected relation selection identity must contain select/test only")
    selection, selection_exclusions = load_manifest_examples(cfg, tokenizer, "select")
    store = SentenceCheckpointStore(run_dir)
    identity = CheckpointIdentity(
        stage="paper-selection-fully-visible-all-heads",
        seed=42,
        normalized_progress=1.0,
        timestep=63,
    )
    selection_rows = store.run(
        selection,
        identity,
        lambda chunk, _start: score_fully_visible_chunk(
            model, tokenizer, chunk, role="select"
        ),
    )
    scores = aggregate_head_scores(selection_rows)
    winners = choose_selection_winners(scores)
    scores.to_csv(run_dir / "selection_all_head_scores.csv", index=False)
    selection_rows.to_parquet(run_dir / "selection_instances.parquet", index=False)
    winners.to_csv(run_dir / "selection_winners.csv", index=False)
    scientific = scientific_configuration(cfg.to_dict())
    locks = write_selection_bundle(
        run_dir / "selection-locks",
        winners,
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        tokenizer_revision=cfg.model.tokenizer_revision,
        dataset_id=cfg.dataset.id,
        selection_manifest_hash=manifest_hashes["select"],
        config_hash=canonical_hash(scientific),
        code_hash=_code_hash(),
    )

    # Scientific firewall: no test manifest is opened above this line.
    test, test_exclusions = load_manifest_examples(cfg, tokenizer, "test")
    test_identity = CheckpointIdentity(
        stage="paper-test-frozen-relation-heads",
        seed=42,
        normalized_progress=1.0,
        timestep=63,
        heads=tuple(sorted(locks.heads)),
    )
    test_rows = store.run(
        test,
        test_identity,
        lambda chunk, _start: score_fully_visible_chunk(
            model, tokenizer, chunk, role="test", heads=locks.heads, locks=locks
        ),
    )
    _validate_locked_rows(test_rows, locks)
    exclusions = pd.concat([selection_exclusions, test_exclusions], ignore_index=True)
    write_frames(run_dir, raw=test_rows, exclusions=exclusions)
    metrics = _test_metrics(test_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    metrics.assign(seed=42, deterministic=True).to_csv(
        run_dir / "per_seed_metrics.csv", index=False
    )
    write_resolved_selection_locks(run_dir, locks)
    return {
        "selection_roles": ["select"],
        "evaluation_roles": ["test"],
        "development_used": False,
        "permutations_used": False,
        "holm_correction_used": False,
        "fully_visible_selection_timestep": 63,
        "selection_lock_dir": str((run_dir / "selection-locks").resolve()),
        "selection_lock_hash": selection_source_hash(run_dir / "selection-locks"),
        "relations_locked": list(RELATION_NAMES),
        "selection_sentences": len(selection),
        "test_sentences": len(test),
        "test_heads_scored": "frozen_relation_heads_only",
        "deterministic_replicates": 1,
    }


def _validate_locked_rows(rows: pd.DataFrame, locks: PaperLockSet) -> None:
    for relation, group in rows.groupby("relation", observed=True):
        lock = locks.resolve(str(relation))
        actual = set(group[["layer", "head"]].itertuples(index=False, name=None))
        if actual != {(lock.layer, lock.head)}:
            raise ArtifactError(f"held-out rows violate frozen lock for {relation}")


def score_time_chunk(model, tokenizer, examples, *, seed: int, locks: PaperLockSet) -> pd.DataFrame:
    rows = []
    for example in examples:
        states = teacher_forced_trajectory(
            model, tokenizer, example.text, steps=64, seed=seed, include_bos=True
        )
        for timestep, state in enumerate(states):
            if timestep in {0, 63} and seed != 42:
                continue
            attentions = attentions_for_state(model, state)
            rows.extend(
                _score_state(
                    example,
                    attentions,
                    state.is_visible,
                    role="test",
                    seed=seed,
                    timestep=timestep,
                    available_heads=locks.heads,
                    locks=locks,
                )
            )
    return pd.DataFrame(rows)


def _time_metrics(raw: pd.DataFrame, minimum_masked_instances: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["seed", "treebank", "relation", "layer", "head", "timestep", "visibility"]
    per_seed = raw.groupby(keys, as_index=False).agg(
        numerator=("correct", "sum"),
        denominator=("correct", "size"),
        accuracy=("correct", "mean"),
    )
    per_seed["included_by_masked_cutoff"] = (
        per_seed["visibility"].ne("both_masked")
        | per_seed["denominator"].ge(minimum_masked_instances)
    )
    group = keys[1:]
    metrics = per_seed.groupby(group, as_index=False).agg(
        numerator=("numerator", "sum"),
        denominator=("denominator", "sum"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_seeds=("seed", "nunique"),
        included_by_masked_cutoff=("included_by_masked_cutoff", "all"),
    )
    metrics["normalized_progress"] = metrics["timestep"] / 63
    return per_seed, metrics


def run_time_or_transfer(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    source_locks: PaperLockSet,
    transfer: bool = False,
    **_unused: Any,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    store = SentenceCheckpointStore(run_dir)
    frames = []
    for seed in cfg.experiment.seeds:
        identity = CheckpointIdentity(
            stage="paper-transfer-trajectory" if transfer else "paper-time-trajectory",
            seed=seed,
            normalized_progress=-1.0,
            timestep=-1,
            heads=tuple(sorted(source_locks.heads)),
        )
        frames.append(
            store.run(
                examples,
                identity,
                lambda chunk, _start, current_seed=seed: score_time_chunk(
                    model, tokenizer, chunk, seed=current_seed, locks=source_locks
                ),
            )
        )
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not np.isfinite(raw["correct"].astype(float)).all():
        raise ArtifactError("nonfinite locked-head transfer evidence")
    _validate_locked_rows(raw, source_locks)
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    minimum = int(cfg.experiment.settings.get("minimum_masked_instances", 25))
    per_seed, metrics = _time_metrics(raw, minimum)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    write_resolved_selection_locks(run_dir, source_locks)
    return {
        "development_used": False,
        "reselection_performed": False,
        "source_selection_dataset": "ewt",
        "source_selection_hash": selection_source_hash(source_locks.source),
        "timesteps": list(range(64)),
        "stochastic_seeds": list(cfg.experiment.seeds),
        "deterministic_steps_deduplicated": [0, 63],
        "visibility_groups": ["both_masked", "at_least_one_revealed"],
        "minimum_masked_instances": minimum,
        "test_sentences": len(examples),
        "transfer_extension": transfer,
    }


def load_paper_locks(path: str | Path, cfg: RunConfig) -> PaperLockSet:
    return load_selection_bundle(path, model_id=cfg.model.id, model_revision=cfg.model.revision)

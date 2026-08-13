"""Shared rigorous execution engine used by local, Colab, and Modal runners."""

from __future__ import annotations

import importlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .artifacts import (
    ArtifactError,
    SelectionLock,
    atomic_json,
    canonical_hash,
    write_shard,
)
from .config import RunConfig, TreebankConfig
from .diffusion import (
    attentions_at_time,
    endpoint_visibility,
    receiver_span_scores,
    states_at_time,
    tokenize,
)
from .evaluation.statistics import sentence_clustered_bootstrap
from .relations import Example, build_example
from .selection import create_selection_lock, write_lock_bundle
from .treebank import acquire_split


def model_smoke_report(model, tokenizer, cfg: RunConfig, metadata: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic shape, attention-row, and final-lens checks."""
    ids = tokenizer.encode("The chef cooked dinner.", add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id, *ids]
    input_ids = torch.tensor([ids], device=model.device)
    report: dict[str, Any] = {
        "status": "passed",
        "model": cfg.model.id,
        "checkpoint": cfg.model.name,
        "revision": cfg.model.revision,
        "tokenizer_revision": cfg.model.tokenizer_revision,
        "remote_code_revision": cfg.model.remote_code_revision,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "capabilities": asdict(cfg.model.capabilities),
        "metadata": metadata,
    }
    if cfg.model.capabilities.attentions:
        first = model.forward_attentions(input_ids, output_hidden_states=True)
        second = model.forward_attentions(input_ids, output_hidden_states=True)
        logits, attentions, hidden = first
        second_logits, second_attentions, second_hidden = second
        if logits is None:
            logits = model.get_logits(hidden[-1])
        if second_logits is None:
            second_logits = model.get_logits(second_hidden[-1])
        row_errors = [float((layer.float().sum(dim=-1) - 1).abs().max()) for layer in attentions]
        determinism = max(
            [float((logits.float() - second_logits.float()).abs().max())]
            + [
                float((left.float() - right.float()).abs().max())
                for left, right in zip(attentions, second_attentions, strict=True)
            ]
        )
        final_lens = model.get_lm_head()(hidden[-1])
        report.update(
            {
                "logits_shape": list(logits.shape),
                "hidden_state_shapes": [list(value.shape) for value in hidden],
                "attention_shapes": [list(value.shape) for value in attentions],
                "attention_row_sum_max_error": max(row_errors),
                "determinism_max_abs_error": determinism,
                "final_depth_logit_lens_max_abs_error": float(
                    (logits.float() - final_lens.float()).abs().max()
                ),
            }
        )
        if max(row_errors) > 1e-3:
            raise ArtifactError("attention rows do not sum to one")
        if determinism > 1e-5:
            raise ArtifactError("model is nondeterministic in eval mode")
    else:
        output = model.backbone(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = getattr(output, "logits", None)
        if logits is None:
            logits = model.get_logits(output.hidden_states[-1])
        report.update(
            {
                "logits_shape": list(logits.shape),
                "hidden_state_shapes": [list(value.shape) for value in output.hidden_states],
                "attention_status": "unsupported_pending_parity",
            }
        )
    return report


def load_adapter(cfg: RunConfig):
    """Load exactly one capability-gated adapter with pinned config values."""
    if cfg.model.family == "fake":
        from .models.fake import FakeAdapter

        return FakeAdapter(), None, {"checkpoint": "fake", "revision": "local-v1"}
    module = importlib.import_module(f"dlmrel.models.{cfg.model.family}")
    model, tokenizer, metadata = module.load(asdict(cfg.model))
    declared = asdict(cfg.model.capabilities)
    actual = getattr(model, "capabilities", None)
    if actual is not None and actual.__dict__ != declared:
        raise ArtifactError(f"adapter capabilities differ from config: {actual.__dict__} != {declared}")
    return model, tokenizer, metadata


def load_manifest_examples(cfg: RunConfig, tokenizer, role: str) -> tuple[list[Example], pd.DataFrame]:
    """Apply tokenizer eligibility to, but never replace, a base manifest."""
    from conllu import parse_incr

    manifest_path = Path("data/manifests") / cfg.dataset.id / cfg.dataset.release / f"{role}.csv"
    if not manifest_path.exists():
        raise ArtifactError(f"missing base manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    original_split = {"select": "train", "dev": "dev", "test": "test"}[role]
    if set(manifest["original_split"]) != {original_split}:
        raise ArtifactError(f"{role} manifest violates official {original_split} boundary")
    path = acquire_split(cfg.dataset, original_split, download=False)
    with path.open(encoding="utf-8") as stream:
        sentences = list(parse_incr(stream))
    by_sent_id = {
        str(sentence.metadata.get("sent_id")): sentence
        for sentence in sentences
        if sentence.metadata.get("sent_id") is not None
    }
    legacy = TreebankConfig(
        treebanks=[cfg.dataset.treebank],
        cache_dir=cfg.dataset.cache_dir,
        n_select=cfg.dataset.n_select,
        n_dev=cfg.dataset.n_dev,
        n_test=cfg.dataset.n_test,
        max_seq_len=128,
        min_seq_len=cfg.dataset.min_words,
        skip_multiword=False,
        require_full_alignment=True,
        dedupe_by_text=False,
    )
    examples: list[Example] = []
    exclusions: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        sentence = by_sent_id.get(str(row.sent_id))
        if sentence is None:
            exclusions.append(_exclusion(row, role, "sent_id_not_found"))
            continue
        sentence.metadata["source_treebank"] = cfg.dataset.treebank
        sentence.metadata["source_split"] = original_split
        example = build_example(sentence, tokenizer, legacy, include_bos=True)
        if example is None:
            exclusions.append(_exclusion(row, role, "tokenization_alignment_or_relation_filter"))
            continue
        example.sentence_id = str(row.sentence_id)
        example.language = cfg.dataset.language
        example.original_split = original_split
        example.source = cfg.dataset.treebank
        for instance in example.relations:
            suffix = instance.instance_id.split(":")[-1]
            instance.instance_id = f"{row.sentence_id}:{suffix}"
        examples.append(example)
    return examples, pd.DataFrame(exclusions)


def _exclusion(row, role: str, reason: str) -> dict[str, Any]:
    return {
        "sentence_id": str(row.sentence_id),
        "instance_id": None,
        "role": role,
        "reason": reason,
    }


def score_attention_heads(
    model,
    tokenizer,
    examples: list[Example],
    cfg: RunConfig,
    *,
    role: str,
    heads: set[tuple[int, int]] | None = None,
    normalized_progress: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Score all heads or a frozen subset, retaining per-instance evidence."""
    progress = normalized_progress if normalized_progress is not None else _selection_progress(cfg)
    timestep = round(progress * (cfg.experiment.steps - 1))
    seed = cfg.experiment.seeds[0] if seed is None else seed
    rows: list[dict[str, Any]] = []
    for sentence_index, example in enumerate(examples):
        attentions, state = attentions_at_time(
            model,
            tokenizer,
            example.text,
            diffusion_time=timestep,
            steps=cfg.experiment.steps,
            seed=seed,
            include_bos=True,
        )
        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]
        selected_heads = heads or {(layer, head) for layer in range(n_layers) for head in range(n_heads)}
        candidates = sorted(example.word_to_tokens)
        for instance in example.relations:
            candidate_words = [
                word
                for word in candidates
                if word != instance.attender_word_idx
                and example.word_to_tokens[word]
                and max(example.word_to_tokens[word]) < len(state.is_visible)
            ]
            if instance.receiver_word_idx not in candidate_words:
                continue
            candidate_spans = [example.word_to_tokens[word] for word in candidate_words]
            gold_index = candidate_words.index(instance.receiver_word_idx)
            visibility = endpoint_visibility(state.is_visible, instance.attender_span, instance.receiver_span)
            for layer in sorted({item[0] for item in selected_heads}):
                layer_heads = sorted(
                    head for candidate_layer, head in selected_heads if candidate_layer == layer
                )
                scores = receiver_span_scores(
                    attentions,
                    layer,
                    instance.attender_span,
                    candidate_spans,
                    row_aggregation=cfg.experiment.scoring.attender_rows,
                    span_aggregation=cfg.experiment.scoring.receiver_span,
                    excluded_positions=set(instance.attender_span),
                )
                for head in layer_heads:
                    prediction_index = int(np.argmax(scores[head]))
                    prediction_word = candidate_words[prediction_index]
                    matched_word, match_level = _matched_word(example, instance, candidate_words)
                    controls = _receiver_controls(example, instance, candidate_words, seed=seed)
                    matched_mass = (
                        float(scores[head, candidate_words.index(matched_word)])
                        if matched_word is not None
                        else float("nan")
                    )
                    rows.append(
                        {
                            **_instance_metadata(example, instance, role),
                            "seed": seed,
                            "timestep": timestep,
                            "normalized_progress": progress,
                            "visibility": visibility,
                            "layer": layer,
                            "head": head,
                            "predicted_word_idx": prediction_word,
                            "gold_receiver_word_idx": instance.receiver_word_idx,
                            "correct": int(prediction_word == instance.receiver_word_idx),
                            "gold_attention_mass": float(scores[head, gold_index]),
                            "matched_word_idx": matched_word,
                            "matched_attention_mass": matched_mass,
                            "matched_relaxation_level": match_level,
                            "matched_gold_greater": (
                                float(scores[head, gold_index]) > matched_mass
                                if matched_word is not None
                                else None
                            ),
                            "n_candidate_words": len(candidate_words),
                            **controls,
                        }
                    )
        if (sentence_index + 1) % 100 == 0:
            print(
                f"[rigorous] {role}: {sentence_index + 1}/{len(examples)} sentences",
                flush=True,
            )
    return pd.DataFrame(rows)


def _selection_progress(cfg: RunConfig) -> float:
    visibility = cfg.experiment.scoring.primary_visibility
    if visibility == "both_masked":
        return 0.0
    if visibility == "both_visible":
        return 1.0
    raise ArtifactError("head selection supports preregistered both_masked or both_visible states")


def _instance_metadata(example: Example, instance, role: str) -> dict[str, Any]:
    return {
        "sentence_id": example.sentence_id,
        "instance_id": instance.instance_id,
        "sentence": example.text,
        "treebank": example.source,
        "language": example.language,
        "original_split": example.original_split,
        "role": role,
        "relation": instance.relation,
        "ud_deprel": instance.dep,
        "attender_word_idx": instance.attender_word_idx,
        "receiver_word_idx": instance.receiver_word_idx,
        "attender_text": instance.attender_text,
        "receiver_text": instance.receiver_text,
        "attender_upos": instance.attender_upos,
        "receiver_upos": instance.receiver_upos,
        "attender_span": instance.attender_span,
        "receiver_span": instance.receiver_span,
        "signed_distance": instance.word_distance,
        "direction": "right" if instance.word_distance > 0 else "left",
        "punctuation_between": instance.punctuation_between,
        "clause_depth": instance.clause_depth,
        "embedded_clause": instance.embedded_clause,
        "coordinated": instance.coordinated,
        "relative_clause": instance.relative_clause,
        "passive_voice": instance.passive_voice,
        "intervening_verbs": instance.intervening_verbs,
        "intervening_nouns": instance.intervening_nouns,
        "sentence_length_words": len(example.tokens),
        "sentence_length_subtokens": example.seq_len,
        "attender_bpe_length": len(instance.attender_span),
        "receiver_bpe_length": len(instance.receiver_span),
    }


def _matched_word(example: Example, instance, candidates: list[int]) -> tuple[int | None, int | None]:
    alternatives = [
        word
        for word in candidates
        if word != instance.receiver_word_idx and example.upos[word] == instance.receiver_upos
    ]
    levels = (
        lambda word: (
            np.sign(word - instance.attender_word_idx)
            == np.sign(instance.receiver_word_idx - instance.attender_word_idx)
            and abs(abs(word - instance.attender_word_idx) - abs(instance.word_distance)) <= 1
        ),
        lambda word: (
            np.sign(word - instance.attender_word_idx)
            == np.sign(instance.receiver_word_idx - instance.attender_word_idx)
        ),
        lambda word: True,
    )
    for level, rule in enumerate(levels):
        admitted = [word for word in alternatives if rule(word)]
        if admitted:
            return min(
                admitted,
                key=lambda word: (
                    abs(abs(word - instance.attender_word_idx) - abs(instance.word_distance)),
                    word,
                ),
            ), level
    return None, None


def _receiver_controls(example: Example, instance, candidates: list[int], *, seed: int) -> dict[str, Any]:
    """Receiver controls fixed without attention or aggregate test outcomes."""
    attender = instance.attender_word_idx
    receiver = instance.receiver_word_idx
    stable = sum((index + 1) * ord(char) for index, char in enumerate(instance.instance_id))
    uniform = candidates[int(np.random.default_rng(seed + stable).integers(len(candidates)))]
    nearest = min(candidates, key=lambda word: (abs(word - attender), word))
    previous = attender - 1 if attender - 1 in candidates else None
    following = attender + 1 if attender + 1 in candidates else None
    same_pos = [word for word in candidates if example.upos[word] == instance.receiver_upos]
    oracle_pos = min(same_pos, key=lambda word: (abs(word - attender), word)) if same_pos else None
    wrong_same_pos = [word for word in same_pos if word != receiver]
    wrong_pos = min(wrong_same_pos, key=lambda word: (abs(word - attender), word)) if wrong_same_pos else None
    return {
        "uniform_receiver_word_idx": uniform,
        "uniform_correct": int(uniform == receiver),
        "nearest_receiver_word_idx": nearest,
        "nearest_correct": int(nearest == receiver),
        "previous_receiver_word_idx": previous,
        "previous_correct": int(previous == receiver),
        "next_receiver_word_idx": following,
        "next_correct": int(following == receiver),
        "oracle_pos_receiver_word_idx": oracle_pos,
        "oracle_pos_correct": int(oracle_pos == receiver),
        "wrong_same_pos_word_idx": wrong_pos,
        "wrong_same_pos_correct": int(wrong_pos == receiver),
    }


def _score_over_seeds(
    model,
    tokenizer,
    examples: list[Example],
    cfg: RunConfig,
    *,
    role: str,
    heads: set[tuple[int, int]] | None = None,
    normalized_progress: float | None = None,
    checkpoint_dir: Path | None = None,
    stage: str = "score",
) -> pd.DataFrame:
    frames = []
    progress = normalized_progress if normalized_progress is not None else _selection_progress(cfg)
    head_suffix = "all" if heads is None else "-".join(f"l{layer}h{head}" for layer, head in sorted(heads))
    for seed in cfg.experiment.seeds:
        checkpoint = (
            checkpoint_dir / f"{stage}-seed{seed}-p{progress:.6f}-{head_suffix}.parquet"
            if checkpoint_dir is not None
            else None
        )
        if checkpoint is not None and checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
        else:
            frame = score_attention_heads(
                model,
                tokenizer,
                examples,
                cfg,
                role=role,
                heads=heads,
                normalized_progress=normalized_progress,
                seed=seed,
            )
            if checkpoint is not None:
                _atomic_parquet(checkpoint, frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def aggregate_head_scores(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["relation", "layer", "head", "accuracy", "n_total", "n_correct"])
    return rows.groupby(["relation", "layer", "head"], as_index=False).agg(
        accuracy=("correct", "mean"), n_total=("correct", "size"), n_correct=("correct", "sum")
    )


def run_head_search(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    """Select all heads on select, choose top-K on dev, test one locked head."""
    select_examples, select_exclusions = load_manifest_examples(cfg, tokenizer, "select")
    dev_examples, dev_exclusions = load_manifest_examples(cfg, tokenizer, "dev")
    test_examples, test_exclusions = load_manifest_examples(cfg, tokenizer, "test")
    select_rows = _score_over_seeds(
        model,
        tokenizer,
        select_examples,
        cfg,
        role="select",
        checkpoint_dir=run_dir / "checkpoints",
        stage="select-all-heads",
    )
    select_scores = aggregate_head_scores(select_rows)
    relation = cfg.experiment.scoring.primary_relation
    dev_rows = _score_over_seeds(
        model,
        tokenizer,
        dev_examples,
        cfg,
        role="dev",
        checkpoint_dir=run_dir / "checkpoints",
        stage="dev-all-heads",
    )
    dev_scores = aggregate_head_scores(dev_rows)
    fixed_offset = _fit_relation_offset(select_examples, relation)
    lock, select_candidates, dev_candidates = create_selection_lock(
        select_scores,
        dev_scores,
        relation=relation,
        top_k=cfg.experiment.scoring.top_k,
        track=cfg.track,
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        dataset_id=cfg.dataset.id,
        config_hash=canonical_hash(cfg.to_dict()),
        select_manifest_hash=manifest_hashes["select"],
        dev_manifest_hash=manifest_hashes["dev"],
        frozen_settings={
            "fixed_offset": fixed_offset,
            "row_aggregation": cfg.experiment.scoring.attender_rows,
            "span_aggregation": cfg.experiment.scoring.receiver_span,
            "selection_progress": _selection_progress(cfg),
            "minimum_denominator": 25,
        },
    )
    write_lock_bundle(run_dir, lock, select_candidates, dev_candidates)
    select_scores.to_csv(run_dir / "select_all_head_scores.csv", index=False)
    dev_scores.to_csv(run_dir / "dev_all_head_scores.csv", index=False)
    select_rows.to_parquet(run_dir / "select_instances.parquet", index=False)
    dev_rows.to_parquet(run_dir / "dev_instances.parquet", index=False)
    locked_head = {(lock.layer, lock.head)}
    test_rows = _score_over_seeds(
        model,
        tokenizer,
        test_examples,
        cfg,
        role="test",
        heads=locked_head,
        checkpoint_dir=run_dir / "checkpoints",
        stage="test-locked-head",
    )
    if test_rows[["layer", "head"]].drop_duplicates().shape[0] > 1:
        raise ArtifactError("locked test exposed more than one head")
    _write_frames(
        run_dir,
        raw=test_rows,
        exclusions=pd.concat([select_exclusions, dev_exclusions, test_exclusions], ignore_index=True),
    )
    metrics = _locked_metrics(test_rows, fixed_offset)
    permutation = selection_aware_permutation(
        select_rows,
        dev_rows,
        relation=relation,
        top_k=cfg.experiment.scoring.top_k,
        n_permutations=1000,
        seed=42,
    )
    metrics["selection_permutation_p"] = permutation["p_value"]
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    _structural_slices(test_rows).to_csv(run_dir / "structural_slices.csv", index=False)
    atomic_json(run_dir / "selection_permutation.json", permutation)
    _per_seed(test_rows).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    return {
        "selection_lock": asdict(lock),
        "n_select_sentences": len(select_examples),
        "n_dev_sentences": len(dev_examples),
        "n_test_sentences": len(test_examples),
        "n_test_instances": int(test_rows["instance_id"].nunique()),
        "test_heads_exposed": 1,
    }


def _fit_relation_offset(examples: list[Example], relation: str) -> int | None:
    offsets = [
        instance.word_distance
        for example in examples
        for instance in example.relations
        if instance.relation == relation and instance.word_distance != 0
    ]
    if not offsets:
        return None
    counts = Counter(offsets)
    maximum = max(counts.values())
    return min((offset for offset, count in counts.items() if count == maximum), key=lambda x: (abs(x), x))


def _locked_metrics(rows: pd.DataFrame, fixed_offset: int | None) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    frame["fixed_offset_correct"] = (
        frame["attender_word_idx"] + fixed_offset == frame["receiver_word_idx"]
        if fixed_offset is not None
        else False
    )
    frame["nearest_correct"] = frame.apply(
        lambda row: abs(row.receiver_word_idx - row.attender_word_idx) == 1, axis=1
    )
    grouped = frame.groupby(
        ["treebank", "relation", "layer", "head", "visibility"],
        as_index=False,
    ).agg(
        accuracy=("correct", "mean"),
        n_instances=("correct", "size"),
        fixed_offset_accuracy=("fixed_offset_correct", "mean"),
        nearest_accuracy=("nearest_correct", "mean"),
        uniform_accuracy=("uniform_correct", "mean"),
        previous_accuracy=("previous_correct", "mean"),
        next_accuracy=("next_correct", "mean"),
        oracle_pos_accuracy=("oracle_pos_correct", "mean"),
        wrong_same_pos_accuracy=("wrong_same_pos_correct", "mean"),
        mean_gold_mass=("gold_attention_mass", "mean"),
        mean_matched_mass=("matched_attention_mass", "mean"),
        p_gold_mass_greater=("matched_gold_greater", "mean"),
    )
    keys = ["treebank", "relation", "layer", "head", "visibility"]
    intervals = []
    for identity, group in frame.groupby(keys, observed=True):
        low, high = sentence_clustered_bootstrap(group, value_col="correct", n_boot=2000, seed=42)
        intervals.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return grouped.merge(pd.DataFrame(intervals), on=keys, how="left")


def _per_seed(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby(["seed", "treebank", "relation", "layer", "head", "visibility"], as_index=False).agg(
        accuracy=("correct", "mean"), n_instances=("correct", "size")
    )


def run_locked_transfer(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    source_lock: SelectionLock,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    rows = _score_over_seeds(
        model,
        tokenizer,
        examples,
        cfg,
        role="test",
        heads={(source_lock.layer, source_lock.head)},
        normalized_progress=float(source_lock.frozen_settings["selection_progress"]),
        checkpoint_dir=run_dir / "checkpoints",
        stage="external-test-locked-head",
    )
    source_lock.write_once(run_dir / "selection_lock.json")
    _write_frames(run_dir, raw=rows, exclusions=exclusions)
    metrics = _locked_metrics(rows, source_lock.frozen_settings.get("fixed_offset"))
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    _structural_slices(rows).to_csv(run_dir / "structural_slices.csv", index=False)
    _per_seed(rows).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    return {
        "source_selection_dataset": source_lock.dataset_id,
        "source_selection_hash": canonical_hash(asdict(source_lock)),
        "n_test_sentences": len(examples),
        "n_test_instances": int(rows["instance_id"].nunique()),
        "test_heads_exposed": int(rows[["layer", "head"]].drop_duplicates().shape[0]),
    }


def run_time_curve(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    source_lock: SelectionLock,
) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    frames = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            checkpoint = (
                run_dir
                / "checkpoints"
                / (f"time-curve-seed{seed}-p{progress:.6f}-l{source_lock.layer}h{source_lock.head}.parquet")
            )
            if checkpoint.exists():
                frame = pd.read_parquet(checkpoint)
            else:
                frame = score_attention_heads(
                    model,
                    tokenizer,
                    examples,
                    cfg,
                    role="test",
                    heads={(source_lock.layer, source_lock.head)},
                    normalized_progress=progress,
                    seed=seed,
                )
                _atomic_parquet(checkpoint, frame)
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    source_lock.write_once(run_dir / "selection_lock.json")
    _write_frames(run_dir, raw=raw, exclusions=exclusions)
    per_seed = raw.groupby(
        [
            "seed",
            "treebank",
            "relation",
            "layer",
            "head",
            "timestep",
            "normalized_progress",
            "visibility",
        ],
        as_index=False,
    ).agg(accuracy=("correct", "mean"), n_instances=("correct", "size"))
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics = per_seed.groupby(
        ["treebank", "relation", "layer", "head", "timestep", "normalized_progress", "visibility"],
        as_index=False,
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_instances=("n_instances", "sum"),
        n_seeds=("seed", "nunique"),
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return {"n_rows": len(raw), "n_sentences": len(examples), "test_heads_exposed": 1}


def run_entropy(model, tokenizer, cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    rows = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            timestep = round(progress * (cfg.experiment.steps - 1))
            for example in examples:
                attentions, state = attentions_at_time(
                    model, tokenizer, example.text, timestep, cfg.experiment.steps, seed, True
                )
                for layer, attention in enumerate(attentions):
                    values = attention[0].float()
                    valid_keys = values.shape[-1]
                    probability = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
                    without_special = probability.clone()
                    without_special[:, :, 0] = 0
                    without_special /= without_special.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    entropy_no_special = -(without_special * without_special.clamp_min(1e-12).log()).sum(
                        dim=-1
                    )
                    for head in range(values.shape[0]):
                        rows.append(
                            {
                                "sentence_id": example.sentence_id,
                                "treebank": example.source,
                                "seed": seed,
                                "timestep": timestep,
                                "normalized_progress": progress,
                                "layer": layer,
                                "head": head,
                                "entropy": float(entropy[head].mean()),
                                "entropy_normalized": float(entropy[head].mean() / np.log(valid_keys)),
                                "entropy_no_special": float(entropy_no_special[head].mean()),
                                "bos_sink_mass": float(probability[head, :, 0].mean()),
                                "valid_key_count": valid_keys,
                                "n_masked": state.n_masked,
                            }
                        )
    raw = pd.DataFrame(rows)
    _write_frames(run_dir, raw=raw, exclusions=exclusions)
    per_seed = raw.groupby(
        ["seed", "treebank", "timestep", "normalized_progress", "layer", "head"],
        as_index=False,
    ).mean(numeric_only=True)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics = per_seed.groupby(
        ["treebank", "timestep", "normalized_progress", "layer", "head"], as_index=False
    ).agg(
        entropy_mean=("entropy", "mean"),
        entropy_normalized=("entropy_normalized", "mean"),
        entropy_no_special=("entropy_no_special", "mean"),
        bos_sink_mass=("bos_sink_mass", "mean"),
        n_seeds=("seed", "nunique"),
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return {"n_rows": len(raw), "n_sentences": len(examples)}


def run_logit_lens(model, tokenizer, cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    rows = []
    final_parity_errors = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            timestep = round(progress * (cfg.experiment.steps - 1))
            for example in examples:
                attentions, hidden_states, state = states_at_time(
                    model, tokenizer, example.text, timestep, cfg.experiment.steps, seed, True
                )
                del attentions
                true_ids, _ = tokenize(tokenizer, example.text, state.input_ids.device, True)
                for depth, hidden in enumerate(hidden_states):
                    transformed = (
                        hidden if depth == len(hidden_states) - 1 else model.get_final_norm()(hidden)
                    )
                    logits = model.get_lm_head()(transformed)[0].float()
                    order = logits.argsort(dim=-1, descending=True)
                    for position in range(logits.shape[0]):
                        rank = int((order[position] == true_ids[0, position]).nonzero()[0]) + 1
                        rows.append(
                            {
                                "sentence_id": example.sentence_id,
                                "treebank": example.source,
                                "seed": seed,
                                "timestep": timestep,
                                "normalized_progress": progress,
                                "depth": depth,
                                "position": position,
                                "position_state": "visible" if state.is_visible[position] else "masked",
                                "target_token_id": int(true_ids[0, position]),
                                "top1": int(rank == 1),
                                "top5": int(rank <= 5),
                                "rank": rank,
                                "mrr": 1.0 / rank,
                                "target_logit": float(logits[position, true_ids[0, position]]),
                            }
                        )
                if hidden_states:
                    direct = model.get_logits(hidden_states[-1]).float()
                    lens = model.get_lm_head()(hidden_states[-1]).float()
                    final_parity_errors.append(float((direct - lens).abs().max()))
    raw = pd.DataFrame(rows)
    _write_frames(run_dir, raw=raw, exclusions=exclusions)
    per_seed = raw.groupby(
        ["seed", "treebank", "timestep", "normalized_progress", "depth", "position_state"],
        as_index=False,
    ).agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        mrr=("mrr", "mean"),
        mean_rank=("rank", "mean"),
        n_positions=("rank", "size"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics = per_seed.groupby(
        ["treebank", "timestep", "normalized_progress", "depth", "position_state"],
        as_index=False,
    ).agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        mrr=("mrr", "mean"),
        mean_rank=("mean_rank", "mean"),
        n_positions=("n_positions", "sum"),
        n_seeds=("seed", "nunique"),
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return {
        "n_rows": len(raw),
        "n_sentences": len(examples),
        "final_depth_max_abs_parity_error": max(final_parity_errors, default=float("nan")),
    }


def run_pos_probe(model, tokenizer, cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    """Masked-word POS probe with select fit, dev C choice, frozen test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    collected = {}
    exclusions = []
    for role in ("select", "dev", "test"):
        examples, dropped = load_manifest_examples(cfg, tokenizer, role)
        exclusions.append(dropped)
        collected[role] = _masked_probe_features(model, tokenizer, examples, cfg)
    train_x, train_y, train_groups, train_forms = collected["select"]
    dev_x, dev_y, _, _ = collected["dev"]
    test_x, test_y, test_groups, test_forms = collected["test"]
    scaler = StandardScaler().fit(train_x)
    candidates = []
    for regularization in (0.01, 0.1, 1.0, 10.0):
        classifier = LogisticRegression(C=regularization, max_iter=2000, random_state=42)
        classifier.fit(scaler.transform(train_x), train_y)
        candidates.append(
            (accuracy_score(dev_y, classifier.predict(scaler.transform(dev_x))), regularization)
        )
    _, selected_c = max(candidates, key=lambda item: (item[0], -item[1]))
    classifier = LogisticRegression(C=selected_c, max_iter=2000, random_state=42)
    classifier.fit(scaler.transform(train_x), train_y)
    prediction = classifier.predict(scaler.transform(test_x))
    majority = Counter(train_y).most_common(1)[0][0]
    lexical = _lexical_predictions(train_forms, train_y, test_forms, majority)
    rng = np.random.default_rng(42)
    shuffled_y = train_y.copy()
    rng.shuffle(shuffled_y)
    shuffled = (
        LogisticRegression(C=selected_c, max_iter=2000, random_state=42)
        .fit(scaler.transform(train_x), shuffled_y)
        .predict(scaler.transform(test_x))
    )
    random_train = rng.normal(size=train_x.shape)
    random_test = rng.normal(size=test_x.shape)
    random_feature = (
        LogisticRegression(C=selected_c, max_iter=2000, random_state=42)
        .fit(random_train, train_y)
        .predict(random_test)
    )
    raw = pd.DataFrame(
        {
            "sentence_id": test_groups,
            "gold_upos": test_y,
            "prediction": prediction,
            "lexical_prediction": lexical,
            "shuffled_prediction": shuffled,
            "random_feature_prediction": random_feature,
        }
    )
    _write_frames(run_dir, raw=raw, exclusions=pd.concat(exclusions, ignore_index=True))
    metrics = pd.DataFrame(
        [
            {
                "selected_c": selected_c,
                "accuracy": accuracy_score(test_y, prediction),
                "macro_f1": f1_score(test_y, prediction, average="macro"),
                "majority_accuracy": accuracy_score(test_y, np.repeat(majority, len(test_y))),
                "lexical_accuracy": accuracy_score(test_y, lexical),
                "shuffled_accuracy": accuracy_score(test_y, shuffled),
                "random_feature_accuracy": accuracy_score(test_y, random_feature),
                "n_test_positions": len(test_y),
                "n_test_sentences": len(set(test_groups)),
            }
        ]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    metrics.assign(seed=42).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    return {"selected_c": selected_c, "n_test_positions": len(test_y)}


def _masked_probe_features(model, tokenizer, examples: list[Example], cfg: RunConfig):
    features, labels, groups, forms = [], [], [], []
    timestep = round(cfg.experiment.normalized_progress[0] * (cfg.experiment.steps - 1))
    for example in examples:
        _, hidden_states, state = states_at_time(
            model,
            tokenizer,
            example.text,
            timestep,
            cfg.experiment.steps,
            cfg.experiment.seeds[0],
            True,
        )
        hidden = hidden_states[len(hidden_states) // 2][0].float().cpu().numpy()
        for word_index, span in example.word_to_tokens.items():
            if not span or max(span) >= len(state.is_visible):
                continue
            if any(state.is_visible[position] for position in span):
                continue
            features.append(hidden[span].mean(axis=0))
            labels.append(example.upos[word_index])
            groups.append(example.sentence_id)
            forms.append(example.tokens[word_index])
    return np.asarray(features), np.asarray(labels), np.asarray(groups), np.asarray(forms)


def _lexical_predictions(train_forms, train_labels, test_forms, fallback):
    counts: dict[str, Counter] = defaultdict(Counter)
    for form, label in zip(train_forms, train_labels, strict=True):
        counts[str(form)][str(label)] += 1
    table = {form: values.most_common(1)[0][0] for form, values in counts.items()}
    return np.asarray([table.get(str(form), fallback) for form in test_forms])


def _write_frames(run_dir: Path, *, raw: pd.DataFrame, exclusions: pd.DataFrame) -> None:
    raw.to_parquet(run_dir / "instances.parquet", index=False)
    if exclusions.empty:
        exclusions = pd.DataFrame(columns=["sentence_id", "instance_id", "role", "reason"])
    exclusions.to_parquet(run_dir / "exclusions.parquet", index=False)
    if not raw.empty:
        size = 10_000
        for shard_id, start in enumerate(range(0, len(raw), size)):
            write_shard(run_dir, shard_id, raw.iloc[start : start + size].to_dict("records"))


def selection_aware_permutation(
    select_rows: pd.DataFrame,
    dev_rows: pd.DataFrame,
    *,
    relation: str,
    top_k: int,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Repeat the select top-K and dev-choice procedure under permuted labels."""
    select = select_rows[select_rows["relation"] == relation].copy()
    dev = dev_rows[dev_rows["relation"] == relation].copy()
    if select.empty or dev.empty:
        return {
            "p_value": float("nan"),
            "n_permutations": 0,
            "reason": "no_rows",
        }
    head_keys = ["layer", "head"]
    observed_top = (
        aggregate_head_scores(select)
        .sort_values(
            ["accuracy", "n_total", "layer", "head"],
            ascending=[False, False, True, True],
        )
        .head(top_k)
    )
    observed_dev = aggregate_head_scores(dev).merge(observed_top[head_keys], on=head_keys, how="inner")
    observed = float(observed_dev["accuracy"].max())
    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(n_permutations):
        permuted = []
        for original in (select, dev):
            frame = original.copy()
            instances = frame[["sentence_id", "instance_id", "gold_receiver_word_idx"]].drop_duplicates()
            shuffled = instances["gold_receiver_word_idx"].to_numpy().copy()
            rng.shuffle(shuffled)
            instances["permuted_receiver"] = shuffled
            frame = frame.drop(columns=["correct"]).merge(
                instances[["sentence_id", "instance_id", "permuted_receiver"]],
                on=["sentence_id", "instance_id"],
                how="left",
            )
            frame["correct"] = (frame["predicted_word_idx"] == frame["permuted_receiver"]).astype(int)
            permuted.append(frame)
        select_scores = aggregate_head_scores(permuted[0])
        top = select_scores.sort_values(
            ["accuracy", "n_total", "layer", "head"],
            ascending=[False, False, True, True],
        ).head(top_k)
        dev_scores = aggregate_head_scores(permuted[1]).merge(top[head_keys], on=head_keys, how="inner")
        null_scores.append(float(dev_scores["accuracy"].max()))
    p_value = (1 + sum(value >= observed for value in null_scores)) / (n_permutations + 1)
    return {
        "observed_dev_accuracy": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
    }


def _structural_slices(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    frame["distance_bin"] = pd.cut(
        frame["signed_distance"].abs(),
        bins=[0, 1, 2, 3, 5, 8, float("inf")],
        labels=["1", "2", "3", "4-5", "6-8", "9+"],
    ).astype(str)
    dimensions = [
        "distance_bin",
        "direction",
        "coordinated",
        "embedded_clause",
        "relative_clause",
        "passive_voice",
        "punctuation_between",
    ]
    outputs = []
    base = ["treebank", "relation", "layer", "head", "visibility"]
    for dimension in dimensions:
        grouped = (
            frame.groupby([*base, dimension], as_index=False, observed=True)
            .agg(
                accuracy=("correct", "mean"),
                n_instances=("correct", "size"),
            )
            .rename(columns={dimension: "slice_value"})
        )
        grouped["slice_dimension"] = dimension
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def read_source_lock(path: str | Path, cfg: RunConfig) -> SelectionLock:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    lock = SelectionLock(**raw)
    if lock.dataset_id != "ewt":
        raise ArtifactError("selection lock must originate from EWT")
    if lock.model_id != cfg.model.id or lock.model_revision != cfg.model.revision:
        raise ArtifactError("selection lock model/revision mismatch")
    return lock

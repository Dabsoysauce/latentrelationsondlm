"""Shared scoring and artifact helpers used by the active experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import ArtifactError, write_shard
from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..controls import matched_word, receiver_controls
from ..diffusion import attentions_at_time, endpoint_visibility, receiver_span_scores
from ..evaluation.statistics import sentence_clustered_bootstrap
from ..relations import Example


def selection_progress(cfg: RunConfig) -> float:
    states = {"both_masked": 0.0, "both_visible": 1.0}
    try:
        return states[cfg.experiment.scoring.primary_visibility]
    except KeyError as exc:
        raise ArtifactError("head selection requires both_masked or both_visible") from exc


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
    sentence_offset: int = 0,
    total_sentences: int | None = None,
) -> pd.DataFrame:
    """Score all heads or one frozen subset and retain per-instance evidence."""
    progress = selection_progress(cfg) if normalized_progress is None else normalized_progress
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
        available = heads or {
            (layer, head)
            for layer, attention in enumerate(attentions)
            for head in range(attention.shape[1])
        }
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
            for layer in sorted({item[0] for item in available}):
                scores = receiver_span_scores(
                    attentions,
                    layer,
                    instance.attender_span,
                    candidate_spans,
                    row_aggregation=cfg.experiment.scoring.attender_rows,
                    span_aggregation=cfg.experiment.scoring.receiver_span,
                    excluded_positions=set(instance.attender_span),
                )
                for head in sorted(head for candidate_layer, head in available if candidate_layer == layer):
                    prediction_word = candidate_words[int(np.argmax(scores[head]))]
                    alternative, relaxation = matched_word(example, instance, candidate_words)
                    alternative_mass = (
                        float(scores[head, candidate_words.index(alternative)])
                        if alternative is not None
                        else float("nan")
                    )
                    rows.append(
                        {
                            **instance_metadata(example, instance, role),
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
                            "matched_word_idx": alternative,
                            "matched_attention_mass": alternative_mass,
                            "matched_relaxation_level": relaxation,
                            "matched_gold_greater": (
                                float(scores[head, gold_index]) > alternative_mass
                                if alternative is not None
                                else None
                            ),
                            "n_candidate_words": len(candidate_words),
                            **receiver_controls(example, instance, candidate_words, seed=seed),
                        }
                    )
        completed = sentence_offset + sentence_index + 1
        if completed % 100 == 0:
            print(
                f"[score] {role} seed={seed}: {completed}/{total_sentences or len(examples)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def score_over_seeds(
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
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    progress = selection_progress(cfg) if normalized_progress is None else normalized_progress
    suffix = (
        "all"
        if heads is None
        else "-".join(f"l{layer}h{head}" for layer, head in sorted(heads))
    )
    frames = []
    store = SentenceCheckpointStore(checkpoint_dir.parent) if checkpoint_dir is not None else None
    for seed in cfg.experiment.seeds if seeds is None else seeds:
        legacy = (
            checkpoint_dir / f"{stage}-seed{seed}-p{progress:.6f}-{suffix}.parquet"
            if checkpoint_dir is not None
            else None
        )
        if store is not None:
            timestep = round(progress * (cfg.experiment.steps - 1))
            selected_heads = tuple(sorted(heads)) if heads is not None else None
            identity = CheckpointIdentity(
                stage=stage,
                seed=seed,
                normalized_progress=progress,
                timestep=timestep,
                heads=selected_heads,
            )
            frame = store.run(
                examples,
                identity,
                lambda chunk, start, current_seed=seed: score_attention_heads(
                    model,
                    tokenizer,
                    list(chunk),
                    cfg,
                    role=role,
                    heads=heads,
                    normalized_progress=progress,
                    seed=current_seed,
                    sentence_offset=start,
                    total_sentences=len(examples),
                ),
                legacy_path=legacy,
            )
        else:
            frame = score_attention_heads(
                model,
                tokenizer,
                examples,
                cfg,
                role=role,
                heads=heads,
                normalized_progress=progress,
                seed=seed,
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def instance_metadata(example: Example, instance, role: str) -> dict[str, Any]:
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


def aggregate_head_scores(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["relation", "layer", "head", "accuracy", "n_total", "n_correct"])
    return rows.groupby(["relation", "layer", "head"], as_index=False).agg(
        accuracy=("correct", "mean"), n_total=("correct", "size"), n_correct=("correct", "sum")
    )


def locked_metrics(rows: pd.DataFrame, fixed_offset: int | None) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    frame["fixed_offset_correct"] = (
        frame["attender_word_idx"] + fixed_offset == frame["receiver_word_idx"]
        if fixed_offset is not None
        else False
    )
    keys = ["treebank", "relation", "layer", "head", "visibility"]
    metrics = frame.groupby(keys, as_index=False).agg(
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
    intervals = []
    for identity, group in frame.groupby(keys, observed=True):
        low, high = sentence_clustered_bootstrap(group, value_col="correct", n_boot=2000, seed=42)
        intervals.append({**dict(zip(keys, identity, strict=True)), "ci_low": low, "ci_high": high})
    return metrics.merge(pd.DataFrame(intervals), on=keys, how="left")


def per_seed_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby(
        ["seed", "treebank", "relation", "layer", "head", "visibility"], as_index=False
    ).agg(accuracy=("correct", "mean"), n_instances=("correct", "size"))


def selection_aware_permutation(
    select_rows: pd.DataFrame,
    dev_rows: pd.DataFrame,
    *,
    relation: str,
    top_k: int,
    n_permutations: int,
    seed: int,
    minimum_denominator: int = 1,
) -> dict[str, Any]:
    """Repeat select-top-K and dev-choice under permuted receiver labels."""
    select = select_rows[select_rows["relation"] == relation].copy()
    dev = dev_rows[dev_rows["relation"] == relation].copy()
    if select.empty or dev.empty:
        return {"p_value": float("nan"), "n_permutations": 0, "reason": "no_rows"}
    head_keys = ["layer", "head"]
    observed_top = aggregate_head_scores(select)
    observed_top = observed_top[observed_top["n_total"] >= minimum_denominator].sort_values(
        ["accuracy", "n_total", "layer", "head"], ascending=[False, False, True, True]
    ).head(top_k)
    if observed_top.empty:
        return {
            "p_value": float("nan"),
            "n_permutations": 0,
            "reason": "no_select_head_meets_minimum_denominator",
        }
    observed_dev = aggregate_head_scores(dev).merge(observed_top[head_keys], on=head_keys, how="inner")
    observed_dev = observed_dev[observed_dev["n_total"] >= minimum_denominator]
    if observed_dev.empty:
        return {
            "p_value": float("nan"),
            "n_permutations": 0,
            "reason": "no_dev_candidate_meets_minimum_denominator",
        }
    observed = float(observed_dev["accuracy"].max())
    rng = np.random.default_rng(seed)
    select_arrays = _permutation_arrays(select)
    dev_arrays = _permutation_arrays(dev)
    dev_head_lookup = {head: index for index, head in enumerate(dev_arrays["heads"])}
    null_scores = []
    for _ in range(n_permutations):
        select_accuracy = _permuted_accuracies(select_arrays, rng)
        dev_accuracy = _permuted_accuracies(dev_arrays, rng)
        eligible_select = [
            index
            for index, denominator in enumerate(select_arrays["denominators"])
            if denominator >= minimum_denominator
        ]
        ranked_select = sorted(
            eligible_select,
            key=lambda index: (
                -select_accuracy[index],
                -select_arrays["denominators"][index],
                *select_arrays["heads"][index],
            ),
        )[:top_k]
        dev_candidates = [
            dev_head_lookup[select_arrays["heads"][index]]
            for index in ranked_select
            if select_arrays["heads"][index] in dev_head_lookup
            and dev_arrays["denominators"][dev_head_lookup[select_arrays["heads"][index]]]
            >= minimum_denominator
        ]
        if not dev_candidates:
            raise ArtifactError("permutation lost every denominator-eligible dev candidate")
        winner = min(
            dev_candidates,
            key=lambda index: (
                -dev_accuracy[index],
                -dev_arrays["denominators"][index],
                *dev_arrays["heads"][index],
            ),
        )
        null_scores.append(float(dev_accuracy[winner]))
    return {
        "observed_dev_accuracy": observed,
        "p_value": (1 + sum(value >= observed for value in null_scores)) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
    }


def _permutation_arrays(frame: pd.DataFrame) -> dict[str, Any]:
    """Encode one role once so each null draw uses only NumPy operations."""
    instance_keys = ["sentence_id", "instance_id"]
    instances = frame[[*instance_keys, "gold_receiver_word_idx"]].drop_duplicates()
    if instances.duplicated(instance_keys).any():
        raise ArtifactError("one relation instance has inconsistent gold receivers")
    instance_index = pd.MultiIndex.from_frame(instances[instance_keys])
    row_instances = instance_index.get_indexer(pd.MultiIndex.from_frame(frame[instance_keys]))
    if (row_instances < 0).any():
        raise ArtifactError("could not index relation instances for permutation")
    heads = sorted(
        (int(layer), int(head))
        for layer, head in frame[["layer", "head"]].drop_duplicates().itertuples(index=False)
    )
    head_index = pd.MultiIndex.from_tuples(heads, names=["layer", "head"])
    row_heads = head_index.get_indexer(pd.MultiIndex.from_frame(frame[["layer", "head"]]))
    return {
        "heads": heads,
        "row_instances": row_instances,
        "row_heads": row_heads,
        "predictions": frame["predicted_word_idx"].to_numpy(),
        "gold": instances["gold_receiver_word_idx"].to_numpy(),
        "denominators": np.bincount(row_heads, minlength=len(heads)),
    }


def _permuted_accuracies(arrays: dict[str, Any], rng) -> np.ndarray:
    labels = arrays["gold"].copy()
    rng.shuffle(labels)
    correct = arrays["predictions"] == labels[arrays["row_instances"]]
    counts = np.bincount(
        arrays["row_heads"], weights=correct.astype(np.int64), minlength=len(arrays["heads"])
    )
    return np.divide(
        counts,
        arrays["denominators"],
        out=np.zeros_like(counts, dtype=float),
        where=arrays["denominators"] > 0,
    )


def structural_slices(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    frame["distance_bin"] = pd.cut(
        frame["signed_distance"].abs(),
        bins=[0, 1, 2, 3, 5, 8, float("inf")],
        labels=["1", "2", "3", "4-5", "6-8", "9+"],
    ).astype(str)
    base = ["treebank", "relation", "layer", "head", "visibility"]
    outputs = []
    for dimension in (
        "distance_bin",
        "direction",
        "coordinated",
        "embedded_clause",
        "relative_clause",
        "passive_voice",
        "punctuation_between",
    ):
        grouped = (
            frame.groupby([*base, dimension], as_index=False, observed=True)
            .agg(accuracy=("correct", "mean"), n_instances=("correct", "size"))
            .rename(columns={dimension: "slice_value"})
        )
        grouped["slice_dimension"] = dimension
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def write_frames(run_dir: Path, *, raw: pd.DataFrame, exclusions: pd.DataFrame) -> None:
    raw.to_parquet(run_dir / "instances.parquet", index=False)
    if exclusions.empty:
        exclusions = pd.DataFrame(columns=["sentence_id", "instance_id", "role", "reason"])
    exclusions.to_parquet(run_dir / "exclusions.parquet", index=False)
    for shard_id, start in enumerate(range(0, len(raw), 10_000)):
        write_shard(run_dir, shard_id, raw.iloc[start : start + 10_000].to_dict("records"))

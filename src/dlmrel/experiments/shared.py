"""Shared scoring and artifact helpers used by the active experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import ArtifactError, write_shard
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
    seed: int,
) -> pd.DataFrame:
    """Score all heads or one frozen subset and retain per-instance evidence."""
    progress = selection_progress(cfg) if normalized_progress is None else normalized_progress
    timestep = round(progress * (cfg.experiment.steps - 1))
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
        if (sentence_index + 1) % 100 == 0:
            print(f"[score] {role}: {sentence_index + 1}/{len(examples)}", flush=True)
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
) -> pd.DataFrame:
    progress = selection_progress(cfg) if normalized_progress is None else normalized_progress
    suffix = (
        "all"
        if heads is None
        else "-".join(f"l{layer}h{head}" for layer, head in sorted(heads))
    )
    frames = []
    for seed in cfg.experiment.seeds:
        checkpoint = (
            checkpoint_dir / f"{stage}-seed{seed}-p{progress:.6f}-{suffix}.parquet"
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
                normalized_progress=progress,
                seed=seed,
            )
            if checkpoint is not None:
                atomic_parquet(checkpoint, frame)
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
) -> dict[str, Any]:
    """Repeat select-top-K and dev-choice under permuted receiver labels."""
    select = select_rows[select_rows["relation"] == relation].copy()
    dev = dev_rows[dev_rows["relation"] == relation].copy()
    if select.empty or dev.empty:
        return {"p_value": float("nan"), "n_permutations": 0, "reason": "no_rows"}
    head_keys = ["layer", "head"]
    observed_top = aggregate_head_scores(select).sort_values(
        ["accuracy", "n_total", "layer", "head"], ascending=[False, False, True, True]
    ).head(top_k)
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
        top = aggregate_head_scores(permuted[0]).sort_values(
            ["accuracy", "n_total", "layer", "head"], ascending=[False, False, True, True]
        ).head(top_k)
        scores = aggregate_head_scores(permuted[1]).merge(top[head_keys], on=head_keys, how="inner")
        null_scores.append(float(scores["accuracy"].max()))
    return {
        "observed_dev_accuracy": observed,
        "p_value": (1 + sum(value >= observed for value in null_scores)) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
    }


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


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_frames(run_dir: Path, *, raw: pd.DataFrame, exclusions: pd.DataFrame) -> None:
    raw.to_parquet(run_dir / "instances.parquet", index=False)
    if exclusions.empty:
        exclusions = pd.DataFrame(columns=["sentence_id", "instance_id", "role", "reason"])
    exclusions.to_parquet(run_dir / "exclusions.parquet", index=False)
    for shard_id, start in enumerate(range(0, len(raw), 10_000)):
        write_shard(run_dir, shard_id, raw.iloc[start : start + 10_000].to_dict("records"))

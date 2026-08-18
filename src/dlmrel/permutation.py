"""Deterministic, checkpointed selection-aware permutation inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ArtifactError, atomic_json, canonical_hash

PERMUTATION_SCHEMA = "dlmrel-selection-permutation-v2"
NULL_DEFINITION = (
    "independently within each split and permutation, sample one receiver word uniformly from "
    "each relation instance's aligned within-sentence candidate words; exclude the attender word "
    "and all non-word BOS, padding, and special-token positions; share the sampled receiver across "
    "all heads and seeds for that instance; select top-K on select, choose on dev, evaluate on test"
)
SPLIT_CODES = {"select": 11, "dev": 23, "test": 37}
INSTANCE_KEYS = ["sentence_id", "instance_id"]


@dataclass(frozen=True)
class EncodedSplit:
    role: str
    heads: tuple[tuple[int, int], ...]
    instance_keys: tuple[tuple[str, str], ...]
    row_instances: np.ndarray
    row_heads: np.ndarray
    predictions: np.ndarray
    gold: np.ndarray
    candidate_values: np.ndarray
    candidate_offsets: np.ndarray
    candidate_lengths: np.ndarray
    denominators: np.ndarray


def selection_aware_permutation(
    select_rows: pd.DataFrame,
    dev_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    relation: str,
    top_k: int,
    n_permutations: int,
    seed: int,
    scientific_config_hash: str,
    minimum_denominator: int = 1,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    progress_interval: int = 50,
    checkpoint_interval: int = 50,
    max_new_permutations: int | None = None,
) -> dict[str, Any]:
    """Run the full select/dev/test protocol under a valid within-instance null."""
    if n_permutations < 1 or top_k < 1 or minimum_denominator < 1:
        raise ValueError("permutations, top_k, and minimum denominator must be positive")
    encoded = {
        "select": _encode_split(select_rows, relation=relation, role="select"),
        "dev": _encode_split(dev_rows, relation=relation, role="dev"),
        "test": _encode_split(test_rows, relation=relation, role="test"),
    }
    _validate_head_coverage(encoded, minimum_denominator)
    observed_head, observed_test = _observed_protocol(
        encoded, top_k=top_k, minimum_denominator=minimum_denominator
    )
    identity = {
        "schema_version": PERMUTATION_SCHEMA,
        "relation": relation,
        "top_k": top_k,
        "minimum_denominator": minimum_denominator,
        "n_permutations": n_permutations,
        "base_seed": seed,
        "scientific_config_hash": scientific_config_hash,
        "evidence_hash": _evidence_hash(encoded),
        "null_definition": NULL_DEFINITION,
        "observed_selected_layer": observed_head[0],
        "observed_selected_head": observed_head[1],
        "observed_test_accuracy": observed_test,
    }
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    completed, null_statistics, selected_heads = _initial_progress(
        checkpoint, identity, resume=resume
    )
    start = len(completed)
    stop = n_permutations
    if max_new_permutations is not None:
        if max_new_permutations < 0:
            raise ValueError("max_new_permutations cannot be negative")
        stop = min(stop, start + max_new_permutations)

    for permutation_index in range(start, stop):
        accuracies = {
            role: _permuted_accuracies(
                frame,
                _sample_labels(frame, seed, permutation_index, role),
            )
            for role, frame in encoded.items()
        }
        selected = _select_head(
            encoded,
            accuracies["select"],
            accuracies["dev"],
            top_k=top_k,
            minimum_denominator=minimum_denominator,
        )
        test_index = encoded["test"].heads.index(selected)
        null_statistics.append(float(accuracies["test"][test_index]))
        selected_heads.append([selected[0], selected[1]])
        completed.append(permutation_index)
        count = permutation_index + 1
        if progress_interval > 0 and (count % progress_interval == 0 or count == n_permutations):
            print(f"[permutation] {relation}: {count}/{n_permutations}", flush=True)
        if checkpoint is not None and checkpoint_interval > 0 and count % checkpoint_interval == 0:
            _write_checkpoint(checkpoint, identity, completed, null_statistics, selected_heads)

    if checkpoint is not None:
        _write_checkpoint(checkpoint, identity, completed, null_statistics, selected_heads)
    complete = len(completed) == n_permutations
    return _result(identity, completed, null_statistics, selected_heads, complete=complete)


def randomized_receiver_labels(
    rows: pd.DataFrame,
    *,
    relation: str,
    role: str,
    seed: int,
    permutation_index: int,
) -> dict[tuple[str, str], int]:
    """Expose one deterministic draw for protocol tests and audits."""
    encoded = _encode_split(rows, relation=relation, role=role)
    labels = _sample_labels(encoded, seed, permutation_index, role)
    return dict(zip(encoded.instance_keys, labels.astype(int), strict=True))


def _encode_split(rows: pd.DataFrame, *, relation: str, role: str) -> EncodedSplit:
    frame = rows[rows["relation"] == relation].copy()
    required = {
        *INSTANCE_KEYS,
        "relation",
        "layer",
        "head",
        "seed",
        "predicted_word_idx",
        "gold_receiver_word_idx",
        "attender_word_idx",
        "sentence_length_words",
        "n_candidate_words",
    }
    if missing := required - set(frame):
        raise ArtifactError(f"{role} permutation evidence missing columns: {sorted(missing)}")
    if frame.empty:
        raise ArtifactError(f"no {role} permutation rows for relation {relation!r}")
    if "role" in frame and set(frame["role"].astype(str)) != {role}:
        raise ArtifactError(f"{role} permutation evidence contains another split role")
    row_identity = [*INSTANCE_KEYS, "seed", "layer", "head"]
    if frame.duplicated(row_identity).any():
        raise ArtifactError(f"{role} permutation evidence contains duplicate head rows")
    frame = frame.sort_values(row_identity, kind="mergesort").reset_index(drop=True)

    metadata_columns = [
        *INSTANCE_KEYS,
        "gold_receiver_word_idx",
        "attender_word_idx",
        "sentence_length_words",
    ]
    instances = frame[metadata_columns].drop_duplicates()
    if instances.duplicated(INSTANCE_KEYS).any():
        raise ArtifactError(f"{role} relation-instance candidate metadata is inconsistent")
    instance_index = pd.MultiIndex.from_frame(instances[INSTANCE_KEYS])
    row_instances = instance_index.get_indexer(pd.MultiIndex.from_frame(frame[INSTANCE_KEYS]))
    if (row_instances < 0).any():
        raise ArtifactError(f"could not index {role} relation instances")

    heads = tuple(
        sorted(
            (int(layer), int(head))
            for layer, head in frame[["layer", "head"]].drop_duplicates().itertuples(index=False)
        )
    )
    head_index = pd.MultiIndex.from_tuples(heads, names=["layer", "head"])
    row_heads = head_index.get_indexer(pd.MultiIndex.from_frame(frame[["layer", "head"]]))
    if (row_heads < 0).any():
        raise ArtifactError(f"could not index {role} heads")

    candidate_lists = []
    for item in instances.itertuples(index=False):
        sentence_length = int(item.sentence_length_words)
        attender = int(item.attender_word_idx)
        if sentence_length < 2 or not 0 <= attender < sentence_length:
            raise ArtifactError(f"{role} instance {item.instance_id!r} has invalid word bounds")
        candidates = np.array(
            [word for word in range(sentence_length) if word != attender], dtype=np.int32
        )
        if len(candidates) == 0:
            raise ArtifactError(f"{role} instance {item.instance_id!r} has no valid receivers")
        if int(item.gold_receiver_word_idx) not in set(candidates.tolist()):
            raise ArtifactError(f"{role} gold receiver is outside its within-sentence candidates")
        candidate_lists.append(candidates)
    lengths = np.array([len(values) for values in candidate_lists], dtype=np.int32)
    offsets = np.zeros(len(candidate_lists), dtype=np.int64)
    if len(offsets) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    candidate_values = np.concatenate(candidate_lists).astype(np.int32, copy=False)
    predictions = frame["predicted_word_idx"].to_numpy(dtype=np.int32)
    gold = instances["gold_receiver_word_idx"].to_numpy(dtype=np.int32)
    if "correct" in frame:
        expected = predictions == gold[row_instances]
        if not np.array_equal(expected.astype(np.int8), frame["correct"].to_numpy(dtype=np.int8)):
            raise ArtifactError(f"{role} correct values disagree with prediction and gold")
    expected_counts = lengths[row_instances]
    if not np.array_equal(
        expected_counts.astype(np.int64), frame["n_candidate_words"].to_numpy(dtype=np.int64)
    ):
        raise ArtifactError(f"{role} saved candidate denominators are inconsistent")

    return EncodedSplit(
        role=role,
        heads=heads,
        instance_keys=tuple(
            (str(sentence_id), str(instance_id))
            for sentence_id, instance_id in instances[INSTANCE_KEYS].itertuples(index=False)
        ),
        row_instances=row_instances.astype(np.int32, copy=False),
        row_heads=row_heads.astype(np.int32, copy=False),
        predictions=predictions,
        gold=gold,
        candidate_values=candidate_values,
        candidate_offsets=offsets,
        candidate_lengths=lengths,
        denominators=np.bincount(row_heads, minlength=len(heads)).astype(np.int64),
    )


def _validate_head_coverage(encoded: dict[str, EncodedSplit], minimum_denominator: int) -> None:
    select = encoded["select"]
    for head_index, head in enumerate(select.heads):
        if select.denominators[head_index] < minimum_denominator:
            continue
        for role in ("dev", "test"):
            if head not in encoded[role].heads:
                raise ArtifactError(f"{role} evidence is missing selectable head {head}")


def _sample_labels(
    frame: EncodedSplit, base_seed: int, permutation_index: int, role: str
) -> np.ndarray:
    if role not in SPLIT_CODES:
        raise ValueError(f"unknown permutation split role: {role!r}")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(permutation_index), SPLIT_CODES[role]])
    )
    positions = np.floor(rng.random(len(frame.candidate_lengths)) * frame.candidate_lengths).astype(
        np.int64
    )
    return frame.candidate_values[frame.candidate_offsets + positions]


def _permuted_accuracies(frame: EncodedSplit, labels: np.ndarray) -> np.ndarray:
    correct = frame.predictions == labels[frame.row_instances]
    counts = np.bincount(
        frame.row_heads,
        weights=correct.astype(np.int64),
        minlength=len(frame.heads),
    )
    return np.divide(
        counts,
        frame.denominators,
        out=np.zeros_like(counts, dtype=float),
        where=frame.denominators > 0,
    )


def _rank_heads(
    frame: EncodedSplit,
    accuracies: np.ndarray,
    *,
    minimum_denominator: int,
    admitted: set[tuple[int, int]] | None = None,
) -> list[int]:
    indices = [
        index
        for index, head in enumerate(frame.heads)
        if frame.denominators[index] >= minimum_denominator
        and (admitted is None or head in admitted)
    ]
    return sorted(
        indices,
        key=lambda index: (
            -accuracies[index],
            -frame.denominators[index],
            *frame.heads[index],
        ),
    )


def _select_head(
    encoded: dict[str, EncodedSplit],
    select_accuracies: np.ndarray,
    dev_accuracies: np.ndarray,
    *,
    top_k: int,
    minimum_denominator: int,
) -> tuple[int, int]:
    select = encoded["select"]
    dev = encoded["dev"]
    top = _rank_heads(
        select, select_accuracies, minimum_denominator=minimum_denominator
    )[:top_k]
    if not top:
        raise ArtifactError("no select head meets the minimum denominator")
    admitted = {select.heads[index] for index in top}
    dev_ranked = _rank_heads(
        dev,
        dev_accuracies,
        minimum_denominator=minimum_denominator,
        admitted=admitted,
    )
    if not dev_ranked:
        raise ArtifactError("no select top-K head has sufficient dev evidence")
    return dev.heads[dev_ranked[0]]


def _observed_protocol(
    encoded: dict[str, EncodedSplit], *, top_k: int, minimum_denominator: int
) -> tuple[tuple[int, int], float]:
    accuracies = {
        role: _permuted_accuracies(frame, frame.gold)
        for role, frame in encoded.items()
    }
    selected = _select_head(
        encoded,
        accuracies["select"],
        accuracies["dev"],
        top_k=top_k,
        minimum_denominator=minimum_denominator,
    )
    return selected, float(accuracies["test"][encoded["test"].heads.index(selected)])


def _evidence_hash(encoded: dict[str, EncodedSplit]) -> str:
    record = {}
    for role, frame in encoded.items():
        arrays = {
            "row_instances": frame.row_instances,
            "row_heads": frame.row_heads,
            "predictions": frame.predictions,
            "gold": frame.gold,
            "candidate_values": frame.candidate_values,
            "candidate_offsets": frame.candidate_offsets,
            "candidate_lengths": frame.candidate_lengths,
            "denominators": frame.denominators,
        }
        record[role] = {
            "heads": frame.heads,
            "instance_keys_hash": canonical_hash(frame.instance_keys),
            "arrays": {name: _array_hash(value) for name, value in arrays.items()},
        }
    return canonical_hash(record)


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_hash(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _initial_progress(
    checkpoint: Path | None, identity: dict[str, Any], *, resume: bool
) -> tuple[list[int], list[float], list[list[int]]]:
    if checkpoint is None or not checkpoint.exists():
        return [], [], []
    if not resume:
        raise ArtifactError(f"permutation checkpoint already exists; use resume: {checkpoint}")
    try:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"permutation checkpoint is unreadable: {checkpoint}") from exc
    for key, value in identity.items():
        if saved.get(key) != value:
            raise ArtifactError(f"permutation checkpoint scientific identity differs at {key}")
    completed = [int(index) for index in saved.get("completed_permutation_indices", [])]
    statistics = [float(value) for value in saved.get("null_statistics", [])]
    heads = [[int(value) for value in head] for head in saved.get("selected_heads", [])]
    if completed != list(range(len(completed))) or not (
        len(completed) == len(statistics) == len(heads)
    ):
        raise ArtifactError("permutation checkpoint progress is not contiguous and consistent")
    return completed, statistics, heads


def _write_checkpoint(
    checkpoint: Path,
    identity: dict[str, Any],
    completed: list[int],
    statistics: list[float],
    selected_heads: list[list[int]],
) -> None:
    atomic_json(
        checkpoint,
        {
            **identity,
            "completion_status": (
                "complete" if len(completed) == identity["n_permutations"] else "incomplete"
            ),
            "completed_permutation_indices": completed,
            "null_statistics": statistics,
            "selected_heads": selected_heads,
        },
    )


def _result(
    identity: dict[str, Any],
    completed: list[int],
    statistics: list[float],
    selected_heads: list[list[int]],
    *,
    complete: bool,
) -> dict[str, Any]:
    p_value = None
    null_mean = None
    null_std = None
    if complete:
        observed = float(identity["observed_test_accuracy"])
        p_value = (1 + sum(value >= observed for value in statistics)) / (
            identity["n_permutations"] + 1
        )
        null_mean = float(np.mean(statistics))
        null_std = float(np.std(statistics))
    return {
        **identity,
        "completion_status": "complete" if complete else "incomplete",
        "completed_permutation_indices": completed,
        "null_statistics": statistics,
        "selected_heads": selected_heads,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
    }

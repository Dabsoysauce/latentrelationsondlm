"""Frozen receiver controls and deterministic matched alternatives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def valid_receivers(n_words: int, attender: int, *, exclude: Iterable[int] = ()) -> list[int]:
    blocked = {attender, *exclude}
    return [index for index in range(n_words) if index not in blocked]


def uniform_receiver(candidates: list[int], *, seed: int, instance_id: str) -> int | None:
    if not candidates:
        return None
    stable = sum((i + 1) * ord(char) for i, char in enumerate(instance_id))
    return candidates[int(np.random.default_rng(seed + stable).integers(len(candidates)))]


def nearest_receiver(candidates: list[int], attender: int) -> int | None:
    return min(candidates, key=lambda x: (abs(x - attender), x)) if candidates else None


def adjacent_receiver(candidates: list[int], attender: int, offset: int) -> int | None:
    wanted = attender + offset
    return wanted if wanted in candidates else None


def fit_fixed_offset(frame: pd.DataFrame) -> int:
    if frame.empty:
        raise ValueError("cannot fit an offset on no select instances")
    counts = frame["receiver_word_idx"].sub(frame["attender_word_idx"]).value_counts()
    counts = counts[counts.index != 0]
    if counts.empty:
        raise ValueError("no nonzero offsets")
    best_count = counts.max()
    return int(sorted(counts[counts == best_count].index, key=lambda x: (abs(x), x))[0])


MATCH_LEVELS = (
    (
        "receiver_upos",
        "direction",
        "distance_bin",
        "punctuation_context",
        "bpe_length",
        "sentence_length_bin",
        "treebank",
        "timestep",
        "visibility",
    ),
    (
        "receiver_upos",
        "direction",
        "distance_bin",
        "bpe_length",
        "sentence_length_bin",
        "treebank",
        "timestep",
        "visibility",
    ),
    ("receiver_upos", "direction", "distance_bin", "treebank", "timestep", "visibility"),
    ("receiver_upos", "direction", "treebank", "visibility"),
)


@dataclass(frozen=True)
class MatchResult:
    alternative_id: str | None
    level: int | None
    matched_fields: tuple[str, ...]
    reason: str | None


def matched_alternative(target: pd.Series, pool: pd.DataFrame) -> MatchResult:
    candidates = pool[
        (pool["sentence_id"] == target["sentence_id"])
        & (pool["instance_id"] != target["instance_id"])
        & (pool["receiver_word_idx"] != target["receiver_word_idx"])
    ]
    for level, fields in enumerate(MATCH_LEVELS):
        usable = [field for field in fields if field in pool and field in target.index]
        matched = candidates
        for field in usable:
            matched = matched[matched[field] == target[field]]
        if not matched.empty:
            best = matched.sort_values("instance_id", kind="mergesort").iloc[0]
            return MatchResult(str(best["instance_id"]), level, tuple(usable), None)
    return MatchResult(None, None, (), "no_candidate_after_frozen_relaxation")


def paired_mass_statistics(gold: np.ndarray, alternative: np.ndarray) -> dict[str, float | int]:
    if gold.shape != alternative.shape:
        raise ValueError("paired arrays must have identical shape")
    delta = gold - alternative
    return {
        "p_gold_greater": float(np.mean(delta > 0)) if len(delta) else float("nan"),
        "mean_paired_difference": float(np.mean(delta)) if len(delta) else float("nan"),
        "ties": int(np.sum(delta == 0)),
        "n": int(len(delta)),
    }

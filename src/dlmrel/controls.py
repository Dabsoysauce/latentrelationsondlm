"""Deterministic receiver baselines fixed without looking at test outcomes."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .relations import Example, RelationInstance


def matched_word(
    example: Example, instance: RelationInstance, candidates: list[int]
) -> tuple[int | None, int | None]:
    """Find a wrong same-POS receiver using a frozen three-level relaxation."""
    alternatives = [
        word
        for word in candidates
        if word != instance.receiver_word_idx and example.upos[word] == instance.receiver_upos
    ]
    rules = (
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
    for level, rule in enumerate(rules):
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


def receiver_controls(
    example: Example, instance: RelationInstance, candidates: list[int], *, seed: int
) -> dict:
    """Return simple positional and POS baselines for one relation instance."""
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


def fit_fixed_offset(offsets: list[int]) -> int | None:
    """Choose the most common nonzero select-set offset with a fixed tie-break."""
    counts = Counter(offset for offset in offsets if offset != 0)
    if not counts:
        return None
    maximum = max(counts.values())
    return min((offset for offset, count in counts.items() if count == maximum), key=lambda x: (abs(x), x))

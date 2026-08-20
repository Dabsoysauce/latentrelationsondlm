import inspect

import numpy as np
import pandas as pd

from dlmrel.controls import fit_fixed_offset, matched_word, receiver_controls
from dlmrel.evaluation.statistics import (
    adjust_pvalues,
    hierarchical_seed_summary,
    sentence_clustered_bootstrap,
)
from dlmrel.relations import Example, RelationInstance


def test_fixed_offset_is_fit_on_select_offsets_only():
    assert fit_fixed_offset([-3, -3, -2, -3, -2]) == -3
    assert fit_fixed_offset([]) is None


def test_fixed_offset_ties_prefer_absolute_distance_then_signed_value():
    assert fit_fixed_offset([-3, -3, 2, 2]) == 2
    assert fit_fixed_offset([-2, -2, 2, 2]) == -2


def test_clustered_bootstrap_keeps_sentence_rows_together():
    frame = pd.DataFrame(
        {
            "sentence_id": ["a", "a", "b", "b"],
            "correct": [1, 1, 0, 0],
        }
    )
    low, high = sentence_clustered_bootstrap(frame, value_col="correct", n_boot=1000, seed=0)
    assert low <= 0.5 <= high


def test_clustered_bootstrap_matches_independent_reference_and_frozen_defaults():
    frame = pd.DataFrame(
        {"sentence_id": ["a", "a", "b", "c", "c", "c"], "correct": [1, 0, 0, 1, 1, 1]}
    )
    actual = sentence_clustered_bootstrap(frame, value_col="correct")
    clusters = [np.asarray([1.0, 0.0]), np.asarray([0.0]), np.asarray([1.0, 1.0, 1.0])]
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(2000):
        chosen = rng.integers(0, 3, 3)
        estimates.append(np.concatenate([clusters[index] for index in chosen]).mean())
    expected = np.quantile(estimates, [0.025, 0.975])

    np.testing.assert_allclose(actual, expected)
    signature = inspect.signature(sentence_clustered_bootstrap)
    assert signature.parameters["n_boot"].default == 2000
    assert signature.parameters["seed"].default == 42
    assert signature.parameters["ci"].default == 0.95


def test_empty_clustered_bootstrap_is_nan():
    low, high = sentence_clustered_bootstrap(pd.DataFrame(), value_col="correct")
    assert np.isnan(low) and np.isnan(high)


def test_seed_summary_does_not_count_repeated_rows_as_new_seeds():
    frame = pd.DataFrame(
        {
            "sentence_id": ["a", "a", "b", "b"],
            "seed": [1, 2, 1, 2],
            "correct": [1, 1, 0, 0],
        }
    )
    summary = hierarchical_seed_summary(frame).iloc[0]
    assert summary.n_seeds == 2
    assert summary.n_sentences == 2
    assert summary["mean"] == 0.5


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = adjust_pvalues([0.01, 0.04, 0.2], method="holm")
    assert adjusted == [0.03, 0.08, 0.2]
    assert all(0 <= value <= 1 for value in adjusted)


def _control_example():
    relation = RelationInstance(
        relation="object_to_verb",
        attender_span=[3],
        receiver_span=[5],
        attender_text="query",
        receiver_text="gold",
        attender_word_idx=2,
        receiver_word_idx=4,
        dep="obj",
        instance_id="stable-instance",
        receiver_upos="NOUN",
    )
    example = Example(
        text="a b query c gold d",
        tokens=["a", "b", "query", "c", "gold", "d"],
        upos=["NOUN", "VERB", "VERB", "NOUN", "NOUN", "NOUN"],
        deprel=["dep"] * 6,
        head=[0] * 6,
        word_to_tokens={index: [index + 1] for index in range(6)},
        relations=[relation],
        seq_len=7,
    )
    return example, relation


def test_receiver_controls_are_deterministic_and_follow_position_rules():
    example, relation = _control_example()
    first = receiver_controls(example, relation, [0, 1, 3, 4, 5], seed=42)
    second = receiver_controls(example, relation, [0, 1, 3, 4, 5], seed=42)

    assert first == second
    assert first["nearest_receiver_word_idx"] == 1
    assert first["previous_receiver_word_idx"] == 1
    assert first["next_receiver_word_idx"] == 3
    assert first["oracle_pos_receiver_word_idx"] == 3
    assert first["wrong_same_pos_word_idx"] == 3


def test_matched_word_uses_all_three_frozen_relaxation_levels():
    example, relation = _control_example()
    assert matched_word(example, relation, [3, 4, 5]) == (3, 0)

    example.upos[3] = "VERB"
    relation.receiver_word_idx = 3
    assert matched_word(example, relation, [0, 3, 5]) == (5, 1)

    relation.receiver_word_idx = 5
    assert matched_word(example, relation, [0, 3, 5]) == (0, 2)

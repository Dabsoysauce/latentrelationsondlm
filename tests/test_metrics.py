import numpy as np
import pandas as pd

from dlmrel.controls import fit_fixed_offset
from dlmrel.evaluation.statistics import (
    adjust_pvalues,
    hierarchical_seed_summary,
    sentence_clustered_bootstrap,
)


def test_fixed_offset_is_fit_on_select_offsets_only():
    assert fit_fixed_offset([-3, -3, -2, -3, -2]) == -3
    assert fit_fixed_offset([]) is None


def test_clustered_bootstrap_keeps_sentence_rows_together():
    frame = pd.DataFrame(
        {
            "sentence_id": ["a", "a", "b", "b"],
            "correct": [1, 1, 0, 0],
        }
    )
    low, high = sentence_clustered_bootstrap(frame, value_col="correct", n_boot=1000, seed=0)
    assert low <= 0.5 <= high


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

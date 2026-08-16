from __future__ import annotations

import pandas as pd
import pytest

from dlmrel.experiments.pos_probe import aggregate_probe_metrics


def test_probe_metrics_summarize_exactly_three_seed_runs():
    per_seed = pd.DataFrame(
        {
            "seed": [42, 43, 44],
            "accuracy": [0.6, 0.7, 0.8],
            "macro_f1": [0.5, 0.6, 0.7],
            "majority_accuracy": [0.3, 0.3, 0.3],
            "lexical_accuracy": [0.4, 0.5, 0.6],
            "shuffled_accuracy": [0.2, 0.3, 0.4],
            "random_feature_accuracy": [0.1, 0.2, 0.3],
            "n_test_positions": [100, 110, 120],
            "n_test_sentences": [20, 21, 22],
        }
    )

    summary = aggregate_probe_metrics(per_seed).iloc[0]

    assert summary.n_seeds == 3
    assert summary.accuracy == pytest.approx(0.7)
    assert summary.accuracy_seed_std == pytest.approx(0.1)
    assert summary.n_test_positions == 330
    assert summary.n_test_sentences == 22

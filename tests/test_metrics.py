from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dlmrel.evaluation.metrics import (
    bootstrap_ci,
    build_null_table,
    fit_offset_null,
    offset_accuracy,
    offset_correctness,
)
from dlmrel.evaluation.statistics import build_head_vs_null_table


def make(rows):
    return pd.DataFrame([{"relation": r, "attender_span": a, "receiver_span": rec} for r, a, rec in rows])


def test_perfect_adjacent_relation_is_solved_by_offset_one():
    df = make([("adj_noun", [i], [i + 1]) for i in range(1, 20)])
    null = fit_offset_null(df, "adj_noun")
    assert null.k == 1
    assert null.fit_accuracy == 1.0


def test_negative_offset_is_found():
    df = make([("obj_verb", [i], [i - 3]) for i in range(5, 25)])
    assert fit_offset_null(df, "obj_verb").k == -3


def test_zero_offset_is_never_selected():
    df = make([("self", [i], [i]) for i in range(1, 15)])
    assert fit_offset_null(df, "self").k != 0


def test_anchor_is_last_subtoken_by_default():
    df = make([("r", [4, 5], [6])])
    assert offset_accuracy(df, 1, attender_token="last") == 1.0
    assert offset_accuracy(df, 1, attender_token="first") == 0.0


def test_multi_token_receiver_counts_as_hit_on_any_subtoken():
    df = make([("r", [2], [3, 4, 5])])
    assert offset_accuracy(df, 1) == 1.0
    assert offset_accuracy(df, 3) == 1.0
    assert offset_accuracy(df, 4) == 0.0


def test_spread_offsets_cap_the_null_below_one():
    rows = []
    for i in range(10, 40):
        for k in (-1, -2, -3, -4):
            rows.append(("obj_verb", [i], [i + k]))
    null = fit_offset_null(make(rows), "obj_verb")
    assert null.fit_accuracy == pytest.approx(0.25, abs=0.01)


def test_span_columns_survive_a_csv_round_trip(tmp_path):
    df = make([("r", [2], [3])])
    path = tmp_path / "r.csv"
    df.to_csv(path, index=False)
    assert offset_accuracy(pd.read_csv(path), 1) == 1.0


def test_null_table_fits_on_select_and_reports_on_test():
    select = make([("r", [i], [i + 1]) for i in range(1, 20)])
    test = make([("r", [i], [i + 1]) for i in range(1, 10)] + [("r", [i], [i + 7]) for i in range(1, 10)])
    table = build_null_table(select, test, ["r"], n_boot=500).iloc[0]
    assert table["k"] == 1
    assert table["null_fit_acc"] == 1.0
    assert table["null_test_acc"] == pytest.approx(0.5)
    assert table["null_ci_lo"] < 0.5 < table["null_ci_hi"]


def test_offset_correctness_is_per_instance():
    df = make([("r", [1], [2]), ("r", [5], [9])])
    np.testing.assert_array_equal(offset_correctness(df, 1), [1.0, 0.0])


def test_bootstrap_ci_brackets_the_mean():
    correct = np.array([1.0] * 70 + [0.0] * 30)
    lo, hi = bootstrap_ci(correct, n_boot=2000, seed=0)
    assert lo < 0.7 < hi
    assert hi - lo < 0.25


def test_bootstrap_ci_on_empty_input_is_nan():
    lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_missing_relation_raises():
    with pytest.raises(ValueError):
        fit_offset_null(make([("a", [1], [2])]), "nonexistent")


class TestVerdictUsesNullInterval:
    def _tables(self, n_correct, n_total, null_acc, null_hi):
        scores = pd.DataFrame(
            [
                {
                    "relation": "r",
                    "layer": 0,
                    "head": 0,
                    "accuracy_select": 0.9,
                    "accuracy_test": n_correct / n_total,
                    "n_correct_test": n_correct,
                    "n_total_test": n_total,
                },
                {
                    "relation": "r",
                    "layer": 1,
                    "head": 1,
                    "accuracy_select": 0.1,
                    "accuracy_test": 0.1,
                    "n_correct_test": 10,
                    "n_total_test": 100,
                },
            ]
        )
        nulls = pd.DataFrame([{"relation": "r", "k": 1, "null_test_acc": null_acc, "null_ci_hi": null_hi}])
        return scores, nulls

    def test_head_inside_the_null_interval_does_not_survive(self):
        scores, nulls = self._tables(221, 340, null_acc=0.60, null_hi=0.72)
        row = build_head_vs_null_table(scores, nulls).iloc[0]
        assert row.verdict == "not distinguishable"
        assert row.margin < 0

    def test_head_clearing_the_null_upper_bound_survives(self):
        scores, nulls = self._tables(860, 1075, null_acc=0.30, null_hi=0.34)
        row = build_head_vs_null_table(scores, nulls).iloc[0]
        assert row.verdict == "survives"
        assert row.margin > 0

    def test_missing_null_ci_falls_back_to_the_point_estimate(self):
        scores, _ = self._tables(860, 1075, 0.30, 0.34)
        nulls = pd.DataFrame([{"relation": "r", "k": 1, "null_test_acc": 0.30}])
        assert build_head_vs_null_table(scores, nulls).iloc[0].verdict == "survives"

"""Tests for the fixed-offset null model.

The null is the load-bearing claim of this repository, so its edge cases are
pinned: multi-token receiver spans, the choice of attender anchor, negative
offsets, and the fit/report separation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dlmrel.nulls import (
    bootstrap_ci,
    build_null_table,
    fit_offset_null,
    offset_accuracy,
    offset_correctness,
)


def make(rows):
    """rows: (relation, attender_span, receiver_span)"""
    return pd.DataFrame(
        [
            {"relation": r, "attender_span": a, "receiver_span": rec}
            for r, a, rec in rows
        ]
    )


def test_perfect_adjacent_relation_is_solved_by_offset_one():
    df = make([("adj_noun", [i], [i + 1]) for i in range(1, 20)])
    null = fit_offset_null(df, "adj_noun")
    assert null.k == 1
    assert null.fit_accuracy == 1.0


def test_negative_offset_is_found():
    df = make([("obj_verb", [i], [i - 3]) for i in range(5, 25)])
    assert fit_offset_null(df, "obj_verb").k == -3


def test_zero_offset_is_never_selected():
    # A receiver that is its own attender would be trivially "predicted" by
    # k=0, but the attender's own positions are excluded from the argmax, so
    # the null must not be allowed to use it either.
    df = make([("self", [i], [i]) for i in range(1, 15)])
    assert fit_offset_null(df, "self").k != 0


def test_anchor_is_last_subtoken_by_default():
    # Attender occupies tokens 4 and 5; receiver is token 6. Only anchoring on
    # the last sub-token makes this a +1 relation.
    df = make([("r", [4, 5], [6])])
    assert offset_accuracy(df, 1, attender_token="last") == 1.0
    assert offset_accuracy(df, 1, attender_token="first") == 0.0


def test_multi_token_receiver_counts_as_hit_on_any_subtoken():
    df = make([("r", [2], [3, 4, 5])])
    assert offset_accuracy(df, 1) == 1.0
    assert offset_accuracy(df, 3) == 1.0
    assert offset_accuracy(df, 4) == 0.0


def test_spread_offsets_cap_the_null_below_one():
    # A relation whose distance varies cannot be solved by any single offset.
    # This is exactly why object->verb survives the null and det->noun does not.
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
    # Spans come back as the strings "[2]" / "[3]" and must still parse.
    assert offset_accuracy(pd.read_csv(path), 1) == 1.0


def test_null_table_fits_on_select_and_reports_on_test():
    select = make([("r", [i], [i + 1]) for i in range(1, 20)])
    # Held-out data where the fitted offset is wrong half the time.
    test = make(
        [("r", [i], [i + 1]) for i in range(1, 10)]
        + [("r", [i], [i + 7]) for i in range(1, 10)]
    )
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
    """A head must clear the null's upper bound, not its point estimate.

    Comparing an interval to a point treats the null as if it were known
    exactly. On UD-EWT that let subject determiner->noun read "survives" on a
    margin of 0.0009 -- an artifact of where the rounding fell, not a result.
    """

    def _tables(self, n_correct, n_total, null_acc, null_hi):
        import pandas as pd

        scores = pd.DataFrame(
            [{"relation": "r", "layer": 0, "head": 0, "accuracy_select": 0.9,
              "accuracy_test": n_correct / n_total, "n_correct_test": n_correct,
              "n_total_test": n_total},
             {"relation": "r", "layer": 1, "head": 1, "accuracy_select": 0.1,
              "accuracy_test": 0.1, "n_correct_test": 10, "n_total_test": 100}]
        )
        nulls = pd.DataFrame([{"relation": "r", "k": 1, "null_test_acc": null_acc,
                               "null_ci_hi": null_hi}])
        return scores, nulls

    def test_head_inside_the_null_interval_does_not_survive(self):
        from dlmrel.stats import build_head_vs_null_table

        # Head at 0.65 with a wide null interval reaching 0.72: not separable.
        scores, nulls = self._tables(221, 340, null_acc=0.60, null_hi=0.72)
        row = build_head_vs_null_table(scores, nulls).iloc[0]
        assert row.verdict == "not distinguishable"
        assert row.margin < 0

    def test_head_clearing_the_null_upper_bound_survives(self):
        from dlmrel.stats import build_head_vs_null_table

        scores, nulls = self._tables(860, 1075, null_acc=0.30, null_hi=0.34)
        row = build_head_vs_null_table(scores, nulls).iloc[0]
        assert row.verdict == "survives"
        assert row.margin > 0

    def test_missing_null_ci_falls_back_to_the_point_estimate(self):
        # Old null tables predate the CI columns; they must still analyze.
        import pandas as pd

        from dlmrel.stats import build_head_vs_null_table

        scores, _ = self._tables(860, 1075, 0.30, 0.34)
        nulls = pd.DataFrame([{"relation": "r", "k": 1, "null_test_acc": 0.30}])
        assert build_head_vs_null_table(scores, nulls).iloc[0].verdict == "survives"

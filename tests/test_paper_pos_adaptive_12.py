from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dlmrel.artifacts import canonical_hash, dataframe_records
from dlmrel.experiments import paper_causal, paper_pos
from dlmrel.experiments.paper_pos_adaptive_12 import candidate_set_12


def _screen(head_count=32):
    heads = tuple(f"head_{index}" for index in range(head_count))
    return pd.DataFrame(
        {
            "feature_kind": list(heads),
            "accuracy": [0.2 + index * 0.01 for index in range(head_count)],
            "n_positions": [1000] * head_count,
        }
    )


def test_fixed_candidate_rule_is_symmetric_deterministic_and_exactly_12():
    selected, reasons = candidate_set_12(_screen())
    shuffled, shuffled_reasons = candidate_set_12(
        _screen().sample(frac=1.0, random_state=9)
    )
    assert selected == shuffled
    assert reasons == shuffled_reasons
    assert len(selected) == 12
    assert {"head_31", "head_30", "head_29", "head_28"}.issubset(selected)
    assert {"head_0", "head_1", "head_2", "head_3"}.issubset(selected)
    assert {"head_27", "head_26", "head_5", "head_4"}.issubset(selected)


def test_fixed_candidate_rule_supports_dream_28_head_inventory():
    selected, reasons = candidate_set_12(_screen(28))
    assert len(selected) == 12
    assert {"head_27", "head_26", "head_25", "head_24"}.issubset(selected)
    assert {"head_0", "head_1", "head_2", "head_3"}.issubset(selected)
    assert {"head_23", "head_22", "head_5", "head_4"}.issubset(selected)
    assert reasons["head_27"] == ["top_core_rank_1_4"]
    assert reasons["head_0"] == ["bottom_core_rank_25_28"]


def test_main_only_probe_matches_main_columns_of_full_logical_probe_exactly():
    rng = np.random.default_rng(7)
    train = pd.DataFrame(
        {
            "label": np.resize(np.array(["NOUN", "VERB", "ADJ"]), 90),
            "feature": list(rng.normal(size=(90, 8))),
        }
    )
    test = pd.DataFrame(
        {
            "label": np.resize(np.array(["NOUN", "VERB", "ADJ"]), 30),
            "feature": list(rng.normal(size=(30, 8))),
        }
    )
    main_evidence, main_metrics = paper_pos._fit_evaluate_main_only(
        train, test, seed=42, regularization=1.0
    )
    full_evidence, full_metrics = paper_pos._fit_evaluate_probe(
        train, test, seed=42, regularization=1.0
    )
    pd.testing.assert_series_equal(main_evidence.prediction, full_evidence.prediction)
    pd.testing.assert_series_equal(
        main_evidence.majority_prediction, full_evidence.majority_prediction
    )
    for name in (
        "accuracy",
        "macro_f1",
        "majority_accuracy",
        "n_positions",
        "class_counts",
    ):
        assert main_metrics[name] == full_metrics[name]


def test_causal_ablation_accepts_hash_verified_fixed_12_bundle(tmp_path, monkeypatch):
    choices = pd.DataFrame(
        [
            {
                "relative_label": depth,
                "actual_layer_index": layer,
                "high_feature_kind": high,
                "low_feature_kind": low,
                "confirmed": True,
            }
            for depth, layer, high, low in (
                ("early", 6, "head_3", "head_1"),
                ("middle", 16, "head_4", "head_2"),
                ("late", 28, "head_5", "head_0"),
            )
        ]
    )
    rankings = pd.DataFrame(
        [{"relative_label": "middle", "feature_kind": "head_4", "accuracy_mean": 0.5}]
    )
    coverage = pd.DataFrame(
        [
            {
                "seed": 42,
                "normalized_progress": 0.5,
                "relative_label": "middle",
                "feature_kind": "head_4",
            }
        ]
    )
    choices.to_csv(tmp_path / "pos_head_choices.csv", index=False)
    rankings.to_csv(tmp_path / "pos_head_rankings_adaptive_12.csv", index=False)
    coverage.to_csv(tmp_path / "coverage_manifest.csv", index=False)
    manifest = {
        "schema_version": "dlmrel-pos-adaptive-12-v1",
        "status": "confirmed",
        "adaptive_reduced_grid": True,
        "matched_causal_ablation_allowed": True,
        "confirmed_depths": ["early", "middle", "late"],
        "choices_file_hash": canonical_hash(dataframe_records(choices)),
        "ranking_file_hash": canonical_hash(dataframe_records(rankings)),
        "coverage_manifest_hash": canonical_hash(dataframe_records(coverage)),
    }
    (tmp_path / "adaptive_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setenv("DLMREL_POS_HEAD_RANKINGS", str(tmp_path))
    assert paper_causal._pos_control_pairs() == [
        ("most_pos_decodable_primary_p050_fixed12_early", 6, 3),
        ("lower_pos_decoding_primary_p050_fixed12_early", 6, 1),
        ("most_pos_decodable_primary_p050_fixed12_late", 28, 5),
        ("lower_pos_decoding_primary_p050_fixed12_late", 28, 0),
        ("most_pos_decodable_primary_p050_fixed12_middle", 16, 4),
        ("lower_pos_decoding_primary_p050_fixed12_middle", 16, 2),
    ]

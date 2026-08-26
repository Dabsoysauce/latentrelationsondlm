from __future__ import annotations

import json

import pandas as pd
import pytest

from dlmrel.artifacts import ArtifactError, canonical_hash, dataframe_records
from dlmrel.config import RunConfig
from dlmrel.experiments import paper_causal
from dlmrel.experiments.paper_pos_adaptive import (
    HEADS,
    _confirmation,
    candidate_set,
    full_head_grid,
)


def _screen():
    return pd.DataFrame(
        {
            "feature_kind": list(HEADS),
            "accuracy": [0.20 + head * 0.01 for head in range(32)],
            "n_positions": [10_000] * 32,
        }
    )


def test_full_head_grid_is_the_exact_1152_fit_protocol():
    cfg = RunConfig.load_files(
        "configs/models/diffullama_7b.yaml",
        "configs/datasets/ewt.yaml",
        "configs/experiments/pos_token_class_linear_probes.yaml",
    )
    grid = full_head_grid(cfg)
    assert len(grid) == 3 * 4 * 3 * 32 == 1152
    assert len(set(grid)) == len(grid)


def test_candidate_selection_is_deterministic_and_retains_both_extremes():
    first, reasons = candidate_set(_screen())
    second, second_reasons = candidate_set(_screen().sample(frac=1.0, random_state=7))
    assert first == second
    assert reasons == second_reasons
    assert {"head_31", "head_30", "head_29", "head_28"}.issubset(first)
    assert {"head_0", "head_1", "head_2", "head_3"}.issubset(first)
    assert all(reasons[head] for head in first)


def test_confirmation_requires_held_out_seed_rank_stability():
    candidates = {f"head_{head}" for head in range(8)}
    rows = []
    for seed in (42, 43, 44):
        for head in range(8):
            rows.append(
                {
                    "seed": seed,
                    "normalized_progress": 0.5,
                    "relative_label": "middle",
                    "feature_kind": f"head_{head}",
                    "accuracy": 0.20 + head * 0.02,
                }
            )
    result = _confirmation("middle", pd.DataFrame(rows), candidates)
    assert result["confirmed"] is True
    assert result["high_feature_kind"] == "head_7"
    assert result["low_feature_kind"] == "head_0"

    unstable = pd.DataFrame(rows)
    unstable.loc[unstable.seed == 43, "accuracy"] = unstable[
        unstable.seed == 43
    ].accuracy.tolist()[::-1]
    assert _confirmation("middle", unstable, candidates)["confirmed"] is False


def _adaptive_bundle(path, *, status="confirmed"):
    choices = pd.DataFrame(
        [
            {
                "relative_label": "early",
                "actual_layer_index": 6,
                "high_feature_kind": "head_3",
                "low_feature_kind": "head_1",
                "confirmed": True,
            },
            {
                "relative_label": "middle",
                "actual_layer_index": 16,
                "high_feature_kind": "head_4",
                "low_feature_kind": "head_2",
                "confirmed": True,
            },
            {
                "relative_label": "late",
                "actual_layer_index": 28,
                "high_feature_kind": "head_5",
                "low_feature_kind": "head_0",
                "confirmed": True,
            },
        ]
    )
    rankings = pd.DataFrame(
        [{"relative_label": "early", "feature_kind": "head_3", "accuracy_mean": 0.5}]
    )
    coverage = pd.DataFrame(
        [
            {
                "seed": 42,
                "normalized_progress": 0.5,
                "relative_label": "early",
                "feature_kind": "head_3",
                "included": True,
                "phase": "screen",
                "reason": "fixture",
            }
        ]
    )
    choices.to_csv(path / "pos_head_choices.csv", index=False)
    rankings.to_csv(path / "pos_head_rankings_adaptive.csv", index=False)
    coverage.to_csv(path / "coverage_manifest.csv", index=False)
    manifest = {
        "schema_version": "dlmrel-pos-adaptive-v1",
        "status": status,
        "adaptive_reduced_grid": True,
        "matched_causal_ablation_allowed": status == "confirmed",
        "choices_file_hash": canonical_hash(dataframe_records(choices)),
        "ranking_file_hash": canonical_hash(dataframe_records(rankings)),
        "coverage_manifest_hash": canonical_hash(dataframe_records(coverage)),
    }
    (path / "adaptive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_matched_ablation_accepts_only_hash_verified_confirmed_adaptive_bundle(
    tmp_path, monkeypatch
):
    _adaptive_bundle(tmp_path)
    monkeypatch.setenv("DLMREL_POS_HEAD_RANKINGS", str(tmp_path))
    assert paper_causal._pos_control_pairs() == [
        ("most_pos_decodable_primary_p050_early", 6, 3),
        ("lower_pos_decoding_primary_p050_early", 6, 1),
        ("most_pos_decodable_primary_p050_late", 28, 5),
        ("lower_pos_decoding_primary_p050_late", 28, 0),
        ("most_pos_decodable_primary_p050_middle", 16, 4),
        ("lower_pos_decoding_primary_p050_middle", 16, 2),
    ]

    frame = pd.read_csv(tmp_path / "pos_head_choices.csv")
    frame.loc[0, "high_feature_kind"] = "head_31"
    frame.to_csv(tmp_path / "pos_head_choices.csv", index=False)
    with pytest.raises(ArtifactError, match="does not match"):
        paper_causal._pos_control_pairs()


def test_matched_ablation_fails_closed_for_unconfirmed_adaptive_bundle(tmp_path, monkeypatch):
    _adaptive_bundle(tmp_path, status="blocked_unstable_rankings")
    monkeypatch.setenv("DLMREL_POS_HEAD_RANKINGS", str(tmp_path))
    with pytest.raises(ArtifactError, match="not passed held-out confirmation"):
        paper_causal._pos_control_pairs()

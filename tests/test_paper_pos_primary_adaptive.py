from __future__ import annotations

import json

import pandas as pd
import pytest

from dlmrel.artifacts import ArtifactError, canonical_hash, dataframe_records
from dlmrel.experiments import paper_causal
from dlmrel.experiments.paper_pos_adaptive import HEADS, _aggregate_rankings
from dlmrel.experiments.paper_pos_primary_adaptive import (
    SCHEMA,
    validate_primary_adaptive,
)


def _bundle(path):
    coverage_rows = []
    for seed in (42, 43, 44):
        for progress in (0.0, 0.25, 0.5, 0.75):
            for depth in ("early", "middle", "late"):
                for head in HEADS:
                    primary = progress == 0.5 and depth == "middle"
                    canonical = primary and (seed == 42 or head in {"head_0", "head_31"})
                    coverage_rows.append(
                        {
                            "seed": seed,
                            "normalized_progress": progress,
                            "relative_label": depth,
                            "feature_kind": head,
                            "included": primary,
                            "evidence_tier": (
                                "canonical_full_controls"
                                if canonical
                                else "ranking_main_only"
                                if primary
                                else "omitted"
                            ),
                            "phase": "test" if primary else "omitted",
                            "reason": "test fixture",
                            "supports_primary_claim": primary,
                        }
                    )
    coverage = pd.DataFrame(coverage_rows)
    metric_rows = []
    for seed in (42, 43, 44):
        for index, head in enumerate(HEADS):
            metric_rows.append(
                {
                    "seed": seed,
                    "normalized_progress": 0.5,
                    "relative_label": "middle",
                    "feature_kind": head,
                    "accuracy": 0.2 + index / 100 + seed / 1_000_000,
                    "macro_f1": 0.1 + index / 200,
                    "majority_accuracy": 0.2,
                    "n_positions": 1000,
                    "class_counts": {"NOUN": 500, "VERB": 500},
                }
            )
    metrics = pd.DataFrame(metric_rows)
    rankings = _aggregate_rankings(metrics)
    rankings["primary_condition_only"] = True
    choices = pd.DataFrame(
        [
            {
                "relative_label": "middle",
                "actual_layer_index": 16,
                "confirmed": True,
                "high_feature_kind": "head_31",
                "low_feature_kind": "head_0",
            }
        ]
    )
    residual = pd.DataFrame(
        [
            {
                "seed": seed,
                "normalized_progress": 0.5,
                "relative_label": "middle",
                "feature_kind": "residual",
                "accuracy": 0.4,
            }
            for seed in (42, 43, 44)
        ]
    )
    selection = {
        "schema_version": SCHEMA,
        "candidates": list(HEADS),
        "reasons": {head: ["fixture"] for head in HEADS},
    }
    benchmark = {
        "schema_version": SCHEMA,
        "sequential_parallel_exact": True,
        "main_full_exact": True,
    }
    coverage.to_csv(path / "coverage_manifest.csv", index=False)
    metrics.to_csv(path / "per_seed_metrics_primary.csv", index=False)
    rankings.to_csv(path / "pos_head_rankings_primary.csv", index=False)
    choices.to_csv(path / "pos_head_choices.csv", index=False)
    residual.to_csv(path / "primary_residual_metrics.csv", index=False)
    (path / "candidate_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (path / "preregistration.json").write_text("{}", encoding="utf-8")
    (path / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    saved_coverage = pd.read_csv(path / "coverage_manifest.csv")
    saved_rankings = pd.read_csv(path / "pos_head_rankings_primary.csv")
    saved_choices = pd.read_csv(path / "pos_head_choices.csv")
    manifest = {
        "schema_version": SCHEMA,
        "status": "confirmed",
        "adaptive_reduced_grid": True,
        "primary_condition_only": True,
        "matched_causal_ablation_allowed": True,
        "confirmed_depths": ["middle"],
        "choices_file_hash": canonical_hash(dataframe_records(saved_choices)),
        "ranking_file_hash": canonical_hash(dataframe_records(saved_rankings)),
        "coverage_manifest_hash": canonical_hash(dataframe_records(saved_coverage)),
    }
    (path / "adaptive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_primary_bundle_validates_and_exposes_only_middle_causal_pair(tmp_path, monkeypatch):
    _bundle(tmp_path)
    result = validate_primary_adaptive(tmp_path)
    assert result["valid"] is True
    assert result["confirmed_depths"] == ["middle"]
    monkeypatch.setenv("DLMREL_POS_HEAD_RANKINGS", str(tmp_path))
    assert paper_causal._pos_control_pairs() == [
        ("most_pos_decodable_primary_p050_only_middle", 16, 31),
        ("lower_pos_decoding_primary_p050_only_middle", 16, 0),
    ]


def test_primary_bundle_fails_closed_when_unconfirmed(tmp_path, monkeypatch):
    manifest = _bundle(tmp_path)
    manifest["status"] = "blocked_unstable_rankings"
    manifest["matched_causal_ablation_allowed"] = False
    (tmp_path / "adaptive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="have not passed confirmation"):
        validate_primary_adaptive(tmp_path)
    monkeypatch.setenv("DLMREL_POS_HEAD_RANKINGS", str(tmp_path))
    with pytest.raises(ArtifactError, match="have not passed held-out confirmation"):
        paper_causal._pos_control_pairs()


def test_primary_validator_rejects_noncanonical_screen_controls(tmp_path):
    _bundle(tmp_path)
    coverage_path = tmp_path / "coverage_manifest.csv"
    coverage = pd.read_csv(coverage_path)
    row = (
        (coverage.seed == 42)
        & (coverage.normalized_progress == 0.5)
        & (coverage.relative_label == "middle")
        & (coverage.feature_kind == "head_7")
    )
    coverage.loc[row, "evidence_tier"] = "ranking_main_only"
    coverage.to_csv(coverage_path, index=False)
    manifest = json.loads((tmp_path / "adaptive_manifest.json").read_text())
    manifest["coverage_manifest_hash"] = canonical_hash(dataframe_records(coverage))
    (tmp_path / "adaptive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="screen is missing canonical control"):
        validate_primary_adaptive(tmp_path)

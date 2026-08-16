from __future__ import annotations

import pandas as pd
import pytest

from dlmrel.cli import _build_config, _experiment_config
from dlmrel.experiments.time_curve import selected_heads

MERGED = pd.DataFrame(
    {
        "relation": ["object_to_verb"] * 3 + ["subject_to_verb"] * 3,
        "layer": [3, 18, 7, 24, 2, 9],
        "head": [11, 10, 4, 10, 1, 3],
        "accuracy_select": [0.86, 0.41, 0.12, 0.62, 0.30, 0.08],
        "accuracy_test": [0.85, 0.40, 0.11, 0.61, 0.29, 0.09],
    }
)

HEADLINE = pd.DataFrame(
    {
        "relation": ["object_to_verb", "subject_to_verb"],
        "layer": [3, 24],
        "head": [11, 10],
        "head_select_acc": [0.86, 0.62],
    }
)

EXPECTED = {"object_to_verb": (3, 11), "subject_to_verb": (24, 10)}


def test_selected_heads_prefers_merged_scores(tmp_path):
    MERGED.to_csv(tmp_path / "head_scores_merged.csv", index=False)
    assert selected_heads(tmp_path) == EXPECTED


def test_selected_heads_falls_back_to_headline_table(tmp_path):
    """The fallback must pick the same head the merged scores would."""
    HEADLINE.to_csv(tmp_path / "head_vs_null.csv", index=False)
    assert selected_heads(tmp_path) == EXPECTED


def test_selected_heads_agree_across_sources(tmp_path):
    MERGED.to_csv(tmp_path / "head_scores_merged.csv", index=False)
    from_merged = selected_heads(tmp_path)

    (tmp_path / "head_scores_merged.csv").unlink()
    HEADLINE.to_csv(tmp_path / "head_vs_null.csv", index=False)
    assert selected_heads(tmp_path) == from_merged


def test_selected_heads_reports_missing_search(tmp_path):
    with pytest.raises(FileNotFoundError, match="run head_search first"):
        selected_heads(tmp_path)


def test_model_config_without_checkpoint_is_rejected():
    """A bare name would be resolved as a Hugging Face repo id and 404."""
    with pytest.raises(KeyError, match="no `checkpoint`"):
        _build_config("bogus", {"name": "bogus", "adapter": "dream"}, {})


def test_experiment_yaml_knobs_reach_the_config():
    cfg = _build_config(
        "diffullama_7b",
        {"name": "diffullama_7b", "checkpoint": "org/model", "adapter": "diffullama"},
        _experiment_config("time_curve"),
    )
    assert cfg.model.name == "org/model"
    # Regression: these were silently dropped, so every run used the defaults
    # and the curve tried all 1000 test sentences.
    assert cfg.diffusion.n_curve_sentences == 300
    assert cfg.diffusion.timesteps == [0, 8, 16, 24, 32, 40, 48, 56]


def test_model_yaml_dtype_and_attention_reach_the_config():
    cfg = _build_config(
        "m",
        {
            "name": "m",
            "checkpoint": "org/model",
            "adapter": "diffullama",
            "dtype": "bfloat16",
            "attn_implementation": "eager",
        },
        {},
    )
    assert cfg.model.dtype == "bfloat16"
    assert cfg.model.attn_implementation == "eager"


def test_unknown_yaml_keys_are_ignored():
    """The experiment YAMLs carry descriptive keys that configure nothing."""
    cfg = _build_config(
        "m",
        {"name": "m", "checkpoint": "org/model", "adapter": "dream"},
        {"classifier": "logistic_regression", "measure_bos_mass": True, "seeds": [1]},
    )
    assert cfg.diffusion.seeds == [1]

"""A new runtime knob must never retroactively invalidate a completed run."""

from __future__ import annotations

import dataclasses

import pytest

from dlmrel.config import ConfigError, RunConfig, RuntimeConfig

# Every field here is execution metadata. If a scientific value is ever added to
# RuntimeConfig this test fails, forcing a deliberate decision rather than
# silently letting that value default when an old config is re-read.
EXECUTION_ONLY_RUNTIME_FIELDS = {
    "results_root",
    "run_id",
    "resume",
    "dry_run",
    "selection_lock",
    "timestep_batch_size",
    "native_batch_size",
    "intervention_batch_size",
    "sentence_batch_size",
    "adaptive_batch_max_size",
    "pos_fit_workers",
    "export_attention_cache",
    "attention_cache",
    "pos_stage",
}


def _resolved() -> dict:
    return RunConfig.load_files(
        "configs/models/dream_7b.yaml",
        "configs/datasets/ewt.yaml",
        "configs/experiments/relation_head_receiver_prediction.yaml",
        runtime=RuntimeConfig(),
    ).to_dict()


def test_runtime_config_still_contains_only_execution_fields():
    actual = {item.name for item in dataclasses.fields(RuntimeConfig)}
    assert actual == EXECUTION_ONLY_RUNTIME_FIELDS, (
        "RuntimeConfig changed. Defaulting a missing runtime field on load is only "
        "safe while every field is execution metadata; re-check before updating."
    )


@pytest.mark.parametrize("absent", sorted(EXECUTION_ONLY_RUNTIME_FIELDS))
def test_config_stored_before_a_runtime_field_existed_still_validates(absent):
    """Reproduces the pos_stage regression for every runtime knob."""
    current = _resolved()
    legacy = {
        **current,
        "runtime": {k: v for k, v in current["runtime"].items() if k != absent},
    }
    restored = RunConfig.from_dict(legacy)
    expected = getattr(RuntimeConfig(), absent)
    assert getattr(restored.runtime, absent) == expected


def test_missing_pos_stage_specifically_no_longer_fails():
    """The exact failure seen by `dlmrel validate` on a completed run."""
    current = _resolved()
    legacy = {
        **current,
        "runtime": {k: v for k, v in current["runtime"].items() if k != "pos_stage"},
    }
    assert RunConfig.from_dict(legacy).runtime.pos_stage == "all"


@pytest.mark.parametrize(
    ("section", "absent"),
    [
        ("model", "revision"),
        ("model", "tokenizer_revision"),
        ("dataset", "revision"),
        ("experiment", "seeds"),
        ("experiment", "normalized_progress"),
    ],
)
def test_missing_scientific_fields_are_still_rejected(section, absent):
    """The compatibility path must not leak into anything scientific."""
    current = _resolved()
    broken = {
        **current,
        section: {k: v for k, v in current[section].items() if k != absent},
    }
    with pytest.raises(ConfigError, match=f"missing required config.{section} field: {absent}"):
        RunConfig.from_dict(broken)


def test_unknown_runtime_fields_are_still_rejected():
    current = _resolved()
    unknown = {**current, "runtime": {**current["runtime"], "bogus": 1}}
    with pytest.raises(ConfigError, match="unknown config.runtime field"):
        RunConfig.from_dict(unknown)


def test_scoring_is_nested_and_still_strict():
    """Nested scientific dataclasses keep requiring every field."""
    current = _resolved()
    scoring = dict(current["experiment"]["scoring"])
    scoring.pop("primary_relation")
    broken = {
        **current,
        "experiment": {**current["experiment"], "scoring": scoring},
    }
    with pytest.raises(ConfigError, match="missing required"):
        RunConfig.from_dict(broken)

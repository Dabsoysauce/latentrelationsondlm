"""Equivalence and fail-closed tests for the shared native trajectory cache."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
import pytest
import torch

from dlmrel.config import ExperimentConfig, ModelConfig, RunConfig, RuntimeConfig
from dlmrel.experiments.paper_native import generate_trajectories
from dlmrel.models.base import NativeTrajectory
from dlmrel.native_cache import NativeCacheIdentity, NativeTrajectoryCache

PROMPT_MANIFEST = "configs/prompts/paper_reasoning_creative.json"


@dataclass(frozen=True)
class _Prompt:
    sentence_id: str


BASE_IDENTITY = NativeCacheIdentity(
    model_id="fake",
    model_revision="rev-1",
    tokenizer_revision="rev-1",
    remote_code_revision="code-1",
    prompt_manifest_hash="manifest-1",
    steps=64,
    generation_length=96,
    temperature=0.95,
    top_p=0.9,
    reveal_policy="random_one_over_remaining_steps",
    prediction_offset=0,
)


def _frame(prompt_id: str, seed: int) -> pd.DataFrame:
    return pd.DataFrame([{"prompt_id": prompt_id, "seed": seed, "value": f"{prompt_id}-{seed}"}])


def test_second_cache_reuses_first_with_no_generation_calls(tmp_path):
    calls = []

    def generate_one(example, seed):
        calls.append((example.sentence_id, seed))
        return _frame(example.sentence_id, seed)

    prompts = [_Prompt("p0"), _Prompt("p1")]
    first = NativeTrajectoryCache(tmp_path, BASE_IDENTITY)
    first.get_or_generate(prompts, [42, 43], generate_one)
    assert len(calls) == 4

    calls.clear()
    second = NativeTrajectoryCache(tmp_path, BASE_IDENTITY)
    reused = second.get_or_generate(prompts, [42, 43], generate_one)
    assert calls == []
    assert len(reused) == 4


def test_interrupted_generation_resumes_only_missing_pairs(tmp_path):
    calls = []

    def generate_one(example, seed):
        calls.append((example.sentence_id, seed))
        return _frame(example.sentence_id, seed)

    prompts = [_Prompt("p0"), _Prompt("p1"), _Prompt("p2")]
    cache = NativeTrajectoryCache(tmp_path, BASE_IDENTITY)
    cache.store("p0", 42, _frame("p0", 42))
    cache.store("p1", 42, _frame("p1", 42))

    cache.get_or_generate(prompts, [42], generate_one)
    assert calls == [("p2", 42)]


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "other"),
        ("model_revision", "rev-2"),
        ("tokenizer_revision", "rev-2"),
        ("remote_code_revision", "code-2"),
        ("prompt_manifest_hash", "manifest-2"),
        ("generation_length", 64),
        ("temperature", 0.5),
        ("top_p", 0.5),
        ("reveal_policy", "other_policy"),
        ("prediction_offset", 1),
    ],
)
def test_any_identity_field_change_rejects_the_cache(tmp_path, field, value):
    calls = []

    def generate_one(example, seed):
        calls.append((example.sentence_id, seed))
        return _frame(example.sentence_id, seed)

    prompts = [_Prompt("p0")]
    first = NativeTrajectoryCache(tmp_path, BASE_IDENTITY)
    first.get_or_generate(prompts, [42], generate_one)
    assert len(calls) == 1

    calls.clear()
    changed_identity = replace(BASE_IDENTITY, **{field: value})
    second = NativeTrajectoryCache(tmp_path, changed_identity)
    second.get_or_generate(prompts, [42], generate_one)
    assert len(calls) == 1


def test_cache_directory_is_keyed_by_identity_hash_not_reused_across_identities(tmp_path):
    first = NativeTrajectoryCache(tmp_path, BASE_IDENTITY)
    other_identity = replace(BASE_IDENTITY, model_id="different")
    second = NativeTrajectoryCache(tmp_path, other_identity)
    assert first.directory != second.directory


def test_corrupted_row_count_metadata_is_treated_as_a_miss(tmp_path):
    cache = NativeTrajectoryCache(tmp_path, BASE_IDENTITY)
    cache.store("p0", 42, _frame("p0", 42))
    path, meta_path = cache._paths("p0", 42)
    metadata = meta_path.read_text(encoding="utf-8").replace('"row_count": 1', '"row_count": 99')
    meta_path.write_text(metadata, encoding="utf-8")
    assert cache.load("p0", 42) is None


class _FakeNativeModel:
    """Minimal native-trajectory model: two masked-then-revealed positions."""

    prediction_offset = 0
    device = "cpu"

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def native_trajectory(self, prompt, *, seed, steps, generation_length, temperature, top_p):
        self.calls.append((prompt, seed))
        step = torch.tensor([1, 1])
        return NativeTrajectory(
            prompt=prompt,
            prefix_length=1,
            pre_forward_ids=tuple(step for _ in range(64)),
            argmax_ids=tuple(step for _ in range(64)),
            final_ids=torch.tensor([1, 2]),
            metadata={"seed": seed},
        )


def _paper_run_config(experiment_id: str, tmp_path) -> RunConfig:
    return RunConfig(
        track="exploratory_extensions",
        model=ModelConfig(
            id="fake_native",
            name="fake_native",
            family="fake",
            revision="rev-1",
            tokenizer_revision="rev-1",
            remote_code_revision="code-1",
        ),
        experiment=ExperimentConfig(
            id=experiment_id,
            type=experiment_id,
            seeds=[42, 43, 44],
            settings={
                "prompt_manifest": PROMPT_MANIFEST,
                "generation_length": 96,
                "temperature": 0.95,
                "top_p": 0.9,
                "reveal_policy": "random_one_over_remaining_steps",
            },
        ),
        runtime=RuntimeConfig(results_root=str(tmp_path / "results")),
    )


def test_two_paper_experiments_share_one_native_generation_pass(tmp_path):
    model = _FakeNativeModel()
    final_token_dir = tmp_path / "final_token_run"
    final_token_dir.mkdir()
    final_token_cfg = _paper_run_config("final_token_prediction_by_layer", tmp_path)
    generate_trajectories(model, final_token_cfg, final_token_dir)
    assert len(model.calls) == 24 * 3

    model.calls.clear()
    timing_dir = tmp_path / "timing_run"
    timing_dir.mkdir()
    trajectories, prompts, _hash = generate_trajectories(
        model, _paper_run_config("prediction_before_unmasking_timing_analysis", tmp_path), timing_dir
    )
    assert model.calls == []
    assert len(trajectories) == 24 * 3
    assert len(prompts) == 24

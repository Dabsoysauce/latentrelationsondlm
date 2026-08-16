from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dlmrel.artifacts import initialize_run
from dlmrel.checkpoints import SentenceCheckpointStore
from dlmrel.experiments import pos_probe
from dlmrel.experiments.pos_probe import aggregate_probe_metrics
from dlmrel.relations import Example


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


def test_pos_runner_evaluates_every_protocol_seed(tmp_path, monkeypatch):
    calls = []

    def fake_load(_cfg, _tokenizer, role):
        return [role], pd.DataFrame()

    def fake_features(
        _model, _tokenizer, _examples, _cfg, *, seed, role, checkpoint_store
    ):
        assert checkpoint_store is not None
        calls.append((seed, role))
        return seed

    def fake_evaluate(collected, seed):
        assert set(collected) == {"select", "dev", "test"}
        assert set(collected.values()) == {seed}
        raw = pd.DataFrame({"seed": [seed], "sentence_id": [f"sentence-{seed}"]})
        metrics = {
            "seed": seed,
            "selected_c": 1.0,
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "majority_accuracy": 0.5,
            "lexical_accuracy": 0.5,
            "shuffled_accuracy": 0.5,
            "random_feature_accuracy": 0.5,
            "n_test_positions": 1,
            "n_test_sentences": 1,
        }
        return raw, metrics

    monkeypatch.setattr(pos_probe, "load_manifest_examples", fake_load)
    monkeypatch.setattr(pos_probe, "masked_features", fake_features)
    monkeypatch.setattr(pos_probe, "evaluate_seed", fake_evaluate)
    monkeypatch.setattr(pos_probe, "write_frames", lambda *_args, **_kwargs: None)
    run = tmp_path / "run"
    initialize_run(run, {"runtime": {}}, "command", {"test": "hash"})
    cfg = SimpleNamespace(experiment=SimpleNamespace(seeds=[42, 43, 44]))

    details = pos_probe.run(object(), object(), cfg, run)

    assert calls == [
        (seed, role)
        for seed in (42, 43, 44)
        for role in ("select", "dev", "test")
    ]
    assert details["n_seeds"] == 3
    assert pd.read_csv(run / "per_seed_metrics.csv")["seed"].tolist() == [42, 43, 44]


def test_pos_features_round_trip_through_sentence_checkpoints(tmp_path, monkeypatch):
    calls = []

    def fake_states(_model, _tokenizer, text, _time, _steps, seed, _include_bos):
        calls.append((text, seed))
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        state = SimpleNamespace(is_visible=[False, False])
        return None, (hidden, hidden), state

    monkeypatch.setattr(pos_probe, "states_at_time", fake_states)
    examples = [
        Example(
            text=f"word {index}",
            tokens=["word"],
            upos=["NOUN"],
            deprel=["root"],
            head=[0],
            word_to_tokens={0: [0]},
            relations=[],
            seq_len=2,
            sentence_id=f"sentence-{index}",
        )
        for index in range(2)
    ]
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(normalized_progress=[0.5], steps=64, seeds=[42, 43, 44])
    )
    run = tmp_path / "run"
    initialize_run(run, {"runtime": {}}, "command", {"test": "hash"})
    store = SentenceCheckpointStore(run, chunk_size=1)

    first = pos_probe.masked_features(
        object(), object(), examples, cfg, seed=42, role="test", checkpoint_store=store
    )
    second = pos_probe.masked_features(
        object(), object(), examples, cfg, seed=42, role="test", checkpoint_store=store
    )

    assert calls == [("word 0", 42), ("word 1", 42)]
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)

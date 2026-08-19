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
        calls.append(("load", role))
        return [role], pd.DataFrame()

    def fake_features(
        _model, _tokenizer, _examples, _cfg, *, seed, role, checkpoint_store
    ):
        assert checkpoint_store is not None
        calls.append(("features", seed, role))
        return seed

    def fake_fit(select, dev, seed):
        assert select == dev == seed
        calls.append(("fit", seed))
        return seed

    def fake_evaluate(fitted, test, seed):
        assert fitted == test == seed
        calls.append(("evaluate", seed))
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
    monkeypatch.setattr(pos_probe, "fit_probe", fake_fit)
    monkeypatch.setattr(pos_probe, "evaluate_fitted_probe", fake_evaluate)
    monkeypatch.setattr(pos_probe, "write_frames", lambda *_args, **_kwargs: None)
    run = tmp_path / "run"
    initialize_run(run, {"runtime": {}}, "command", {"test": "hash"})
    cfg = SimpleNamespace(experiment=SimpleNamespace(seeds=[42, 43, 44]))

    details = pos_probe.run(object(), object(), cfg, run)

    test_load_index = calls.index(("load", "test"))
    assert calls[:test_load_index] == [
        ("load", "select"),
        ("load", "dev"),
        ("features", 42, "select"),
        ("features", 42, "dev"),
        ("fit", 42),
        ("features", 43, "select"),
        ("features", 43, "dev"),
        ("fit", 43),
        ("features", 44, "select"),
        ("features", 44, "dev"),
        ("fit", 44),
    ]
    assert calls[test_load_index + 1 :] == [
        item
        for seed in (42, 43, 44)
        for item in (("features", seed, "test"), ("evaluate", seed))
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


def _features(values, labels, prefix):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels)
    groups = np.asarray([f"{prefix}-{index // 2}" for index in range(len(labels))])
    forms = np.asarray([f"form-{index % 3}" for index in range(len(labels))])
    word_indices = np.arange(len(labels))
    return values, labels, groups, forms, word_indices


def test_probe_scaler_and_c_use_select_and_dev_only_with_smallest_c_tie():
    select = _features([-3, -2, -1, 1, 2, 3], ["N", "N", "N", "V", "V", "V"], "s")
    dev = _features([-2.5, -0.5, 0.5, 2.5], ["N", "N", "V", "V"], "d")

    fitted = pos_probe.fit_probe(select, dev, seed=42)

    assert fitted.scaler.mean_[0] == pytest.approx(np.mean(select[0]))
    assert fitted.selected_c == 0.01


def test_probe_metrics_and_controls_are_reproducible_after_c_is_frozen():
    select = _features([-3, -2, -1, 1, 2, 3], ["N", "N", "N", "V", "V", "V"], "s")
    dev = _features([-2.5, -0.5, 0.5, 2.5], ["N", "N", "V", "V"], "d")
    test = _features([-4, -0.25, 0.25, 4], ["N", "N", "V", "V"], "t")
    fitted = pos_probe.fit_probe(select, dev, seed=43)

    first_rows, first_metrics = pos_probe.evaluate_fitted_probe(fitted, test, seed=43)
    second_rows, second_metrics = pos_probe.evaluate_fitted_probe(fitted, test, seed=43)

    pd.testing.assert_frame_equal(first_rows, second_rows)
    assert first_metrics == second_metrics
    assert first_metrics["accuracy"] == 1.0
    assert first_metrics["macro_f1"] == 1.0


@pytest.mark.parametrize(
    "select,dev,error",
    [
        (
            (
                np.asarray([]),
                np.asarray([]),
                np.asarray([]),
                np.asarray([]),
                np.asarray([]),
            ),
            _features([0, 1], ["N", "V"], "d"),
            "nonempty 2D matrix",
        ),
        (
            _features([0, 1], ["N", "N"], "s"),
            _features([0, 1], ["N", "V"], "d"),
            "at least two classes",
        ),
    ],
)
def test_probe_invalid_tiny_or_single_class_splits_fail_usefully(select, dev, error):
    with pytest.raises(ValueError, match=error):
        pos_probe.fit_probe(select, dev, seed=42)

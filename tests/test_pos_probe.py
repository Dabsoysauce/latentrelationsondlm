from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from dlmrel.artifacts import initialize_run
from dlmrel.checkpoints import SentenceCheckpointStore
from dlmrel.experiments import pos_probe
from dlmrel.relations import Example


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
        experiment=SimpleNamespace(normalized_progress=[0.5], steps=64, seeds=[42])
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

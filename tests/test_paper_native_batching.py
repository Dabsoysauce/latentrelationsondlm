"""Equivalence tests for final-token hidden-state microbatching."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from dlmrel.experiments.paper_native import final_token_rows
from dlmrel.paper_protocol import map_relative_depths


class _Tokenizer:
    mask_token_id = 0


class _FinalTokenModel:
    """Deterministic hidden states/logits, batch-shape-agnostic like a real adapter."""

    prediction_offset = 0
    device = "cpu"

    def __init__(self, layers: int = 2, hidden: int = 6, vocab: int = 10):
        self.n_layers, self.hidden, self.vocab = layers, hidden, vocab
        self.tokenizer = _Tokenizer()
        self.forward_calls = 0
        self.batch_sizes_seen: list[int] = []
        torch.manual_seed(3)
        self.unembed = torch.randn(hidden, vocab)

    def forward_attentions(self, input_ids, output_hidden_states: bool = False):
        self.forward_calls += 1
        self.batch_sizes_seen.append(input_ids.shape[0])
        basis = torch.nn.functional.one_hot(input_ids % self.hidden, self.hidden).float()
        hidden_states = tuple(basis + layer * 0.1 for layer in range(self.n_layers + 1))
        logits = hidden_states[-1] @ self.unembed
        return logits, (), hidden_states

    def get_final_norm(self):
        return torch.nn.Identity()

    def get_lm_head(self):
        return lambda x: x @ self.unembed


def _row(prefix_length: int, seq_len: int, n_steps: int = 12):
    torch.manual_seed(11)
    final_ids = torch.randint(0, 10, (seq_len,)).tolist()
    pre_forward_ids = []
    for step in range(n_steps):
        state = list(final_ids)
        reveal_before = min(prefix_length + step, seq_len)
        for position in range(reveal_before, seq_len):
            state[position] = 0  # mask id
        pre_forward_ids.append(state)
    return SimpleNamespace(
        prompt_id="p0",
        task="reasoning",
        seed=42,
        prefix_length=prefix_length,
        pre_forward_ids=pre_forward_ids,
        final_ids=final_ids,
    )


def test_batched_and_unbatched_final_token_rows_are_identical():
    model = _FinalTokenModel()
    depths = map_relative_depths(model.n_layers, {"early": 0.2, "middle": 0.5, "late": 0.9})
    row = _row(prefix_length=3, seq_len=8)

    unbatched, _ = final_token_rows(model, row, depths, collect_probe_features=False, batch_size=1)
    model2 = _FinalTokenModel()
    batched, _ = final_token_rows(model2, row, depths, collect_probe_features=False, batch_size=5)

    pd.testing.assert_frame_equal(unbatched, batched)


def test_batching_reduces_model_forward_calls():
    row = _row(prefix_length=3, seq_len=8, n_steps=12)
    depths = map_relative_depths(2, {"early": 0.2, "middle": 0.5, "late": 0.9})

    model_unbatched = _FinalTokenModel()
    final_token_rows(model_unbatched, row, depths, collect_probe_features=False, batch_size=1)
    assert model_unbatched.forward_calls == 12
    assert set(model_unbatched.batch_sizes_seen) == {1}

    model_batched = _FinalTokenModel()
    final_token_rows(model_batched, row, depths, collect_probe_features=False, batch_size=5)
    assert model_batched.forward_calls == 3  # ceil(12 / 5)
    assert model_batched.batch_sizes_seen == [5, 5, 2]


def test_probe_feature_collection_matches_across_batch_sizes():
    model = _FinalTokenModel()
    depths = map_relative_depths(model.n_layers, {"early": 0.2, "middle": 0.5, "late": 0.9})
    row = _row(prefix_length=3, seq_len=8)

    _, unbatched_features = final_token_rows(
        model, row, depths, collect_probe_features=True, batch_size=1
    )
    model2 = _FinalTokenModel()
    _, batched_features = final_token_rows(
        model2, row, depths, collect_probe_features=True, batch_size=4
    )

    pd.testing.assert_frame_equal(
        unbatched_features.drop(columns="feature"), batched_features.drop(columns="feature")
    )
    for left, right in zip(unbatched_features["feature"], batched_features["feature"], strict=True):
        assert list(left) == list(right)

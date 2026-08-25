"""Equivalence tests for attention heatmap trajectory microbatching."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from test_paper_optimizations import TinyTokenizer

from dlmrel.experiments.paper_visuals import _numeric_attention_matrix, trajectory_chunk


class _HeatmapModel:
    """Batch-aware fake: each item's attention depends only on its own tokens."""

    device = "cpu"

    def __init__(self, layers: int = 2, heads: int = 3):
        self.layers, self.heads = layers, heads
        self.forward_calls = 0
        self.batch_sizes_seen: list[int] = []

    def forward_attentions_only(self, input_ids):
        self.forward_calls += 1
        self.batch_sizes_seen.append(input_ids.shape[0])
        batch, seq = input_ids.shape
        base = input_ids.float().unsqueeze(1).unsqueeze(1).expand(batch, self.heads, seq, seq)
        offsets = torch.arange(seq, dtype=torch.float).view(1, 1, 1, seq)
        attentions = tuple(
            torch.softmax(base + offsets + layer, dim=-1) for layer in range(self.layers)
        )
        return attentions


def test_nested_parquet_object_attention_restores_exact_numeric_matrix():
    expected = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    parquet_value = np.empty(2, dtype=object)
    parquet_value[:] = [row.astype(object) for row in expected]

    actual = _numeric_attention_matrix(parquet_value)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


@dataclass
class _Instance:
    attender_span: list
    receiver_span: list
    attender_word_idx: int
    receiver_word_idx: int


def _case(sentence_id: str, text: str):
    example = SimpleNamespace(text=text, sentence_id=sentence_id)
    instance = _Instance(
        attender_span=[2], receiver_span=[4, 5], attender_word_idx=1, receiver_word_idx=3
    )
    return SimpleNamespace(
        sentence_id=f"{sentence_id}:object_to_verb",
        relation="object_to_verb",
        example=example,
        instance=instance,
    )


class _Lock:
    layer = 1
    head = 2


class _Locks:
    def resolve(self, relation):
        return _Lock()


def test_batched_and_unbatched_trajectory_chunks_are_identical():
    tokenizer = TinyTokenizer()
    cases = [_case("s1", "6")]

    model_unbatched = _HeatmapModel()
    unbatched = trajectory_chunk(model_unbatched, tokenizer, cases, seed=42, locks=_Locks(), batch_size=1)

    model_batched = _HeatmapModel()
    batched = trajectory_chunk(model_batched, tokenizer, cases, seed=42, locks=_Locks(), batch_size=16)

    pd.testing.assert_frame_equal(unbatched, batched)


def test_batching_reduces_model_forward_calls():
    tokenizer = TinyTokenizer()
    cases = [_case("s1", "6")]

    model_unbatched = _HeatmapModel()
    trajectory_chunk(model_unbatched, tokenizer, cases, seed=42, locks=_Locks(), batch_size=1)
    assert model_unbatched.forward_calls == 64  # seed 42 keeps every timestep, none deduplicated

    model_batched = _HeatmapModel()
    trajectory_chunk(model_batched, tokenizer, cases, seed=42, locks=_Locks(), batch_size=16)
    assert model_batched.forward_calls == 4  # ceil(64 / 16)


def test_non_primary_seed_deduplicates_endpoints_the_same_way_batched_or_not():
    tokenizer = TinyTokenizer()
    cases = [_case("s1", "6")]

    model_unbatched = _HeatmapModel()
    unbatched = trajectory_chunk(model_unbatched, tokenizer, cases, seed=43, locks=_Locks(), batch_size=1)
    model_batched = _HeatmapModel()
    batched = trajectory_chunk(model_batched, tokenizer, cases, seed=43, locks=_Locks(), batch_size=16)

    assert sorted(unbatched["timestep"]) == sorted(batched["timestep"])
    assert 0 not in set(unbatched["timestep"])
    assert 63 not in set(unbatched["timestep"])
    pd.testing.assert_frame_equal(unbatched, batched)

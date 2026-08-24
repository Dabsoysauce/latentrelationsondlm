"""Exact-output and call-count tests for the final execution-only optimizations."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from dlmrel.batching import adaptive_forward_batches
from dlmrel.diffusion import TrajectoryStateCache, state_at_time
from dlmrel.models._backbone import WrappedAdapter
from dlmrel.models.native import random_reveal_trajectories, random_reveal_trajectory


class _Tokenizer:
    bos_token_id = 1
    mask_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(character) % 7) for character in text]


class _NativeAdapter:
    device = "cpu"
    prediction_offset = 0

    def __init__(self):
        self.forward_calls = 0

    def forward_logits(self, input_ids):
        self.forward_calls += 1
        vocab = 12
        token = torch.arange(vocab, dtype=torch.float32)
        return -(
            token.view(1, 1, -1)
            - ((input_ids.unsqueeze(-1) + torch.arange(input_ids.shape[1]).view(1, -1, 1)) % vocab)
        ).abs()


def _assert_trajectory_equal(reference, optimized):
    assert reference.prompt == optimized.prompt
    assert reference.prefix_length == optimized.prefix_length
    assert torch.equal(reference.final_ids, optimized.final_ids)
    assert all(
        torch.equal(left, right)
        for left, right in zip(reference.pre_forward_ids, optimized.pre_forward_ids, strict=True)
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(reference.argmax_ids, optimized.argmax_ids, strict=True)
    )


def test_native_prompt_batching_replays_each_legacy_rng_stream_exactly():
    tokenizer = _Tokenizer()
    prompts = ["a", "bc", "def", "ghij"]
    reference_model = _NativeAdapter()
    references = tuple(
        random_reveal_trajectory(
            reference_model,
            tokenizer,
            prompt,
            seed=43,
            generation_length=12,
            temperature=0.95,
            top_p=0.9,
        )
        for prompt in prompts
    )
    optimized_model = _NativeAdapter()
    optimized = random_reveal_trajectories(
        optimized_model,
        tokenizer,
        prompts,
        seed=43,
        generation_length=12,
        temperature=0.95,
        top_p=0.9,
    )
    for reference, batched in zip(references, optimized, strict=True):
        _assert_trajectory_equal(reference, batched)
    assert reference_model.forward_calls == 64 * len(prompts)
    assert optimized_model.forward_calls == 64


def test_adaptive_batches_grow_without_reordering_items():
    seen = []

    def forward(current):
        seen.append(len(current))
        return list(current)

    batches = list(
        adaptive_forward_batches(list(range(40)), forward, initial_size=8, maximum_size=32)
    )
    assert seen == [8, 16, 16]
    assert [item for _start, current, _result in batches for item in current] == list(range(40))


def test_adaptive_batches_back_off_only_on_memory_errors():
    attempts = []

    def forward(current):
        attempts.append(len(current))
        if len(current) > 4:
            raise RuntimeError("CUDA out of memory")
        return list(current)

    batches = list(
        adaptive_forward_batches(list(range(10)), forward, initial_size=8, maximum_size=32)
    )
    assert attempts == [8, 4, 4, 2]
    assert [item for _start, current, _result in batches for item in current] == list(range(10))


class _StateModel:
    device = "cpu"
    mask_free = False


class _StateTokenizer:
    bos_token_id = 1
    mask_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(2, 2 + int(text)))

    def decode(self, token_ids):
        return str(token_ids[0])


def test_selected_trajectory_state_reuse_matches_independent_reconstruction_exactly():
    model, tokenizer = _StateModel(), _StateTokenizer()
    cache = TrajectoryStateCache(model, tokenizer, [0, 20, 40, 63])
    for seed in (42, 43):
        for timestep in (0, 20, 40, 63):
            reference = state_at_time(model, tokenizer, "30", timestep, 64, seed, True)
            reused = cache.get("sentence", "30", seed, timestep)
            assert torch.equal(reference.input_ids, reused.input_ids)
            assert reference.is_visible == reused.is_visible
            assert reference.unmask_step == reused.unmask_step


class _Denoise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *, inputs_embeds, output_attentions, output_hidden_states, **_kwargs):
        self.calls.append((output_attentions, output_hidden_states))
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            attentions=(torch.ones(1),) if output_attentions else None,
            hidden_states=(inputs_embeds, inputs_embeds + 1) if output_hidden_states else None,
        )


class _WrappedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_model = _Denoise()
        self.lm_head = torch.nn.Identity()
        self.logit_calls = 0

    def get_embeds(self, input_ids):
        return input_ids.float().unsqueeze(-1)

    def get_logits(self, hidden_state):
        self.logit_calls += 1
        return hidden_state


def test_wrapped_adapter_requests_only_each_experiment_output(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "model",
        SimpleNamespace(get_anneal_attn_mask=lambda **_kwargs: None),
    )
    backbone = _WrappedBackbone()
    adapter = WrappedAdapter(backbone, tokenizer=None, device="cpu")
    input_ids = torch.tensor([[1, 2]])

    adapter.forward_logits(input_ids)
    assert backbone.denoise_model.calls[-1] == (False, False)
    assert backbone.logit_calls == 1

    hidden = adapter.forward_hidden_states(input_ids)
    assert backbone.denoise_model.calls[-1] == (False, True)
    assert len(hidden) == 2
    assert backbone.logit_calls == 1

    adapter.forward_capture_only(input_ids)
    assert backbone.denoise_model.calls[-1] == (False, False)
    assert backbone.logit_calls == 1

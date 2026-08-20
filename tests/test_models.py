from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dlmrel.models._backbone import _wrapper_key_map


class TestWrapperKeyMap:
    def test_llama_body_prefix_is_rewritten(self):
        assert (
            _wrapper_key_map(
                "denoise_model.layers.0.self_attn.q_proj.weight",
                body="model",
                embed_target="model.embed_tokens.weight",
            )
            == "model.layers.0.self_attn.q_proj.weight"
        )

    def test_gpt2_body_prefix_is_rewritten(self):
        assert (
            _wrapper_key_map(
                "denoise_model.h.0.attn.c_attn.weight",
                body="transformer",
                embed_target="transformer.wte.weight",
            )
            == "transformer.h.0.attn.c_attn.weight"
        )

    def test_hoisted_embedding_goes_back_into_the_body(self):
        assert (
            _wrapper_key_map(
                "embed_tokens.weight", body="model", embed_target="model.embed_tokens.weight"
            )
            == "model.embed_tokens.weight"
        )
        assert (
            _wrapper_key_map(
                "embed_tokens.weight", body="transformer", embed_target="transformer.wte.weight"
            )
            == "transformer.wte.weight"
        )

    def test_lm_head_is_already_top_level(self):
        assert (
            _wrapper_key_map("lm_head.weight", body="model", embed_target="model.embed_tokens.weight")
            == "lm_head.weight"
        )

    def test_unknown_keys_are_reported_rather_than_guessed(self):
        assert (
            _wrapper_key_map("optimizer.state", body="model", embed_target="model.embed_tokens.weight")
            is None
        )

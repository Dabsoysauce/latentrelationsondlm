"""Guards for checkpoint loading.

The bug these cover cost a full head search: `diffusionfamily/diffugpt-s` is
serialized under DiffuLLaMA's wrapper parameter names, so loading it as a bare
GPT2LMHeadModel matched zero tensors and randomly initialized the network while
only printing a warning. Every accuracy came out at chance and nothing raised.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dlmrel.model import _wrapper_key_map


class TestWrapperKeyMap:
    def test_gpt2_body_prefix_is_rewritten(self):
        assert (
            _wrapper_key_map("denoise_model.h.0.attn.c_attn.weight", "diffugpt")
            == "transformer.h.0.attn.c_attn.weight"
        )

    def test_llama_body_prefix_is_rewritten(self):
        assert (
            _wrapper_key_map("denoise_model.layers.0.self_attn.q_proj.weight", "diffullama")
            == "model.layers.0.self_attn.q_proj.weight"
        )

    def test_hoisted_embedding_goes_back_into_the_body(self):
        # The wrapper deletes the body's own embedding and keeps its own copy,
        # and the two families spell that parameter differently.
        assert _wrapper_key_map("embed_tokens.weight", "diffugpt") == (
            "transformer.wte.weight"
        )
        assert _wrapper_key_map("embed_tokens.weight", "diffullama") == (
            "model.embed_tokens.weight"
        )

    def test_lm_head_is_already_top_level(self):
        assert _wrapper_key_map("lm_head.weight", "diffullama") == "lm_head.weight"

    def test_unknown_keys_are_reported_rather_than_guessed(self):
        assert _wrapper_key_map("optimizer.state", "diffugpt") is None

    def test_positional_embedding_survives_the_rewrite(self):
        # wpe lives inside the body, unlike wte, so it must not be special-cased.
        assert (
            _wrapper_key_map("denoise_model.wpe.weight", "diffugpt")
            == "transformer.wpe.weight"
        )

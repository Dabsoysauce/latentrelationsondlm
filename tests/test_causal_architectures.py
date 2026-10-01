"""Opt-in pinned-source CPU tests: random tiny weights, never pretrained evidence.

Run with DLMREL_TEST_PINNED_DREAM=1 after installing requirements/dream.txt.
Only configuration and model Python source are fetched; no checkpoint weights.
"""

import os

import pytest
import torch

from dlmrel.models.dream import DreamAdapter
from dlmrel.models.interventions import HeadScale, scale_projection_heads


@pytest.mark.skipif(os.environ.get("DLMREL_TEST_PINNED_DREAM") != "1", reason="opt-in pinned remote source")
def test_pinned_dream_projection_identity_and_behavior():
    from transformers import AutoConfig, AutoModel

    torch.set_num_threads(1)
    revision = "6572adb5535263e4d1a337b56942ba48b6dee2a9"
    config = AutoConfig.from_pretrained(
        "Dream-org/Dream-v0-Base-7B", revision=revision, trust_remote_code=True
    )
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size = 32
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.max_position_embeddings = 32
    config._attn_implementation = "eager"
    model = AutoModel.from_config(config, trust_remote_code=True, code_revision=revision).eval()
    adapter = DreamAdapter(model, None, "cpu").eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    baseline = adapter.forward_logits(ids)
    with scale_projection_heads(adapter, [[HeadScale(0, 2, 1)]], torch.tensor([True])):
        identity = adapter.forward_logits(ids)
    torch.testing.assert_close(baseline, identity, rtol=0, atol=0)
    with scale_projection_heads(adapter, [[HeadScale(0, 2, 0)]], torch.tensor([True])):
        ablated = adapter.forward_logits(ids)
    assert not torch.equal(baseline, ablated)
    torch.testing.assert_close(adapter.forward_logits(ids), baseline, rtol=0, atol=0)
    from types import SimpleNamespace

    from dlmrel.experiments.causal_run import assert_equivalent
    from dlmrel.experiments.causal_trajectory import CausalSettings, reconstruct

    tokenizer = SimpleNamespace(
        bos_token_id=1, mask_token_id=0, encode=lambda *a, **kw: [], decode=lambda ids: str(ids)
    )
    instance = SimpleNamespace(attender_span=[2], receiver_span=[3])
    reference = reconstruct(adapter, tokenizer, ids, instance, 42, [None], CausalSettings())[0]
    identities = reconstruct(
        adapter, tokenizer, ids, instance, 42, [None, (((0, 2),), 1.0, "entire")], CausalSettings()
    )
    for identity in identities:
        assert_equivalent(reference, identity)

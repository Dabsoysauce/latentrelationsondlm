"""Pinned GPT-2 static/final-state comparison adapter."""

from __future__ import annotations

import torch

from .base import Capabilities, ModelAdapter


class GPT2Adapter(ModelAdapter, torch.nn.Module):
    mask_free = True
    capabilities = Capabilities(logits=True, hidden_states=True, attentions=True)

    def __init__(self, backbone, tokenizer, device: str):
        torch.nn.Module.__init__(self)
        ModelAdapter.__init__(self, backbone, tokenizer, device)

    def get_embeds(self, input_ids):
        return self.backbone.get_input_embeddings()(input_ids)

    def get_logits(self, hidden_state):
        return self.backbone.lm_head(hidden_state)

    def get_final_norm(self):
        return self.backbone.transformer.ln_f

    def get_lm_head(self):
        return self.backbone.lm_head

    @torch.no_grad()
    def forward_attentions(self, input_ids, output_hidden_states: bool = False):
        output = self.backbone(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )
        if output_hidden_states:
            return output.logits, output.attentions, output.hidden_states
        return output.logits, output.attentions


def load(model_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = model_cfg["name"]
    revision = model_cfg["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=model_cfg["tokenizer_revision"])
    backbone = AutoModelForCausalLM.from_pretrained(name, revision=revision).eval()
    device = model_cfg.get("device", "cpu")
    adapter = GPT2Adapter(backbone.to(device), tokenizer, device).eval()
    cfg = backbone.config
    return (
        adapter,
        tokenizer,
        {
            "checkpoint": name,
            "revision": revision,
            "n_layers": cfg.n_layer,
            "n_heads": cfg.n_head,
            "hidden_size": cfg.n_embd,
            "capabilities": adapter.capabilities.__dict__,
        },
    )

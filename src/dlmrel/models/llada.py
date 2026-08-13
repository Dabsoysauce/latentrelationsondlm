"""LLaDA logits/hidden-state adapter; attention stays locked pending parity."""

from __future__ import annotations

import torch

from .base import Capabilities, ModelAdapter


class LLaDAAdapter(ModelAdapter, torch.nn.Module):
    capabilities = Capabilities(
        logits=True,
        hidden_states=True,
        attentions=False,
        native_timestep=True,
    )

    def __init__(self, backbone, tokenizer, device: str):
        torch.nn.Module.__init__(self)
        ModelAdapter.__init__(self, backbone, tokenizer, device)

    def get_embeds(self, input_ids):
        return self.backbone.get_input_embeddings()(input_ids)

    def get_logits(self, hidden_state):
        return self.get_lm_head()(hidden_state)

    def get_final_norm(self):
        for path in (("model", "norm"), ("transformer", "ln_f")):
            value = self.backbone
            for part in path:
                value = getattr(value, part, None)
                if value is None:
                    break
            if value is not None:
                return value
        raise NotImplementedError("LLaDA final norm was not located")

    def get_lm_head(self):
        head = self.backbone.get_output_embeddings()
        if head is None:
            raise NotImplementedError("LLaDA output embedding is unavailable")
        return head

    def forward_attentions(self, input_ids, output_hidden_states: bool = False):
        raise NotImplementedError(
            "LLaDA attention is disabled until official/instrumented numerical parity passes"
        )

    @torch.no_grad()
    def forward_hidden_states(self, input_ids):
        output = self.backbone(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states
        logits = getattr(output, "logits", None)
        if logits is None:
            logits = self.get_logits(hidden[-1])
        return logits, hidden


def load(model_cfg: dict):
    from transformers import AutoModel, AutoTokenizer

    name, revision = model_cfg["name"], model_cfg["revision"]
    code_revision = model_cfg.get("remote_code_revision") or revision
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=model_cfg["tokenizer_revision"],
        trust_remote_code=True,
        code_revision=code_revision,
    )
    backbone = AutoModel.from_pretrained(
        name,
        revision=revision,
        code_revision=code_revision,
        trust_remote_code=True,
        torch_dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        device_map="auto",
    ).eval()
    adapter = LLaDAAdapter(backbone, tokenizer, model_cfg.get("device", "cuda")).eval()
    config = backbone.config
    return (
        adapter,
        tokenizer,
        {
            "checkpoint": name,
            "revision": revision,
            "remote_code_revision": code_revision,
            "n_layers": getattr(config, "num_hidden_layers", None),
            "n_heads": getattr(config, "num_attention_heads", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "capabilities": adapter.capabilities.__dict__,
            "attention_parity": "not_run",
        },
    )

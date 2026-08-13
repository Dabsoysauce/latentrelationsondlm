from __future__ import annotations

import torch

from .base import Capabilities, ModelAdapter

CHECKPOINT = "Dream-org/Dream-v0-Base-7B"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class DreamAdapter(ModelAdapter, torch.nn.Module):
    mask_free = True
    capabilities = Capabilities(
        logits=True, hidden_states=True, attentions=True, native_generation=True
    )

    def __init__(self, backbone, tokenizer, device: str):
        torch.nn.Module.__init__(self)
        ModelAdapter.__init__(self, backbone, tokenizer, device)

    def get_embeds(self, input_ids):
        return self.backbone.get_input_embeddings()(input_ids)

    def get_logits(self, hidden_state):
        return self.backbone.lm_head(hidden_state)

    def get_final_norm(self):
        return self.backbone.model.norm

    def get_lm_head(self):
        return self.backbone.lm_head

    @torch.no_grad()
    def forward_attentions(self, input_ids, output_hidden_states: bool = False):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=None,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if getattr(out, "attentions", None) is None:
            raise RuntimeError(
                "Dream returned no attention weights; load with attn_implementation='eager'."
            )
        if output_hidden_states:
            return None, out.attentions, out.hidden_states
        return None, out.attentions


def load(model_cfg: dict):
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    checkpoint = model_cfg.get("checkpoint", CHECKPOINT)
    dtype = _DTYPES[model_cfg.get("dtype", "bfloat16")]
    device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    attn = model_cfg.get("attn_implementation", "eager")

    if "default" not in ROPE_INIT_FUNCTIONS and "rope" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["rope"]

    revision = model_cfg["revision"]
    remote_revision = model_cfg.get("remote_code_revision") or revision
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=model_cfg["tokenizer_revision"],
        trust_remote_code=True,
        code_revision=remote_revision,
    )

    try:
        backbone = AutoModel.from_pretrained(
            checkpoint,
            revision=revision,
            code_revision=remote_revision,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation=attn,
        ).eval()
    except (TypeError, ValueError):
        backbone = AutoModel.from_pretrained(
            checkpoint,
            revision=revision,
            code_revision=remote_revision,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

    if tokenizer.mask_token_id is None:
        raise ValueError(f"{checkpoint} exposes no mask token")

    adapter = DreamAdapter(backbone, tokenizer, device).eval()

    hf = backbone.config
    meta = {
        "checkpoint": checkpoint,
        "revision": revision,
        "remote_code_revision": remote_revision,
        "capabilities": adapter.capabilities.__dict__,
        "n_layers": hf.num_hidden_layers,
        "n_heads": hf.num_attention_heads,
        "hidden_size": hf.hidden_size,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }

    probe = torch.tensor(
        [
            [tokenizer.bos_token_id]
            + tokenizer.encode("The cat sat on the mat.", add_special_tokens=False)
        ],
        device=adapter.device,
    )
    _, attentions = adapter.forward_attentions(probe)
    if not attentions or attentions[0] is None:
        raise RuntimeError("Dream returned no attention weights at load time")
    return adapter, tokenizer, meta

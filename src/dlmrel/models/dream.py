"""Dream-7B adapter.

Dream-org/Dream-v0-Base-7B is Qwen2-based and natively bidirectional, so it
takes the raw ids with attention_mask=None and needs no third-party patch. It
requires transformers>=4.51 (mutually exclusive with the DiffuLLaMA families'
4.44.2 pin) and ships its own modelling code, so trust_remote_code is inherent.

Established by the loading smoke test: the tokenizer prepends no BOS but a BOS
token exists, so include_bos gives position 0 a dedicated sink slot; ~52% of
attention lands there, so excluding it is a correction, not a distortion.
"""

from __future__ import annotations

import torch

from .base import ModelAdapter

CHECKPOINT = "Dream-org/Dream-v0-Base-7B"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class DreamAdapter(ModelAdapter, torch.nn.Module):
    mask_free = True

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
                "Dream returned no attention weights; the remote code is "
                "ignoring output_attentions and every accuracy would be zero. "
                "Load with attn_implementation='eager'."
            )
        if output_hidden_states:
            return None, out.attentions, out.hidden_states
        return None, out.attentions


def load(model_cfg: dict):
    """Load Dream from a model config (configs/models/dream_7b.yaml).

    Reads `checkpoint`; dtype/device/attn are optional and default to the
    known-good Dream settings. Returns (adapter, tokenizer, meta).
    """
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    checkpoint = model_cfg.get("checkpoint", CHECKPOINT)
    dtype = _DTYPES[model_cfg.get("dtype", "bfloat16")]
    device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    attn = model_cfg.get("attn_implementation", "eager")

    # Dream's remote code asks for rope_type="default", which some transformers
    # builds expose as "rope" instead.
    if "default" not in ROPE_INIT_FUNCTIONS and "rope" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["rope"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    try:
        backbone = AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation=attn,
        ).eval()
    except (TypeError, ValueError):
        backbone = AutoModel.from_pretrained(
            checkpoint, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
        ).eval()

    if tokenizer.mask_token_id is None:
        raise ValueError(f"{checkpoint} exposes no mask token; the schedule needs one")

    adapter = DreamAdapter(backbone, tokenizer, device).eval()

    hf = backbone.config
    meta = {
        "checkpoint": checkpoint,
        "n_layers": hf.num_hidden_layers,
        "n_heads": hf.num_attention_heads,
        "hidden_size": hf.hidden_size,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }

    # Fail loudly at load time rather than silently scoring zeros.
    probe = torch.tensor(
        [[tokenizer.bos_token_id] + tokenizer.encode("The cat sat on the mat.", add_special_tokens=False)],
        device=adapter.device,
    )
    _, attentions = adapter.forward_attentions(probe)
    if not attentions or attentions[0] is None:
        raise RuntimeError("Dream returned no attention weights at load time")
    return adapter, tokenizer, meta

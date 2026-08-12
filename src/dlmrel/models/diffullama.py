from __future__ import annotations

from ._backbone import WrappedAdapter, load_wrapped


class DiffuLlamaAdapter(WrappedAdapter):
    final_norm_attr = "norm"


def load(model_cfg: dict):
    return load_wrapped("diffullama", "LlamaForCausalLM", model_cfg, DiffuLlamaAdapter)

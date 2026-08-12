from __future__ import annotations

from ._backbone import WrappedAdapter, load_wrapped


class DiffuGPTAdapter(WrappedAdapter):
    final_norm_attr = "ln_f"


def load(model_cfg: dict):
    return load_wrapped("diffugpt", "GPT2LMHeadModel", model_cfg, DiffuGPTAdapter)

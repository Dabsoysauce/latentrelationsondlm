from __future__ import annotations

from ._backbone import WrappedAdapter, load_diffugpt


class DiffuGptAdapter(WrappedAdapter):
    final_norm_attr = "ln_f"


def load(model_cfg: dict):
    return load_diffugpt(model_cfg, DiffuGptAdapter)

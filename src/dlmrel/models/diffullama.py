from __future__ import annotations

from ._backbone import WrappedAdapter, load_diffullama


class DiffuLlamaAdapter(WrappedAdapter):
    final_norm_attr = "norm"


def load(model_cfg: dict):
    return load_diffullama(model_cfg, DiffuLlamaAdapter)

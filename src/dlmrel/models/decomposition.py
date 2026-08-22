"""Exact attention-output projection capture and single-head intervention."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch

from ..paper_protocol import zero_projection_head_input


@dataclass(frozen=True)
class ProjectionCapture:
    values: torch.Tensor
    weight: torch.Tensor
    number_of_heads: int
    module_path: str


def _layers(adapter):
    candidates = (
        (getattr(getattr(adapter, "backbone", None), "model", None), "backbone.model"),
        (getattr(adapter, "denoise_model", None), "denoise_model"),
        (getattr(getattr(adapter, "denoise_model", None), "model", None), "denoise_model.model"),
    )
    for owner, path in candidates:
        layers = getattr(owner, "layers", None)
        if layers is not None:
            return layers, path
    raise RuntimeError("adapter architecture exposes no validated transformer layer stack")


def projection_module(adapter, layer: int):
    layers, prefix = _layers(adapter)
    if not 0 <= layer < len(layers):
        raise ValueError("requested layer is outside the transformer")
    attention = getattr(layers[layer], "self_attn", None)
    projection = getattr(attention, "o_proj", None)
    heads = getattr(attention, "num_heads", None)
    if projection is None or heads is None or not hasattr(projection, "weight"):
        raise RuntimeError("adapter lacks a validated Llama-style per-head output projection")
    return projection, int(heads), f"{prefix}.layers.{layer}.self_attn.o_proj"


@contextmanager
def capture_or_ablate_projection(adapter, layer: int, *, ablate_head: int | None = None):
    """Capture the exact concatenated value-weighted heads entering ``o_proj``."""
    projection, number_of_heads, path = projection_module(adapter, layer)
    captured: list[torch.Tensor] = []

    def hook(_module, args):
        values = args[0]
        captured.append(values.detach().clone())
        if ablate_head is None:
            return None
        changed = zero_projection_head_input(
            values, head=ablate_head, number_of_heads=number_of_heads
        )
        return (changed, *args[1:])

    handle = projection.register_forward_pre_hook(hook)
    try:
        yield captured, ProjectionCapture(
            values=torch.empty(0),
            weight=projection.weight,
            number_of_heads=number_of_heads,
            module_path=path,
        )
    finally:
        handle.remove()


@contextmanager
def capture_projection_inputs(adapter, layers: list[int]):
    """Capture several layer projection inputs during one model forward pass."""
    captures: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    metadata: dict[int, ProjectionCapture] = {}
    with ExitStack() as stack:
        for layer in layers:
            projection, number_of_heads, path = projection_module(adapter, layer)

            def hook(_module, args, current_layer=layer):
                captures[current_layer].append(args[0].detach().clone())

            stack.callback(projection.register_forward_pre_hook(hook).remove)
            metadata[layer] = ProjectionCapture(
                values=torch.empty(0),
                weight=projection.weight,
                number_of_heads=number_of_heads,
                module_path=path,
            )
        yield captures, metadata

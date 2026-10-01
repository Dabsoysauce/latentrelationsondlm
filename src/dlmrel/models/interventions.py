"""Scoped scaling/patching of exact pre-o_proj head outputs (zero-based indices)."""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch

from .decomposition import projection_module


@dataclass(frozen=True)
class HeadScale:
    layer: int
    head: int
    alpha: float = 0.0


def temporal_gate(
    mask, dependent, governor, *, step, steps, window="pre", start=0, stop=None, split=0.5, first_reveal=None
):
    """Gate each batch row using scheduler state, never predicted token IDs.

    ``first_reveal`` is the precomputed first endpoint reveal step per row.
    A token revealed at step k is still masked at the forward at step k.
    """
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a [batch, tokens] boolean tensor")
    if not 0 <= step < steps or not 0 <= start < (steps if stop is None else stop) <= steps:
        raise ValueError("invalid half-open step range")
    if not 0 < split < 1:
        raise ValueError("split must lie strictly between zero and one")
    if len(dependent) != len(mask) or len(governor) != len(mask):
        raise ValueError("one pair of spans is required per batch row")
    active = []
    for row, (dep, gov) in enumerate(zip(dependent, governor, strict=True)):
        positions = list(set(dep) | set(gov))
        if not dep or not gov or min(positions) < 0 or max(positions) >= mask.shape[1]:
            raise ValueError("dependency spans are empty or out of bounds")
        selected = mask[row, positions]
        pre, post = bool(selected.all()), bool((~selected).all())
        if window == "entire":
            value = True
        elif window == "pre":
            value = pre
        elif window == "post":
            value = post
        elif window in {"early_pre", "late_pre"}:
            if first_reveal is None:
                raise ValueError("early/late windows require exogenous first_reveal")
            boundary = math.ceil((int(first_reveal[row]) + 1) * split)
            value = pre and (step < boundary if window == "early_pre" else step >= boundary)
        else:
            raise ValueError(f"unknown window: {window}")
        active.append(value and start <= step < (steps if stop is None else stop))
    return torch.tensor(active, device=mask.device, dtype=torch.bool)


@contextmanager
def scale_projection_heads(adapter, requests, active, *, capture=None, patches=None):
    """Scale any number of heads per batch row; restore hooks even on failure.

    ``requests`` is a list of lists of HeadScale, one list per batch row.
    Optional capture/patch dictionaries use (row, layer, head) keys and [T,Dh]
    tensors. Callers own cross-run time/position alignment; shape mismatches fail.
    Captures are detached CPU copies; no persistent model weights are edited.
    """
    if active.ndim != 1 or active.dtype != torch.bool or len(active) != len(requests):
        raise ValueError("active must be a boolean gate per batch row")
    layers = {}
    for items in requests:
        seen = set()
        for item in items:
            key = (item.layer, item.head)
            if key in seen or not math.isfinite(item.alpha):
                raise ValueError("duplicate head or nonfinite alpha")
            seen.add(key)
            module, heads, path = projection_module(adapter, item.layer)
            if not 0 <= item.head < heads or module.weight.shape[1] % heads:
                raise ValueError("head index or projection width is invalid")
            layers[item.layer] = (module, heads, path)
    calls = dict.fromkeys(layers, 0)
    with ExitStack() as stack:
        for layer, (module, heads, _path) in layers.items():

            def hook(_module, args, layer=layer, heads=heads):
                values = args[0]
                if values.ndim != 3 or len(values) != len(requests):
                    raise RuntimeError("expected [batch, tokens, concatenated heads]")
                if values.shape[-1] != _module.weight.shape[1]:
                    raise RuntimeError("projection input width does not match weight")
                calls[layer] += 1
                width = values.shape[-1] // heads
                changed = None
                for row, items in enumerate(requests):
                    for item in items:
                        if item.layer != layer:
                            continue
                        key = (row, layer, item.head)
                        part = slice(item.head * width, (item.head + 1) * width)
                        if capture is not None:
                            capture[key] = values[row, :, part].detach().cpu().clone()
                        if not bool(active[row]):
                            continue
                        patch = None if patches is None else patches.get(key)
                        if patch is None and item.alpha == 1:
                            continue  # exact identity, including the tensor object
                        if changed is None:
                            changed = values.clone()
                        if patch is not None:
                            if patch.shape != values[row, :, part].shape:
                                raise ValueError("patch has incompatible position/head shape")
                            changed[row, :, part] = patch.to(values)
                        else:
                            changed[row, :, part] = values[row, :, part] * item.alpha
                return None if changed is None else (changed, *args[1:])

            stack.callback(module.register_forward_pre_hook(hook).remove)
        yield calls
        if any(count != 1 for count in calls.values()):
            raise RuntimeError("each requested o_proj must execute exactly once per scoped forward")

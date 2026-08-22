"""Native generated trajectories under the preserved DiffuGPT sampler."""

from __future__ import annotations

import random
from typing import Any

import torch

from .base import NativeTrajectory


def aligned_logits(logits: torch.Tensor, input_ids: torch.Tensor, prediction_offset: int) -> torch.Tensor:
    """Align adapter logits so axis position ``p`` predicts token ``p``."""
    if prediction_offset == 0:
        return logits
    if prediction_offset == -1:
        first = torch.nn.functional.one_hot(
            input_ids[:, :1], num_classes=logits.shape[-1]
        ).to(logits.dtype)
        return torch.cat([first, logits[:, :-1]], dim=1)
    raise ValueError(f"unsupported prediction offset: {prediction_offset}")


def _top_p_sample(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must lie in (0, 1]")
    scores = logits.float() / temperature
    sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
    cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative - sorted_scores.softmax(dim=-1) >= top_p
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    sampled = torch.distributions.Categorical(logits=sorted_scores).sample().unsqueeze(-1)
    return sorted_indices.gather(-1, sampled).squeeze(-1)


def _forward_logits(adapter, input_ids: torch.Tensor) -> torch.Tensor:
    logits, _attentions = adapter.forward_attentions(input_ids)
    if logits is None:
        raise RuntimeError("native generation requires adapter logits")
    return logits


@torch.inference_mode()
def random_reveal_trajectory(
    adapter,
    tokenizer,
    prompt: str,
    *,
    seed: int,
    steps: int = 64,
    generation_length: int = 96,
    temperature: float = 0.95,
    top_p: float = 0.9,
    **_unused: Any,
) -> NativeTrajectory:
    """Run the executable old sampler and retain all pre-forward states.

    The RNG is reset once for the whole prompt trajectory.  At each reverse
    step, every still-masked position is independently revealed with
    probability ``1 / remaining_steps``.  The representation alignment is an
    adapter property, not a global DiffuGPT assumption.
    """
    if steps != 64:
        raise ValueError("paper native trajectories require exactly 64 steps")
    if tokenizer is None or tokenizer.mask_token_id is None:
        raise RuntimeError("native generation requires a tokenizer mask token")
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prefix = [tokenizer.bos_token_id, *tokenizer.encode(prompt, add_special_tokens=False)]
    prefix = prefix[: generation_length - 1]
    prefix_length = len(prefix)
    base = torch.tensor(
        [prefix + [0] * (generation_length - prefix_length)],
        dtype=torch.long,
        device=adapter.device,
    )
    maskable = torch.zeros_like(base, dtype=torch.bool)
    maskable[:, prefix_length:] = True
    xt = base.masked_fill(maskable, int(tokenizer.mask_token_id))

    states: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    current_mask = maskable.clone()
    final_sample = xt.clone()
    for step_index in range(steps):
        states.append(xt[0].detach().cpu().clone())
        raw_logits = _forward_logits(adapter, xt)
        logits = aligned_logits(raw_logits, xt, int(adapter.prediction_offset))
        predictions.append(logits.argmax(dim=-1)[0].detach().cpu().clone())
        final_sample = _top_p_sample(logits, temperature=temperature, top_p=top_p)
        final_sample = xt.masked_scatter(current_mask, final_sample[current_mask])
        remaining_steps = steps - step_index
        reveal = current_mask & (
            torch.rand_like(current_mask, dtype=torch.float) < (1.0 / remaining_steps)
        )
        if remaining_steps == 1:
            reveal = current_mask
        xt = xt.clone()
        xt[reveal] = final_sample[reveal]
        current_mask &= ~reveal

    return NativeTrajectory(
        prompt=prompt,
        prefix_length=prefix_length,
        pre_forward_ids=tuple(states),
        argmax_ids=tuple(predictions),
        final_ids=final_sample[0].detach().cpu().clone(),
        metadata={
            "steps": steps,
            "generation_length": generation_length,
            "temperature": temperature,
            "top_p": top_p,
            "reveal_policy": "random_one_over_remaining_steps",
            "seed": seed,
            "prediction_offset": int(adapter.prediction_offset),
            "pre_forward_states": True,
        },
    )

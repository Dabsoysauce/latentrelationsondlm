"""Native generated trajectories under the preserved DiffuGPT sampler."""

from __future__ import annotations

import random
from contextlib import nullcontext
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
    if hasattr(adapter, "forward_logits"):
        logits = adapter.forward_logits(input_ids)
    else:
        logits, _attentions = adapter.forward_attentions(input_ids)
    if logits is None:
        raise RuntimeError("native generation requires adapter logits")
    return logits


def _capture_rng_state() -> tuple[torch.Tensor, list[torch.Tensor], object]:
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), cuda, random.getstate()


def _restore_rng_state(state: tuple[torch.Tensor, list[torch.Tensor], object]) -> None:
    cpu, cuda, python = state
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)
    random.setstate(python)


def _seeded_rng_state(seed: int) -> tuple[torch.Tensor, list[torch.Tensor], object]:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return _capture_rng_state()


@torch.inference_mode()
def random_reveal_trajectories(
    adapter,
    tokenizer,
    prompts: list[str],
    *,
    seed: int,
    steps: int = 64,
    generation_length: int = 96,
    temperature: float = 0.95,
    top_p: float = 0.9,
    reconstruction_ids: torch.Tensor | None = None,
    reconstruction_mask: torch.Tensor | None = None,
    reveal_plan: torch.Tensor | None = None,
    forward_context=None,
    observer=None,
    **_unused: Any,
) -> tuple[NativeTrajectory, ...]:
    """Batch independent prompts while replaying each trajectory's exact RNG stream.

    The legacy sampler resets the global CPU, CUDA, and Python RNGs to ``seed``
    for every prompt. This implementation gives each batch row its own snapshot
    of precisely that seeded global state, restores it for the row's sampling
    and reveal draws, and saves the advanced state for the next reverse step.
    Consequently each row consumes the same random numbers in the same order as
    an isolated ``random_reveal_trajectory`` call; only the deterministic model
    forward is shared.
    """
    if not prompts:
        return ()
    if steps != 64:
        raise ValueError("paper native trajectories require exactly 64 steps")
    if tokenizer is None or tokenizer.mask_token_id is None:
        raise RuntimeError("native generation requires a tokenizer mask token")

    prefixes = []
    for prompt in prompts:
        prefix = [tokenizer.bos_token_id, *tokenizer.encode(prompt, add_special_tokens=False)]
        prefixes.append(prefix[: generation_length - 1])
    prefix_lengths = [len(prefix) for prefix in prefixes]
    base = torch.tensor(
        [prefix + [0] * (generation_length - len(prefix)) for prefix in prefixes],
        dtype=torch.long,
        device=adapter.device,
    )
    maskable = torch.zeros_like(base, dtype=torch.bool)
    for batch_index, prefix_length in enumerate(prefix_lengths):
        maskable[batch_index, prefix_length:] = True
    if reconstruction_ids is not None:
        if reconstruction_mask is None or reconstruction_mask.shape != reconstruction_ids.shape:
            raise ValueError("reconstruction requires equally shaped ids and mask")
        if reconstruction_ids.ndim != 2 or len(reconstruction_ids) != len(prompts):
            raise ValueError("reconstruction batch must match prompts")
        if reconstruction_mask.dtype != torch.bool or bool(reconstruction_mask[:, 0].any()):
            raise ValueError("reconstruction mask must be boolean with BOS visible")
        base = reconstruction_ids.to(adapter.device).clone()
        maskable = reconstruction_mask.to(adapter.device).clone()
        generation_length = base.shape[1]
        prefix_lengths = [1] * len(prompts)
    if reveal_plan is not None:
        if reveal_plan.shape != (steps, *base.shape) or reveal_plan.dtype != torch.bool:
            raise ValueError("reveal plan must be boolean [steps, batch, tokens]")
        if not torch.equal(reveal_plan.sum(dim=0).cpu(), maskable.long().cpu()):
            raise ValueError("reveal plan must reveal each masked position exactly once")
        reveal_plan = reveal_plan.to(adapter.device)
    xt = base.masked_fill(maskable, int(tokenizer.mask_token_id))
    current_mask = maskable.clone()

    initial_rng = _seeded_rng_state(seed)
    rng_states = [
        (initial_rng[0].clone(), [item.clone() for item in initial_rng[1]], initial_rng[2])
        for _prompt in prompts
    ]
    states: list[list[torch.Tensor]] = [[] for _prompt in prompts]
    predictions: list[list[torch.Tensor]] = [[] for _prompt in prompts]
    final_samples = xt.clone()

    for step_index in range(steps):
        for batch_index in range(len(prompts)):
            states[batch_index].append(xt[batch_index].detach().cpu().clone())
        context = nullcontext() if forward_context is None else forward_context(step_index, current_mask)
        with context:
            raw_logits = _forward_logits(adapter, xt)
        logits = aligned_logits(raw_logits, xt, int(adapter.prediction_offset))
        remaining_steps = steps - step_index
        for batch_index in range(len(prompts)):
            predictions[batch_index].append(
                logits[batch_index].argmax(dim=-1).detach().cpu().clone()
            )
            _restore_rng_state(rng_states[batch_index])
            row_logits = logits[batch_index : batch_index + 1]
            sampled = _top_p_sample(row_logits, temperature=temperature, top_p=top_p)
            row_mask = current_mask[batch_index : batch_index + 1]
            final = xt[batch_index : batch_index + 1].masked_scatter(
                row_mask, sampled[row_mask]
            )
            reveal = row_mask & (
                torch.rand_like(row_mask, dtype=torch.float) < (1.0 / remaining_steps)
            )
            if remaining_steps == 1:
                reveal = row_mask
            if reveal_plan is not None:
                reveal = reveal_plan[step_index, batch_index : batch_index + 1]
            if observer is not None:
                observer(step_index, batch_index, xt[batch_index], row_mask[0],
                         logits[batch_index], reveal[0])
            xt[batch_index : batch_index + 1] = xt[
                batch_index : batch_index + 1
            ].masked_scatter(reveal, final[reveal])
            current_mask[batch_index : batch_index + 1] &= ~reveal
            final_samples[batch_index : batch_index + 1] = final
            rng_states[batch_index] = _capture_rng_state()

    return tuple(
        NativeTrajectory(
            prompt=prompt,
            prefix_length=prefix_lengths[index],
            pre_forward_ids=tuple(states[index]),
            argmax_ids=tuple(predictions[index]),
            final_ids=final_samples[index].detach().cpu().clone(),
            metadata={
                "steps": steps,
                "generation_length": generation_length,
                "temperature": temperature,
                "top_p": top_p,
                "reveal_policy": "random_one_over_remaining_steps",
                "seed": seed,
                "prediction_offset": int(adapter.prediction_offset),
                "pre_forward_states": True,
                "batched_forward": True,
                "rng_equivalence": "per_trajectory_global_state_replay",
                **({"task": "reconstruction", "external_reveal_plan": reveal_plan is not None}
                   if reconstruction_ids is not None else {}),
            },
        )
        for index, prompt in enumerate(prompts)
    )


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

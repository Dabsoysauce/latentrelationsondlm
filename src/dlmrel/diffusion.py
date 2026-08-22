"""Reconstructing intermediate denoising states and reading attention from them.

The masking schedule is teacher-forced: rather than letting the model generate,
the true sentence is progressively revealed with per-step probability
`1 / (steps - progress)`. That reproduces the marginal masking rate the model
saw in training while keeping the gold parse valid for the visible tokens, which
is what makes the masked-state measurement well defined at all.

Reproduced from the original notebooks; the schedule is unchanged so numbers
stay comparable with the previous runs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DenoisingState:
    input_ids: torch.Tensor  # [1, seq_len] token ids at this timestep
    tokens: list[str]  # readable token strings
    is_visible: list[bool]  # True where the token has been revealed
    unmask_step: list[int]  # step at which each position was revealed, -1 if never

    @property
    def n_masked(self) -> int:
        return sum(1 for v in self.is_visible if not v)


@torch.no_grad()
def forward_with_attentions(model, input_ids, attention_mask):
    """One denoising forward pass returning logits and per-layer attentions.

    `attentions[layer]` has shape [batch, heads, seq_len, seq_len].
    """
    if getattr(model, "mask_free", False):
        return model.forward_attentions(input_ids)

    embeds = model.get_embeds(input_ids)
    outputs = model.denoise_model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )
    return model.get_logits(outputs.last_hidden_state), outputs.attentions


def tokenize(tokenizer, text: str, device, add_bos: bool = True):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if add_bos:
        ids = [tokenizer.bos_token_id] + ids
    tensor = torch.tensor([ids], device=device)
    return tensor, [tokenizer.decode([i]) for i in ids]


@torch.no_grad()
def state_at_time(
    model,
    tokenizer,
    text: str,
    diffusion_time: int,
    steps: int = 64,
    seed: int = 42,
    include_bos: bool = True,
    force_final_unmasked: bool = True,
) -> DenoisingState:
    """Rebuild x_t at a chosen diffusion timestep.

    `diffusion_time=0` is the fully masked start; `steps - 1` is the final
    frame, forced fully unmasked so the head search is scored on a complete
    sentence.
    """
    if not 0 <= diffusion_time < steps:
        raise ValueError(f"diffusion_time must be in [0, {steps}), got {diffusion_time}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    true_ids, tokens = tokenize(tokenizer, text, model.device, include_bos)
    seq_len = true_ids.shape[1]

    if tokenizer.mask_token_id is None:
        if not getattr(model, "mask_free", False) or diffusion_time != steps - 1:
            raise ValueError("a model without a mask token supports only its final static state")
        return DenoisingState(
            input_ids=true_ids,
            tokens=tokens,
            is_visible=[True] * seq_len,
            unmask_step=[0] * seq_len,
        )

    maskable = torch.ones_like(true_ids, dtype=torch.bool)
    if include_bos:
        maskable[:, 0] = False

    xt = true_ids.masked_fill(maskable, tokenizer.mask_token_id)
    remaining = maskable.clone()

    unmask_step = [-1] * seq_len
    if include_bos:
        unmask_step[0] = 0

    for progress in range(diffusion_time):
        p_reveal = 1.0 / (steps - progress)
        draw = torch.rand_like(remaining, dtype=torch.float, device=model.device)
        reveal = remaining & (draw < p_reveal)

        xt = xt.clone()
        xt[reveal] = true_ids[reveal]
        for pos in reveal[0].nonzero(as_tuple=True)[0].tolist():
            if unmask_step[pos] == -1:
                unmask_step[pos] = progress + 1
        remaining &= ~reveal

    if force_final_unmasked and diffusion_time == steps - 1:
        xt = true_ids.clone()
        remaining = torch.zeros_like(remaining, dtype=torch.bool)
        unmask_step = [diffusion_time if s == -1 else s for s in unmask_step]

    return DenoisingState(
        input_ids=xt,
        tokens=tokens,
        is_visible=(~remaining[0]).cpu().tolist(),
        unmask_step=unmask_step,
    )


@torch.no_grad()
def teacher_forced_trajectory(
    model,
    tokenizer,
    text: str,
    *,
    steps: int = 64,
    seed: int = 42,
    include_bos: bool = True,
) -> tuple[DenoisingState, ...]:
    """Build one nested old-schedule trajectory with a single RNG reset."""
    if steps != 64:
        raise ValueError("the paper protocol requires exactly 64 steps")
    torch.manual_seed(seed)
    np.random.seed(seed)
    true_ids, tokens = tokenize(tokenizer, text, model.device, include_bos)
    sequence_length = true_ids.shape[1]
    maskable = torch.ones_like(true_ids, dtype=torch.bool)
    if include_bos:
        maskable[:, 0] = False
    xt = true_ids.masked_fill(maskable, tokenizer.mask_token_id)
    remaining = maskable.clone()
    unmask_step = [-1] * sequence_length
    if include_bos:
        unmask_step[0] = 0
    states = []
    for timestep in range(steps):
        if timestep == steps - 1:
            xt = true_ids.clone()
            remaining = torch.zeros_like(remaining)
            unmask_step = [timestep if value == -1 else value for value in unmask_step]
        states.append(
            DenoisingState(
                input_ids=xt.clone(),
                tokens=list(tokens),
                is_visible=(~remaining[0]).cpu().tolist(),
                unmask_step=list(unmask_step),
            )
        )
        if timestep >= steps - 1:
            continue
        reveal_probability = 1.0 / (steps - timestep)
        draw = torch.rand_like(remaining, dtype=torch.float, device=model.device)
        reveal = remaining & (draw < reveal_probability)
        xt = xt.clone()
        xt[reveal] = true_ids[reveal]
        for position in reveal[0].nonzero(as_tuple=True)[0].tolist():
            if unmask_step[position] == -1:
                unmask_step[position] = timestep + 1
        remaining &= ~reveal
    return tuple(states)


@torch.no_grad()
def attentions_for_state(model, state: DenoisingState):
    """Read attentions for an already materialized nested trajectory state."""
    if hasattr(model, "forward_attentions_only"):
        return model.forward_attentions_only(state.input_ids)
    _logits, attentions = model.forward_attentions(state.input_ids)
    return attentions


@torch.no_grad()
def attention_batches_for_states(
    model,
    states: Sequence[DenoisingState],
    *,
    batch_size: int = 8,
) -> Iterator[tuple[int, tuple[DenoisingState, ...], tuple[torch.Tensor, ...]]]:
    """Forward equal-length trajectory states in small, memory-bounded batches.

    Batching changes only execution shape: state order, token IDs, attention
    formula, and all downstream scoring remain identical to one-state passes.
    """
    if batch_size < 1:
        raise ValueError("trajectory batch_size must be positive")
    materialized = tuple(states)
    if not materialized:
        return
    shape = materialized[0].input_ids.shape
    if shape[0] != 1 or any(state.input_ids.shape != shape for state in materialized):
        raise ValueError("trajectory microbatching requires equal-length singleton states")
    for start in range(0, len(materialized), batch_size):
        current = materialized[start : start + batch_size]
        input_ids = torch.cat([state.input_ids for state in current], dim=0)
        if hasattr(model, "forward_attentions_only"):
            attentions = model.forward_attentions_only(input_ids)
        else:
            _logits, attentions = model.forward_attentions(input_ids)
        if any(layer.shape[0] != len(current) for layer in attentions):
            raise RuntimeError("adapter returned an incorrect attention batch dimension")
        yield start, current, attentions


@torch.no_grad()
def attentions_at_time(
    model,
    tokenizer,
    text: str,
    diffusion_time: int,
    steps: int = 64,
    seed: int = 42,
    include_bos: bool = True,
):
    """Return `(attentions, state)` for one sentence at one timestep."""
    state = state_at_time(model, tokenizer, text, diffusion_time, steps, seed, include_bos)
    return attentions_for_state(model, state), state


@torch.no_grad()
def states_at_time(
    model,
    tokenizer,
    text: str,
    diffusion_time: int,
    steps: int = 64,
    seed: int = 42,
    include_bos: bool = True,
):
    """Return `(attentions, hidden_states, state)` for one sentence."""
    state = state_at_time(model, tokenizer, text, diffusion_time, steps, seed, include_bos)

    if hasattr(model, "forward_hidden_states"):
        _, hidden = model.forward_hidden_states(state.input_ids)
        return (), hidden, state

    if hasattr(model, "forward_features"):
        attentions, hidden = model.forward_features(state.input_ids)
        return attentions, hidden, state

    from model import get_anneal_attn_mask

    embeds = model.get_embeds(state.input_ids)
    attn_mask = get_anneal_attn_mask(
        seq_len=state.input_ids.shape[1],
        bsz=state.input_ids.shape[0],
        dtype=embeds.dtype,
        device=state.input_ids.device,
        attn_mask_ratio=1.0,
    )
    outputs = model.denoise_model(
        inputs_embeds=embeds,
        attention_mask=attn_mask,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    return outputs.attentions, outputs.hidden_states, state


def receiver_predictions(
    attentions,
    layer: int,
    attender_span: list[int],
    attender_token: str = "last",
    exclude_bos: bool = True,
    exclude_self: bool = True,
    batch_index: int = 0,
) -> np.ndarray:
    """Argmax receiver predicted by every head in one layer.

    BOS is excluded because diffusion LMs park enormous attention mass on it
    (the "attention sink"), and the attender's own token positions are excluded
    because self-attention would otherwise win almost every row. Both
    exclusions are applied before the argmax, never after.
    """
    row_idx = attender_span[-1] if attender_token == "last" else attender_span[0]
    row = attentions[layer][batch_index, :, row_idx, :].detach().float().clone()

    if exclude_bos:
        row[:, 0] = float("-inf")
    if exclude_self:
        for col in attender_span:
            row[:, col] = float("-inf")

    return row.argmax(dim=1).cpu().numpy()


def endpoint_visibility(is_visible: list[bool], attender_span: list[int], receiver_span: list[int]) -> str:
    """Four mutually exclusive whole-word endpoint visibility states."""
    attender_visible = all(is_visible[index] for index in attender_span)
    receiver_visible = all(is_visible[index] for index in receiver_span)
    if attender_visible and receiver_visible:
        return "both_visible"
    if attender_visible:
        return "attender_visible_only"
    if receiver_visible:
        return "receiver_visible_only"
    return "both_masked"


def receiver_span_scores(
    attentions,
    layer: int,
    attender_span: list[int],
    receiver_spans: list[list[int]],
    *,
    row_aggregation: str = "mean",
    span_aggregation: str = "sum",
    excluded_positions: set[int] | None = None,
) -> np.ndarray:
    """Score candidates at word-span level so token count cannot win silently."""
    rows = attentions[layer][0, :, attender_span, :].detach().float()
    if row_aggregation == "mean":
        rows = rows.mean(dim=1)
    elif row_aggregation == "first":
        rows = rows[:, 0]
    elif row_aggregation == "last":
        rows = rows[:, -1]
    else:
        raise ValueError(f"unknown row aggregation: {row_aggregation}")
    if excluded_positions:
        rows[:, sorted(excluded_positions)] = float("-inf")
    values = []
    for span in receiver_spans:
        selected = rows[:, span]
        if span_aggregation == "sum":
            value = selected.sum(dim=-1)
        elif span_aggregation == "mean":
            value = selected.mean(dim=-1)
        elif span_aggregation == "max":
            value = selected.max(dim=-1).values
        else:
            raise ValueError(f"unknown span aggregation: {span_aggregation}")
        values.append(value)
    return torch.stack(values, dim=-1).cpu().numpy()

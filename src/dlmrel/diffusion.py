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
        raise ValueError(
            f"diffusion_time must be in [0, {steps}), got {diffusion_time}"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    true_ids, tokens = tokenize(tokenizer, text, model.device, include_bos)
    seq_len = true_ids.shape[1]

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
    state = state_at_time(
        model, tokenizer, text, diffusion_time, steps, seed, include_bos
    )

    if getattr(model, "mask_free", False):
        _, attentions = forward_with_attentions(model, state.input_ids, None)
        return attentions, state

    from model import get_anneal_attn_mask

    embeds = model.get_embeds(state.input_ids)
    attn_mask = get_anneal_attn_mask(
        seq_len=state.input_ids.shape[1],
        bsz=state.input_ids.shape[0],
        dtype=embeds.dtype,
        device=state.input_ids.device,
        attn_mask_ratio=1.0,
    )
    _, attentions = forward_with_attentions(model, state.input_ids, attn_mask)
    return attentions, state


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
    state = state_at_time(
        model, tokenizer, text, diffusion_time, steps, seed, include_bos
    )

    if getattr(model, "mask_free", False):
        _, attentions, hidden = model.forward_attentions(
            state.input_ids, output_hidden_states=True
        )
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
) -> np.ndarray:
    """Argmax receiver predicted by every head in one layer.

    BOS is excluded because diffusion LMs park enormous attention mass on it
    (the "attention sink"), and the attender's own token positions are excluded
    because self-attention would otherwise win almost every row. Both
    exclusions are applied before the argmax, never after.
    """
    row_idx = attender_span[-1] if attender_token == "last" else attender_span[0]
    row = attentions[layer][0, :, row_idx, :].detach().float().clone()

    if exclude_bos:
        row[:, 0] = float("-inf")
    if exclude_self:
        for col in attender_span:
            row[:, col] = float("-inf")

    return row.argmax(dim=1).cpu().numpy()

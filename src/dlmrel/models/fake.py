"""Deterministic tiny CPU adapter for protocol and artifact tests."""

from __future__ import annotations

import torch

from .base import AdapterOutput, Capabilities, ModelAdapter


class FakeAdapter(ModelAdapter):
    capabilities = Capabilities(
        logits=True,
        hidden_states=True,
        attentions=True,
        native_timestep=True,
        head_residuals=True,
        head_ablation=True,
        native_generation=False,
    )

    def __init__(self, layers: int = 2, heads: int = 3, hidden: int = 12, vocab: int = 32):
        self.layers = layers
        self.heads = heads
        self.hidden = hidden
        self.vocab = vocab
        self.device = "cpu"
        self.mask_free = False

    def forward(
        self, input_ids: torch.Tensor, *, timestep: int = 0, ablate: tuple[int, int] | None = None
    ) -> AdapterOutput:
        batch, seq = input_ids.shape
        positions = torch.arange(seq, dtype=torch.float32)
        attentions = []
        head_outputs = torch.zeros(self.layers, batch, self.heads, seq, self.hidden)
        for layer in range(self.layers):
            scores = torch.empty(batch, self.heads, seq, seq)
            for head in range(self.heads):
                distance = (positions[:, None] - positions[None, :]).abs()
                raw = -(distance - float(head + 1)).abs() + layer * 0.01 + timestep * 0.001
                scores[:, head] = raw.softmax(dim=-1)
                head_outputs[layer, :, head, :, head] = 1.0 + layer
            if ablate and ablate[0] == layer:
                scores[:, ablate[1]] = 0.0
                head_outputs[layer, :, ablate[1]] = 0.0
            attentions.append(scores)
        basis = torch.nn.functional.one_hot(input_ids % self.hidden, self.hidden).float()
        hidden_states = tuple(basis + layer for layer in range(self.layers + 1))
        unembed = (
            torch.arange(self.hidden * self.vocab, dtype=torch.float32).reshape(
                self.hidden, self.vocab
            )
            / 1000
        )
        logits = hidden_states[-1] @ unembed
        if ablate:
            contribution = torch.zeros(batch, seq, self.hidden)
            contribution[:, :, ablate[1]] = 1.0 + ablate[0]
            logits = logits - contribution @ unembed
        return AdapterOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=tuple(attentions),
            attention_mask=torch.ones(batch, seq, dtype=torch.bool),
            visibility_mask=input_ids.ne(0),
            head_outputs=head_outputs,
            native_timestep=timestep,
        )

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(input_ids % self.hidden, self.hidden).float()

    def forward_attentions(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        output = self.forward(input_ids)
        if output_hidden_states:
            return output.logits, output.attentions, output.hidden_states
        return output.logits, output.attentions

    def ablation_delta(
        self,
        input_ids: torch.Tensor,
        layer: int,
        head: int,
        target_position: int,
        target_token: int,
    ) -> float:
        base = self.forward(input_ids).logits[0, target_position, target_token]
        ablated = self.forward(input_ids, ablate=(layer, head)).logits[
            0, target_position, target_token
        ]
        return float(base - ablated)

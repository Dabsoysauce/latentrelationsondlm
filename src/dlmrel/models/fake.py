"""Deterministic tiny CPU adapter for protocol and artifact tests."""

from __future__ import annotations

import torch

from .base import AdapterOutput, Capabilities, ModelAdapter


class FakeAdapter(ModelAdapter):
    prediction_offset = 0
    capabilities = Capabilities(
        logits=True,
        hidden_states=True,
        attentions=True,
    )

    def __init__(self, layers: int = 2, heads: int = 3, hidden: int = 12, vocab: int = 32):
        self.layers = layers
        self.heads = heads
        self.hidden = hidden
        self.vocab = vocab
        self.device = "cpu"
        self.mask_free = False
        self.tokenizer = None
        self.projection_weight = (
            torch.arange(hidden * hidden, dtype=torch.float32).reshape(hidden, hidden) / 100.0
        )

    def forward(self, input_ids: torch.Tensor, *, timestep: int = 0) -> AdapterOutput:
        batch, seq = input_ids.shape
        positions = torch.arange(seq, dtype=torch.float32)
        attentions = []
        for layer in range(self.layers):
            scores = torch.empty(batch, self.heads, seq, seq)
            for head in range(self.heads):
                distance = (positions[:, None] - positions[None, :]).abs()
                raw = -(distance - float(head + 1)).abs() + layer * 0.01 + timestep * 0.001
                scores[:, head] = raw.softmax(dim=-1)
            attentions.append(scores)
        basis = torch.nn.functional.one_hot(input_ids % self.hidden, self.hidden).float()
        hidden_states = tuple(basis + layer for layer in range(self.layers + 1))
        unembed = (
            torch.arange(self.hidden * self.vocab, dtype=torch.float32).reshape(self.hidden, self.vocab)
            / 1000
        )
        logits = hidden_states[-1] @ unembed
        return AdapterOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=tuple(attentions),
            attention_mask=torch.ones(batch, seq, dtype=torch.bool),
            visibility_mask=input_ids.ne(0),
        )

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(input_ids % self.hidden, self.hidden).float()

    def forward_attentions(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        output = self.forward(input_ids)
        if output_hidden_states:
            return output.logits, output.attentions, output.hidden_states
        return output.logits, output.attentions

    def projection_input(self, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        """Deterministic concatenated-head values for exact decomposition tests."""
        if not 0 <= layer < self.layers:
            raise ValueError("fake layer outside model")
        basis = self.get_embeds(input_ids)
        return basis + float(layer) / 10.0

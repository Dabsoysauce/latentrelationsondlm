"""Common model-adapter interface.

Every model is wrapped in a ModelAdapter so the shared experiment code reads
attention and hidden states the same way regardless of architecture. Loaders
live in the per-model files (dream.py, diffugpt.py, ...) and return an adapter
plus its tokenizer and a metadata dict.

Only eager attention returns attention weights; sdpa and flash_attention_2
accept output_attentions=True and silently return None, which turns every
downstream accuracy into zero. An adapter must load its backbone eagerly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ModelAdapter(ABC):
    # Bidirectional models (Dream) run with attention_mask=None. Causal-derived
    # diffusion models (DiffuGPT, DiffuLLaMA) build an anneal mask instead, so
    # the diffusion code branches on this flag.
    mask_free: bool = False

    def __init__(self, backbone, tokenizer, device: str):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Input embeddings for a token-id tensor."""

    @abstractmethod
    def forward_attentions(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        """One denoising forward pass.

        Returns (logits, attentions) or, when output_hidden_states, adds the
        per-layer hidden states as a third element. attentions[layer] has shape
        [batch, heads, seq_len, seq_len].
        """

    def get_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project a hidden state through the LM head. Needed by logit_lens."""
        raise NotImplementedError

    def get_final_norm(self):
        """The norm applied after the last block, needed before get_logits on
        an intermediate layer. Needed by logit_lens."""
        raise NotImplementedError

    def get_lm_head(self):
        """The output projection module. Needed by logit_lens."""
        raise NotImplementedError

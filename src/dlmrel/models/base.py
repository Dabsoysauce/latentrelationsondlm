from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Capabilities:
    logits: bool = False
    hidden_states: bool = False
    attentions: bool = False


@dataclass
class AdapterOutput:
    """Normalized output used by the deterministic CPU test adapter."""

    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    attention_mask: torch.Tensor | None = None
    visibility_mask: torch.Tensor | None = None


class ModelAdapter(ABC):
    mask_free: bool = False
    capabilities: Capabilities = Capabilities()

    def __init__(self, backbone, tokenizer, device: str):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def forward_attentions(self, input_ids: torch.Tensor, output_hidden_states: bool = False): ...

    def get_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_final_norm(self):
        raise NotImplementedError

    def get_lm_head(self):
        raise NotImplementedError

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Capabilities:
    logits: bool = False
    hidden_states: bool = False
    attentions: bool = False
    native_timestep: bool = False
    head_residuals: bool = False
    head_ablation: bool = False
    native_generation: bool = False

    def require(self, name: str) -> None:
        if not getattr(self, name, False):
            raise NotImplementedError(f"adapter capability {name!r} is unsupported")


@dataclass
class AdapterOutput:
    """Normalized shapes: logits [B,S,V], hidden [L+1,B,S,D], attention [L,B,H,S,S]."""

    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    attention_mask: torch.Tensor | None = None
    visibility_mask: torch.Tensor | None = None
    head_outputs: torch.Tensor | None = None
    native_timestep: Any = None


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

    def metadata(self) -> dict[str, Any]:
        return {"capabilities": self.capabilities.__dict__.copy()}

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ModelAdapter(ABC):
    mask_free: bool = False

    def __init__(self, backbone, tokenizer, device: str):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def forward_attentions(
        self, input_ids: torch.Tensor, output_hidden_states: bool = False
    ): ...

    def get_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_final_norm(self):
        raise NotImplementedError

    def get_lm_head(self):
        raise NotImplementedError

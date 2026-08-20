"""The attention row-sum check has to survive bfloat16 without going blind."""

from __future__ import annotations

import pytest
import torch

from dlmrel.artifacts import ArtifactError
from dlmrel.config import RunConfig, RuntimeConfig
from dlmrel.models.base import Capabilities
from dlmrel.pipeline import model_smoke_report

ROOT = "configs"


class _Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return list(range(2, 2 + len(text.split())))


class _Model(torch.nn.Module):
    """Emits real softmax attention in a chosen dtype, optionally unnormalized."""

    capabilities = Capabilities(logits=True, hidden_states=True, attentions=True)

    def __init__(self, dtype, *, scale=1.0, n_layers=4, n_heads=4):
        super().__init__()
        self.dtype_ = dtype
        self.scale = scale
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = "cpu"

    def _attention(self, seq_len):
        torch.manual_seed(0)
        rows = torch.softmax(torch.randn(1, self.n_heads, seq_len, seq_len) * 3, dim=-1)
        return (rows * self.scale).to(self.dtype_)

    def forward_attentions(self, input_ids, output_hidden_states=False):
        seq_len = input_ids.shape[1]
        attentions = tuple(self._attention(seq_len) for _ in range(self.n_layers))
        hidden = tuple(
            torch.zeros(1, seq_len, 8, dtype=self.dtype_) for _ in range(self.n_layers + 1)
        )
        if output_hidden_states:
            return None, attentions, hidden
        return None, attentions

    def get_logits(self, hidden_state):
        return torch.zeros(1, hidden_state.shape[1], 16, dtype=self.dtype_)

    def get_lm_head(self):
        return self.get_logits


def _config():
    return RunConfig.load_files(
        f"{ROOT}/models/fake.yaml",
        f"{ROOT}/datasets/ewt.yaml",
        f"{ROOT}/experiments/head_search.yaml",
        runtime=RuntimeConfig(),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_correctly_normalized_attention_passes_in_every_dtype(dtype):
    """Regression: bfloat16 rounding alone exceeded the old fixed 1e-3 bound."""
    report = model_smoke_report(_Model(dtype), _Tokenizer(), _config(), {})
    assert report["status"] == "passed"
    assert report["attention_row_sum_max_error"] <= report["attention_row_sum_tolerance"]


def test_genuinely_unnormalized_attention_still_fails():
    """The looser bound must not hide rows that are actually not a distribution."""
    with pytest.raises(ArtifactError, match="attention rows do not sum to one"):
        model_smoke_report(_Model(torch.bfloat16, scale=1.5), _Tokenizer(), _config(), {})


def test_the_failure_reports_the_measured_error():
    """The original message named no number, so a caller could not tell 2e-3 from 5.0."""
    with pytest.raises(ArtifactError) as caught:
        model_smoke_report(_Model(torch.float32, scale=2.0), _Tokenizer(), _config(), {})
    assert "max |row sum - 1| =" in str(caught.value)


def test_bfloat16_rounding_alone_would_have_failed_the_old_bound():
    """Pins why the tolerance changed, so nobody tightens it back to 1e-3."""
    report = model_smoke_report(_Model(torch.bfloat16), _Tokenizer(), _config(), {})
    assert report["attention_row_sum_max_error"] > 1e-3
    assert report["attention_dtype"] == "torch.bfloat16"

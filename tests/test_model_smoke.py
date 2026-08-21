from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from dlmrel.artifacts import ArtifactError
from dlmrel.models.base import Capabilities
from dlmrel.models.fake import FakeAdapter
from dlmrel.pipeline import (
    ATTENTION_ROW_SUM_TOLERANCE,
    attention_normalization_diagnostics,
    model_smoke_report,
)


class _Tokenizer:
    bos_token_id = 1

    def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [2, 3, 4, 5]


def _config():
    capabilities = Capabilities(logits=True, hidden_states=True, attentions=True)
    model = SimpleNamespace(
        id="test-model",
        name="test/checkpoint",
        revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        remote_code_revision="remote-code-revision",
        capabilities=capabilities,
    )
    return SimpleNamespace(model=model)


class _SmokeAdapter(FakeAdapter):
    def get_lm_head(self):
        unembed = (
            torch.arange(self.hidden * self.vocab, dtype=torch.float32).reshape(
                self.hidden,
                self.vocab,
            )
            / 1000
        )
        return lambda hidden: hidden @ unembed


class _MalformedAttentionAdapter(_SmokeAdapter):
    def forward_attentions(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        logits, attentions, hidden_states = super().forward_attentions(
            input_ids,
            output_hidden_states=True,
        )
        malformed = list(attentions)
        malformed[1] = malformed[1].clone()
        malformed[1][0, 0, 2] *= 0.5
        if output_hidden_states:
            return logits, tuple(malformed), hidden_states
        return logits, tuple(malformed)


def test_model_smoke_reports_normalized_attention():
    report = model_smoke_report(_SmokeAdapter(), _Tokenizer(), _config(), {"source": "test"})

    diagnostics = report["attention_normalization"]
    assert report["status"] == "passed"
    assert diagnostics["passed"] is True
    assert diagnostics["rows_exceeding_tolerance"] == 0
    assert diagnostics["max_error_from_one"] <= ATTENTION_ROW_SUM_TOLERANCE
    assert all(layer["shape"] == [1, 3, 5, 5] for layer in diagnostics["layers"])
    assert all(layer["dtype"] == "float32" for layer in diagnostics["layers"])
    assert diagnostics["mask_padding_assessment"]["padding_can_explain_failure"] is False


def test_measured_dream_bfloat16_row_sum_error_passes():
    row = torch.tensor([205, 205, 205, 206, 206], dtype=torch.float32).div(1024)
    attention = row.to(torch.bfloat16).reshape(1, 1, 1, 5).repeat(1, 1, 5, 1)

    diagnostics = attention_normalization_diagnostics((attention,), sequence_length=5)

    assert ATTENTION_ROW_SUM_TOLERANCE == 1e-2
    assert diagnostics["max_error_from_one"] == pytest.approx(0.0029296875)
    assert diagnostics["layers"][0]["dtype"] == "bfloat16"
    assert diagnostics["rows_exceeding_tolerance"] == 0
    assert diagnostics["passed"] is True


def test_model_smoke_rejects_malformed_attention_with_layer_diagnostics():
    with pytest.raises(ArtifactError, match="attention rows do not sum to one") as raised:
        model_smoke_report(
            _MalformedAttentionAdapter(),
            _Tokenizer(),
            _config(),
            {"source": "test"},
        )

    diagnostic_text = str(raised.value).split("diagnostics=", maxsplit=1)[1]
    diagnostics = json.loads(diagnostic_text)
    failed_layer = diagnostics["layers"][1]
    assert diagnostics["passed"] is False
    assert diagnostics["rows_exceeding_tolerance"] == 1
    assert failed_layer["rows_exceeding_tolerance"] == 1
    assert failed_layer["row_sum_min"] == pytest.approx(0.5)
    assert failed_layer["row_sum_max"] == pytest.approx(1.0)
    assert failed_layer["max_error_from_one"] == pytest.approx(0.5)
    assert failed_layer["representative_worst_rows"][0]["row_sum"] == pytest.approx(0.5)
    assert diagnostics["mask_padding_assessment"] == {
        "attention_shapes_match_unpadded_input": True,
        "input_padding_applied": False,
        "input_sequence_length": 5,
        "padding_can_explain_failure": False,
        "reason": (
            "The smoke input is one unpadded sequence. Causal or bidirectional key masks may zero "
            "individual attention entries, but valid softmax rows must still sum to one."
        ),
    }

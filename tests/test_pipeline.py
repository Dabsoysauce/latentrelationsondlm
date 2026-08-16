from __future__ import annotations

import pytest
import torch

from dlmrel.artifacts import ArtifactError
from dlmrel.pipeline import _attention_normalization_report


def test_bfloat16_attention_roundoff_is_measured_not_misclassified():
    layer = torch.tensor([[[[1 / 3, 1 / 3, 1 / 3]]]], dtype=torch.bfloat16)

    report = _attention_normalization_report((layer,))

    assert report["attention_row_sum_value"] == 1.001953125
    assert report["attention_row_sum_max_error"] == 0.001953125
    expected_bound = torch.finfo(torch.bfloat16).eps / 2 + torch.finfo(torch.float32).eps
    assert report["attention_row_sum_allowed_error"] == expected_bound
    assert report["attention_row_sum_dtype"] == "torch.bfloat16"


def test_genuinely_unnormalized_attention_is_rejected_with_measurement():
    layer = torch.tensor([[[[0.2, 0.2, 0.2]]]], dtype=torch.float32)

    with pytest.raises(
        ArtifactError,
        match=r"attention rows do not sum to one: max_abs_error=.*dtype=torch.float32",
    ):
        _attention_normalization_report((layer,))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_attention_is_rejected(bad_value):
    layer = torch.tensor([[[[bad_value, 0.0]]]], dtype=torch.float32)

    with pytest.raises(ArtifactError, match="contains non-finite values"):
        _attention_normalization_report((layer,))

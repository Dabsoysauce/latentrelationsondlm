from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dlmrel.experiments import attention_entropy


def test_entropy_rows_match_hand_calculation_and_renormalize(monkeypatch):
    attention = torch.tensor([[[[2.0, 2.0], [3.0, 0.0]]]])
    state = SimpleNamespace(n_masked=1)
    monkeypatch.setattr(
        attention_entropy,
        "attentions_at_time",
        lambda *_args, **_kwargs: ((attention,), state),
    )
    example = SimpleNamespace(sentence_id="s1", source="ewt", text="x")
    cfg = SimpleNamespace(experiment=SimpleNamespace(steps=64))

    row = attention_entropy.entropy_rows(
        object(), object(), [example], cfg, seed=42, progress=0.5
    ).iloc[0]

    assert row.timestep == 32
    assert row.entropy == pytest.approx(np.log(2) / 2)
    assert row.entropy_normalized == pytest.approx(0.5)
    assert row.entropy_no_bos == pytest.approx(0.0)
    assert row.bos_sink_mass == pytest.approx(0.75)


def test_all_bos_entropy_is_finite_and_normalized_to_zero(monkeypatch):
    attention = torch.tensor([[[[1.0]]]])
    monkeypatch.setattr(
        attention_entropy,
        "attentions_at_time",
        lambda *_args, **_kwargs: ((attention,), SimpleNamespace(n_masked=0)),
    )
    example = SimpleNamespace(sentence_id="s1", source="ewt", text="")
    cfg = SimpleNamespace(experiment=SimpleNamespace(steps=64))

    row = attention_entropy.entropy_rows(
        object(), object(), [example], cfg, seed=42, progress=1.0
    ).iloc[0]

    assert row.entropy == 0.0
    assert row.entropy_normalized == 0.0
    assert row.entropy_no_bos == 0.0
    values = row[["entropy", "entropy_normalized", "entropy_no_bos"]].to_numpy(dtype=float)
    assert np.isfinite(values).all()

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from dlmrel.experiments import logit_lens


class ScaleByTwo(torch.nn.Module):
    def forward(self, value):
        return value * 2


class TinyLensModel:
    def __init__(self):
        self.head = torch.nn.Identity()
        self.norm = ScaleByTwo()

    def get_logits(self, hidden):
        return self.head(hidden)

    def get_lm_head(self):
        return self.head

    def get_final_norm(self):
        return self.norm


def test_logit_lens_known_ranks_depths_visibility_and_final_parity(monkeypatch):
    intermediate = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]]
    )
    final = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 2.0, 1.0]]]
    )
    state = SimpleNamespace(
        input_ids=torch.tensor([[0, 1, 2]]), is_visible=[True, False, True]
    )
    monkeypatch.setattr(
        logit_lens,
        "states_at_time",
        lambda *_args, **_kwargs: ((), (intermediate, final), state),
    )
    monkeypatch.setattr(
        logit_lens,
        "tokenize",
        lambda *_args, **_kwargs: (torch.tensor([[0, 1, 2]]), ["BOS", "word", "."]),
    )
    example = SimpleNamespace(sentence_id="s1", source="ewt", text="word .")
    cfg = SimpleNamespace(experiment=SimpleNamespace(steps=64))

    rows = logit_lens.logit_lens_rows(
        TinyLensModel(), object(), [example], cfg, seed=42, progress=0.5
    )

    assert len(rows) == 6
    assert set(rows["depth"]) == {0, 1}
    assert set(rows["position"]) == {0, 1, 2}
    assert rows.groupby("position")["position_state"].first().to_dict() == {
        0: "visible",
        1: "masked",
        2: "visible",
    }
    intermediate_rows = rows[rows.depth == 0].sort_values("position")
    assert intermediate_rows["rank"].tolist() == [1, 2, 1]
    assert intermediate_rows["target_logit"].tolist() == [2.0, 2.0, 4.0]
    final_rows = rows[rows.depth == 1].sort_values("position")
    assert final_rows["rank"].tolist() == [1, 1, 3]
    assert final_rows["target_logit"].tolist() == [3.0, 3.0, 1.0]
    assert final_rows["mrr"].tolist() == pytest.approx([1.0, 1.0, 1 / 3])
    assert rows["_final_depth_parity_error"].max() == 0.0
    assert pd.api.types.is_integer_dtype(rows["top1"])

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dlmrel.artifacts import ArtifactError
from dlmrel.diffusion import endpoint_visibility, receiver_span_scores, state_at_time
from dlmrel.experiments import shared
from dlmrel.relations import Example, RelationInstance


class TinyTokenizer:
    bos_token_id = 1
    mask_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(2, 2 + int(text)))

    def decode(self, token_ids):
        return str(token_ids[0])


def test_frozen_denoising_schedule_endpoints_monotonicity_and_rng_reset():
    model = SimpleNamespace(device="cpu")
    tokenizer = TinyTokenizer()
    timesteps = [0, 8, 16, 24, 32, 39, 47, 55, 63]
    states = [state_at_time(model, tokenizer, "100", time, seed=42) for time in timesteps]

    assert states[0].is_visible == [True] + [False] * 100
    assert states[-1].is_visible == [True] * 101
    assert all(
        not earlier[position] or later[position]
        for earlier, later in zip(
            (state.is_visible for state in states),
            (state.is_visible for state in states[1:]),
            strict=False,
        )
        for position in range(101)
    )
    repeated = state_at_time(model, tokenizer, "100", 32, seed=42)
    different = state_at_time(model, tokenizer, "100", 32, seed=43)
    assert repeated.is_visible == states[4].is_visible
    assert different.is_visible != repeated.is_visible
    # The frozen implementation resets RNG per sentence, so equal token lengths
    # receive equal visibility paths even if sentence content differs.
    assert state_at_time(model, tokenizer, "0100", 32, seed=42).is_visible == repeated.is_visible


def test_denoising_rejects_invalid_time_and_whole_word_visibility_needs_all_subtokens():
    model = SimpleNamespace(device="cpu")
    with pytest.raises(ValueError, match="diffusion_time"):
        state_at_time(model, TinyTokenizer(), "4", 64, steps=64)
    assert endpoint_visibility([True, True, False, True], [1, 2], [3]) == "receiver_visible_only"


@pytest.mark.parametrize(
    "row_mode,span_mode,expected",
    [
        ("mean", "sum", [0.5, 0.25]),
        ("first", "mean", [0.15, 0.4]),
        ("last", "max", [0.4, 0.1]),
    ],
)
def test_receiver_span_aggregations_match_hand_calculation(row_mode, span_mode, expected):
    attention = torch.zeros(1, 1, 6, 6)
    attention[0, 0, 1] = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.2, 0.4])
    attention[0, 0, 2] = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.4, 0.1])

    scores = receiver_span_scores(
        (attention,),
        0,
        [1, 2],
        [[3, 4], [5]],
        row_aggregation=row_mode,
        span_aggregation=span_mode,
        excluded_positions={1, 2},
    )

    np.testing.assert_allclose(scores[0], expected)


def _scoring_example(receiver_word_idx=2):
    relation = RelationInstance(
        relation="object_to_verb",
        attender_span=[2],
        receiver_span=[3],
        attender_text="query",
        receiver_text="gold",
        attender_word_idx=1,
        receiver_word_idx=receiver_word_idx,
        dep="obj",
        instance_id="s:r",
        attender_upos="NOUN",
        receiver_upos="VERB",
    )
    return Example(
        text="punct query gold",
        tokens=["!", "query", "gold"],
        upos=["PUNCT", "NOUN", "VERB"],
        deprel=["punct", "obj", "root"],
        head=[2, 3, 0],
        word_to_tokens={0: [1], 1: [2], 2: [3]},
        relations=[relation],
        seq_len=4,
        source="ewt",
        sentence_id="s",
        language="en",
        original_split="test",
    )


def test_scoring_candidate_order_includes_punctuation_and_breaks_exact_tie_first(monkeypatch):
    attention = torch.full((1, 1, 4, 4), 0.25)
    state = SimpleNamespace(is_visible=[True, False, False, False])
    monkeypatch.setattr(shared, "attentions_at_time", lambda *_args, **_kwargs: ((attention,), state))
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            steps=64,
            seeds=[42, 43, 44],
            scoring=SimpleNamespace(
                primary_visibility="both_masked",
                attender_rows="mean",
                receiver_span="sum",
            ),
        )
    )

    row = shared.score_attention_heads(
        object(), object(), [_scoring_example()], cfg, role="test", seed=42
    ).iloc[0]

    assert row.predicted_word_idx == 0
    assert row.n_candidate_words == 2
    assert row.correct == 0


def test_scoring_rejects_an_invalid_gold_receiver(monkeypatch):
    attention = torch.full((1, 1, 4, 4), 0.25)
    state = SimpleNamespace(is_visible=[True, False, False, False])
    monkeypatch.setattr(shared, "attentions_at_time", lambda *_args, **_kwargs: ((attention,), state))
    example = _scoring_example(receiver_word_idx=1)
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            steps=64,
            seeds=[42, 43, 44],
            scoring=SimpleNamespace(
                primary_visibility="both_masked",
                attender_rows="mean",
                receiver_span="sum",
            ),
        )
    )
    with pytest.raises(ArtifactError, match="gold receiver"):
        shared.score_attention_heads(object(), object(), [example], cfg, role="test")

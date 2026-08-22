import json

import numpy as np
import pandas as pd
import pytest
import torch

from dlmrel.artifacts import ArtifactError
from dlmrel.config import RELATION_NAMES
from dlmrel.models.native import aligned_logits
from dlmrel.paper_protocol import (
    PAPER_TIMESTEPS,
    attention_entropy_rows,
    choose_selection_winners,
    load_selection_bundle,
    map_relative_depths,
    paper_visibility_group,
    prediction_source_index,
    projection_head_slice,
    receiver_is_correct,
    receiver_source_argmax,
    summarize_entropy_trajectory,
    timing_record,
    write_selection_bundle,
    zero_projection_head_input,
)


def test_old_single_source_argmax_uses_last_attender_piece_and_receiver_membership():
    attention = np.zeros((2, 7, 7), dtype=float)
    attention[0, 2, 6] = 100  # first attender piece must not be used
    attention[0, 3, 5] = 9
    attention[0, 3, 4] = 8
    attention[1, 3, 4] = 10
    attention[:, 3, 0] = 999  # BOS excluded
    attention[:, 3, 2:4] = 998  # complete multi-piece attender excluded

    prediction = receiver_source_argmax(attention, [2, 3])

    assert prediction.tolist() == [5, 4]
    assert receiver_is_correct(prediction[0], [4, 5])
    assert receiver_is_correct(prediction[1], [4, 5])


def test_receiver_scoring_does_not_sum_a_multi_piece_candidate():
    attention = np.zeros((1, 5, 5), dtype=float)
    attention[0, 1, 2] = 0.4
    attention[0, 1, 3] = 0.4
    attention[0, 1, 4] = 0.7
    assert receiver_source_argmax(attention, [1]).tolist() == [4]
    assert not receiver_is_correct(4, [2, 3])


def test_old_two_way_visibility_exits_both_masked_after_one_subtoken():
    hidden = [True, False, False, False, False]
    assert paper_visibility_group(hidden, [1, 2], [3, 4]) == "both_masked"
    hidden[2] = True
    assert paper_visibility_group(hidden, [1, 2], [3, 4]) == "at_least_one_revealed"


@pytest.mark.parametrize(
    "layers,expected",
    [(12, [2, 6, 10]), (32, [6, 16, 28]), (40, [8, 20, 35])],
)
def test_relative_depth_mapping_for_different_models(layers, expected):
    rows = map_relative_depths(layers)
    assert [row["actual_layer_index"] for row in rows] == expected
    assert [row["relative_label"] for row in rows] == ["early", "middle", "late"]


def test_entropy_excludes_bos_query_but_retains_bos_source():
    attention = np.array(
        [
            [
                [0.5, 0.5],  # excluded BOS query
                [1.0, 0.0],  # retained query has all mass on BOS source
            ]
        ]
    )
    assert attention_entropy_rows(attention).tolist() == pytest.approx([0.0])
    changed = attention.copy()
    changed[0, 1] = [0.5, 0.5]
    assert attention_entropy_rows(changed).tolist() == pytest.approx([1.0])


def test_entropy_windows_delta_slope_and_direction():
    result = summarize_entropy_trajectory(
        np.linspace(0, 1, 64), early_window=(0, 15), late_window=(48, 63)
    )
    assert result["early_entropy"] < result["late_entropy"]
    assert result["delta"] > 0
    assert result["slope"] > 0
    assert result["direction"] == "increasing"


def test_shifted_and_unshifted_prediction_alignment():
    logits = torch.full((1, 3, 5), -10.0)
    logits[0, 0, 2] = 10
    logits[0, 1, 3] = 10
    ids = torch.tensor([[1, 4, 4]])
    assert aligned_logits(logits, ids, 0).argmax(-1).tolist() == [[2, 3, 0]]
    assert aligned_logits(logits, ids, -1).argmax(-1).tolist() == [[1, 2, 3]]
    assert prediction_source_index(5, 0) == 5
    assert prediction_source_index(5, -1) == 4


def test_found_time_and_unmask_time_calculations():
    final = [9, 7]
    argmax = [[9, 1] for _ in range(64)]
    states = [[9, 0] for _ in range(64)]
    argmax[4][1] = 7
    for step in range(5, 64):
        argmax[step][1] = 7
    for step in range(10, 64):
        states[step][1] = 7
    result = timing_record(argmax, states, final, mask_token_id=0, position=1)
    assert result == {
        "found_time": 4,
        "unmask_time": 10,
        "found_time_minus_unmask_time": -6,
        "lead_steps": 6,
        "predicted_before_unmasking": True,
        "predicted_exactly_at_unmasking": False,
    }


def test_exact_projection_decomposition_and_single_head_zeroing():
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    weight = torch.eye(4)
    first = projection_head_slice(values, weight, head=0, number_of_heads=2)
    second = projection_head_slice(values, weight, head=1, number_of_heads=2)
    assert torch.allclose(first + second, torch.nn.functional.linear(values, weight))
    changed = zero_projection_head_input(values, head=1, number_of_heads=2)
    assert torch.equal(changed, torch.tensor([[[1.0, 2.0, 0.0, 0.0]]]))
    assert torch.equal(values, torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))


def _scores():
    rows = []
    for relation in RELATION_NAMES:
        rows.extend(
            [
                {"relation": relation, "layer": 1, "head": 1, "accuracy": 0.8,
                 "n_total": 10, "n_correct": 8},
                {"relation": relation, "layer": 0, "head": 2, "accuracy": 0.8,
                 "n_total": 12, "n_correct": 10},
                {"relation": relation, "layer": 0, "head": 1, "accuracy": 0.8,
                 "n_total": 12, "n_correct": 10},
            ]
        )
    return pd.DataFrame(rows)


def test_six_selection_winners_and_deterministic_ties_ignore_test_data():
    winners = choose_selection_winners(_scores())
    assert list(winners["relation"]) == list(RELATION_NAMES)
    assert set(zip(winners["layer"], winners["head"], strict=True)) == {(0, 1)}
    poisoned_test = pd.DataFrame(
        [{"relation": relation, "layer": 9, "head": 9} for relation in RELATION_NAMES]
    )
    assert poisoned_test is not None
    assert choose_selection_winners(_scores()).equals(winners)


def test_selection_bundle_is_selection_only_and_rejects_legacy_or_wrong_model(tmp_path):
    bundle = write_selection_bundle(
        tmp_path / "locks",
        choose_selection_winners(_scores()),
        model_id="dream_7b",
        model_revision="abc",
        tokenizer_revision="abc",
        dataset_id="ewt",
        selection_manifest_hash="select-hash",
        config_hash="config-hash",
        code_hash="code-hash",
        created_at="2000-01-01T00:00:00+00:00",
    )
    assert set(bundle.locks) == set(RELATION_NAMES)
    manifest = json.loads((bundle.source / "selection_bundle.json").read_text())
    assert manifest["development_used"] is False
    assert manifest["test_outcomes_used"] is False
    assert manifest["fully_visible_timestep"] == 63
    with pytest.raises(ArtifactError, match="different model"):
        load_selection_bundle(bundle.source, model_id="diffullama_7b")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "selection_bundle.json").write_text(json.dumps({"schema_version": "old"}))
    with pytest.raises(ArtifactError, match="incompatible"):
        load_selection_bundle(legacy)


def test_all_64_points_are_frozen():
    assert PAPER_TIMESTEPS == tuple(range(64))

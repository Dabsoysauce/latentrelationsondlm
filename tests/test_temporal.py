import pandas as pd

from dlmrel.diffusion import endpoint_visibility
from dlmrel.experiments.time_curve import aggregate_curve


def test_four_visibility_states():
    assert endpoint_visibility([False, False], [0], [1]) == "both_masked"
    assert endpoint_visibility([True, False], [0], [1]) == "attender_visible_only"
    assert endpoint_visibility([False, True], [0], [1]) == "receiver_visible_only"
    assert endpoint_visibility([True, True], [0], [1]) == "both_visible"


def test_timestep_is_not_collapsed():
    frame = pd.DataFrame(
        [
            {
                "relation": "r",
                "treebank": "ewt",
                "layer": 0,
                "head": 0,
                "seed": 42,
                "timestep": 0,
                "normalized_progress": 0.0,
                "visibility": "both_visible",
                "correct": 0,
            },
            {
                "relation": "r",
                "treebank": "ewt",
                "layer": 0,
                "head": 0,
                "seed": 42,
                "timestep": 8,
                "normalized_progress": 1.0,
                "visibility": "both_visible",
                "correct": 1,
            },
        ]
    )
    _, output = aggregate_curve(frame)
    assert output["timestep"].tolist() == [0, 8]
    assert output["accuracy_mean"].tolist() == [0.0, 1.0]

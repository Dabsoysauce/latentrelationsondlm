import pandas as pd

from dlmrel.evaluation.compare_models import common_instance_comparison


def test_comparison_uses_common_instance_intersection():
    first = pd.DataFrame(
        {
            "model": ["a", "a"],
            "treebank": ["ewt", "ewt"],
            "relation": ["r", "r"],
            "instance_id": ["i1", "i2"],
            "correct": [1, 0],
        }
    )
    second = pd.DataFrame(
        {
            "model": ["b", "b"],
            "treebank": ["ewt", "ewt"],
            "relation": ["r", "r"],
            "instance_id": ["i2", "i3"],
            "correct": [1, 1],
        }
    )
    output = common_instance_comparison([first, second])
    assert set(output["n_common_instances"]) == {1}
    assert set(output["accuracy"]) == {0.0, 1.0}

import pandas as pd

from dlmrel.experiments.shared import selection_aware_permutation


def _rows(role):
    rows = []
    for instance in range(12):
        gold = instance % 3
        for head in range(3):
            rows.append(
                {
                    "role": role,
                    "relation": "r",
                    "sentence_id": f"s{instance}",
                    "instance_id": f"i{instance}",
                    "layer": 0,
                    "head": head,
                    "gold_receiver_word_idx": gold,
                    "predicted_word_idx": gold if head == 0 else head,
                    "correct": int(head == 0),
                }
            )
    return pd.DataFrame(rows)


def test_selection_aware_permutation_is_deterministic():
    first = selection_aware_permutation(
        _rows("select"),
        _rows("dev"),
        relation="r",
        top_k=2,
        n_permutations=30,
        seed=42,
    )
    second = selection_aware_permutation(
        _rows("select"),
        _rows("dev"),
        relation="r",
        top_k=2,
        n_permutations=30,
        seed=42,
    )
    assert first == second
    assert 0 < first["p_value"] <= 1

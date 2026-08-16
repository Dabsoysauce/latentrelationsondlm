import pandas as pd

from dlmrel.selection import create_selection_lock, locked_test_view


def _scores(values):
    return pd.DataFrame(
        [
            {
                "relation": "object_to_verb",
                "layer": layer,
                "head": head,
                "accuracy": accuracy,
                "n_total": 100,
            }
            for layer, head, accuracy in values
        ]
    )


def test_test_scores_cannot_change_locked_head():
    select = _scores([(0, 0, 0.9), (0, 1, 0.8), (1, 0, 0.7)])
    dev = _scores([(0, 0, 0.4), (0, 1, 0.8), (1, 0, 0.9)])
    lock, _, _ = create_selection_lock(
        select,
        dev,
        relation="object_to_verb",
        top_k=2,
        track="confirmatory_ewt",
        model_id="fake",
        model_revision="local-v1",
        dataset_id="ewt",
        config_hash="c",
        select_manifest_hash="s",
        dev_manifest_hash="d",
        frozen_settings={"fixed_offset": -2},
    )
    assert (lock.layer, lock.head) == (0, 1)
    test = _scores([(0, 0, 1.0), (0, 1, 0.0), (1, 0, 1.0)])
    view = locked_test_view(test, lock)
    assert len(view) == 1
    assert (view.iloc[0].layer, view.iloc[0]["head"]) == (0, 1)


def test_dev_tie_break_is_deterministic():
    select = _scores([(1, 1, 0.9), (0, 2, 0.9)])
    dev = _scores([(1, 1, 0.5), (0, 2, 0.5)])
    lock, _, _ = create_selection_lock(
        select,
        dev,
        relation="object_to_verb",
        top_k=2,
        track="confirmatory_ewt",
        model_id="fake",
        model_revision="local-v1",
        dataset_id="ewt",
        config_hash="c",
        select_manifest_hash="s",
        dev_manifest_hash="d",
        frozen_settings={},
    )
    assert (lock.layer, lock.head) == (0, 2)

from __future__ import annotations

import json

import pandas as pd
import pytest

from dlmrel.artifacts import ArtifactError
from dlmrel.head_search_recovery import _apply_secondary_holm
from dlmrel.permutation import randomized_receiver_labels, selection_aware_permutation
from dlmrel.relation_selection import PRIMARY_RELATION, SECONDARY_RELATIONS


def _rows(role: str, *, reverse_test: bool = False) -> pd.DataFrame:
    rows = []
    for instance in range(8):
        sentence_length = 4 + (instance % 2)
        attender = instance % sentence_length
        candidates = [word for word in range(sentence_length) if word != attender]
        gold = candidates[instance % len(candidates)]
        for seed in (42, 43):
            for head in range(3):
                if role == "select":
                    prediction = gold if head == 0 or (head == 1 and instance % 2 == 0) else candidates[0]
                elif role == "dev":
                    prediction = gold if head == 1 or (head == 0 and instance % 3 == 0) else candidates[-1]
                else:
                    successful = head == 1
                    if reverse_test:
                        successful = head == 2
                    prediction = gold if successful else candidates[(instance + head + 1) % len(candidates)]
                rows.append(
                    {
                        "role": role,
                        "relation": "r",
                        "sentence_id": f"{role}-sentence-{instance}",
                        "instance_id": f"{role}-instance-{instance}",
                        "seed": seed,
                        "layer": 0,
                        "head": head,
                        "attender_word_idx": attender,
                        "sentence_length_words": sentence_length,
                        "n_candidate_words": len(candidates),
                        "gold_receiver_word_idx": gold,
                        "predicted_word_idx": prediction,
                        "correct": int(prediction == gold),
                    }
                )
    return pd.DataFrame(rows)


def _run(select, dev, test, **overrides):
    options = {
        "relation": "r",
        "top_k": 2,
        "n_permutations": 20,
        "seed": 42,
        "scientific_config_hash": "scientific-config",
        "progress_interval": 0,
        "checkpoint_interval": 5,
    }
    options.update(overrides)
    return selection_aware_permutation(select, dev, test, **options)


def _slow_reference(select, dev, test, *, permutations: int) -> list[float]:
    frames = {"select": select, "dev": dev, "test": test}
    statistics = []
    for permutation_index in range(permutations):
        scores = {}
        for role, frame in frames.items():
            labels = randomized_receiver_labels(
                frame,
                relation="r",
                role=role,
                seed=42,
                permutation_index=permutation_index,
            )
            permuted = frame.copy()
            permuted["permuted"] = [
                labels[(row.sentence_id, row.instance_id)]
                for row in permuted.itertuples(index=False)
            ]
            permuted["null_correct"] = (
                permuted["predicted_word_idx"] == permuted["permuted"]
            ).astype(int)
            scores[role] = permuted.groupby(["layer", "head"], as_index=False).agg(
                accuracy=("null_correct", "mean"), denominator=("null_correct", "size")
            )
        top = scores["select"].sort_values(
            ["accuracy", "denominator", "layer", "head"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).head(2)
        dev_candidates = scores["dev"].merge(top[["layer", "head"]], on=["layer", "head"])
        selected = dev_candidates.sort_values(
            ["accuracy", "denominator", "layer", "head"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).iloc[0]
        test_score = scores["test"]
        test_score = test_score[
            (test_score.layer == selected.layer) & (test_score["head"] == selected["head"])
        ].iloc[0]
        statistics.append(float(test_score.accuracy))
    return statistics


def test_random_labels_are_within_sentence_candidates_and_shared_across_heads():
    rows = _rows("select")
    labels = randomized_receiver_labels(
        rows, relation="r", role="select", seed=42, permutation_index=3
    )

    for (sentence_id, instance_id), label in labels.items():
        instance = rows[
            (rows.sentence_id == sentence_id) & (rows.instance_id == instance_id)
        ]
        attender = int(instance.attender_word_idx.iloc[0])
        sentence_length = int(instance.sentence_length_words.iloc[0])
        assert label in set(range(sentence_length)) - {attender}
        assert instance.assign(randomized=label).groupby(["sentence_id", "instance_id"])[
            "randomized"
        ].nunique().eq(1).all()


def test_optimized_permutation_matches_slow_reference_and_finite_sample_p_value():
    select, dev, test = _rows("select"), _rows("dev"), _rows("test")
    result = _run(select, dev, test)
    reference = _slow_reference(select, dev, test, permutations=20)

    assert result["null_statistics"] == reference
    expected = (1 + sum(value >= result["observed_test_accuracy"] for value in reference)) / 21
    assert result["p_value"] == pytest.approx(expected)
    assert result["null_definition"].startswith("independently within each split")


def test_permutation_is_deterministic_and_test_never_changes_selection():
    select, dev = _rows("select"), _rows("dev")
    first = _run(select, dev, _rows("test"))
    second = _run(select, dev, _rows("test"))
    changed_test = _run(select, dev, _rows("test", reverse_test=True))

    assert first == second
    assert first["selected_heads"] == changed_test["selected_heads"]
    assert first["observed_selected_layer"] == changed_test["observed_selected_layer"]
    assert first["observed_selected_head"] == changed_test["observed_selected_head"]
    assert first["null_statistics"] != changed_test["null_statistics"]

    shuffled = _run(
        select.sample(frac=1, random_state=7),
        dev.sample(frac=1, random_state=8),
        _rows("test").sample(frac=1, random_state=9),
    )
    assert shuffled == first


def test_checkpointed_resume_matches_uninterrupted_byte_for_byte(tmp_path):
    select, dev, test = _rows("select"), _rows("dev"), _rows("test")
    checkpoint = tmp_path / "permutation.json"
    partial = _run(
        select,
        dev,
        test,
        checkpoint_path=checkpoint,
        max_new_permutations=7,
    )
    assert partial["completion_status"] == "incomplete"
    assert partial["completed_permutation_indices"] == list(range(7))

    resumed = _run(select, dev, test, checkpoint_path=checkpoint, resume=True)
    uninterrupted = _run(select, dev, test)

    assert resumed == uninterrupted
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["completion_status"] == "complete"
    assert saved["null_statistics"] == uninterrupted["null_statistics"]


def test_incompatible_or_unrequested_checkpoint_reuse_is_rejected(tmp_path):
    select, dev, test = _rows("select"), _rows("dev"), _rows("test")
    checkpoint = tmp_path / "permutation.json"
    _run(select, dev, test, checkpoint_path=checkpoint, max_new_permutations=3)

    with pytest.raises(ArtifactError, match="use resume"):
        _run(select, dev, test, checkpoint_path=checkpoint)
    with pytest.raises(ArtifactError, match="scientific identity differs"):
        _run(
            select,
            dev,
            test,
            checkpoint_path=checkpoint,
            resume=True,
            scientific_config_hash="changed-science",
        )


@pytest.mark.parametrize("corruption", ["nonfinite", "head", "status", "overrun"])
def test_corrupt_permutation_progress_fails_safely(tmp_path, corruption):
    select, dev, test = (_rows(role) for role in ("select", "dev", "test"))
    checkpoint = tmp_path / "permutation.json"
    _run(select, dev, test, checkpoint_path=checkpoint, max_new_permutations=3)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    if corruption == "nonfinite":
        saved["null_statistics"][0] = float("nan")
    elif corruption == "head":
        saved["selected_heads"][0] = [999, 999]
    elif corruption == "status":
        saved["completion_status"] = "complete"
    else:
        saved["completed_permutation_indices"] = list(range(21))
        saved["null_statistics"] = [0.0] * 21
        saved["selected_heads"] = [saved["selected_heads"][0]] * 21
        saved["completion_status"] = "complete"
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ArtifactError, match="permutation checkpoint"):
        _run(select, dev, test, checkpoint_path=checkpoint, resume=True)


def test_incomplete_permutation_temporary_file_is_removed(tmp_path):
    select, dev, test = (_rows(role) for role in ("select", "dev", "test"))
    checkpoint = tmp_path / "permutation.json"
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    temporary.write_bytes(b"incomplete")

    result = _run(select, dev, test, checkpoint_path=checkpoint)

    assert result["completion_status"] == "complete"
    assert not temporary.exists()


def test_holm_adjustment_is_limited_to_the_five_predefined_secondaries():
    p_values = [0.04, 0.01, 0.03, 0.20, 0.02]
    rows = [
        {
            "relation": PRIMARY_RELATION,
            "raw_p_value": 0.001,
            "holm_adjusted_p_value": None,
        },
        *[
            {
                "relation": relation,
                "raw_p_value": p_value,
                "holm_adjusted_p_value": None,
            }
            for relation, p_value in zip(SECONDARY_RELATIONS, p_values, strict=True)
        ],
    ]

    _apply_secondary_holm(rows)

    by_relation = {row["relation"]: row for row in rows}
    assert by_relation[PRIMARY_RELATION]["holm_adjusted_p_value"] is None
    assert {
        relation: by_relation[relation]["holm_adjusted_p_value"]
        for relation in SECONDARY_RELATIONS
    } == pytest.approx(
        dict(zip(SECONDARY_RELATIONS, [0.09, 0.05, 0.09, 0.20, 0.08], strict=True))
    )

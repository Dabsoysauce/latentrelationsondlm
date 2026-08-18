import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from dlmrel.artifacts import (
    ArtifactError,
    atomic_json,
    canonical_hash,
    initialize_run,
    merge_shards,
    selection_lock_hash,
    validate_run,
    write_shard,
)
from dlmrel.experiments.shared import write_frames


def test_canonical_hash_normalizes_numpy_without_changing_plain_json():
    plain = {"span": [1, 2], "nested": {"flag": True, "score": 0.5}}
    numpy_value = {
        "span": np.array([1, 2], dtype=np.int64),
        "nested": {"flag": np.bool_(True), "score": np.float32(0.5)},
    }

    assert canonical_hash(numpy_value) == canonical_hash(plain)
    assert canonical_hash(plain) == "fb59f76e3208ee3966dee63433d8b91fcf1cd074fc290a1f0ae785fe61c87fee"


def test_atomic_json_recursively_normalizes_numpy_scalars_and_containers(tmp_path):
    path = tmp_path / "numpy.json"
    atomic_json(
        path,
        {
            "matrix": np.array([[np.int64(1), np.int64(2)]]),
            "mixed": (np.float64(1.25), [np.bool_(False)]),
        },
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "matrix": [[1, 2]],
        "mixed": [1.25, [False]],
    }


def test_selection_lock_hash_normalizes_numpy_and_ignores_only_timestamp():
    python_lock = {"created_at": "first", "frozen_settings": {"span": [1, 2]}}
    parquet_lock = {
        "created_at": "second",
        "frozen_settings": {"span": np.array([1, 2], dtype=np.int64)},
    }

    assert selection_lock_hash(python_lock) == selection_lock_hash(parquet_lock)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.float32("-inf")])
def test_non_finite_values_are_rejected(value, tmp_path):
    with pytest.raises(ArtifactError, match="non-finite"):
        canonical_hash({"value": value})
    with pytest.raises(ArtifactError, match="non-finite"):
        atomic_json(tmp_path / "invalid.json", {"value": value})


def test_parquet_span_roundtrip_can_finalize_json_shards(tmp_path):
    checkpoint = tmp_path / "span-checkpoint.parquet"
    pd.DataFrame(
        [{"instance_id": "i0", "attender_span": [1, 2], "receiver_span": [4, 5]}]
    ).to_parquet(checkpoint, index=False)
    restored = pd.read_parquet(checkpoint)
    assert isinstance(restored.iloc[0]["attender_span"], np.ndarray)

    (tmp_path / "run").mkdir()
    write_frames(tmp_path / "run", raw=restored, exclusions=pd.DataFrame())

    assert merge_shards(tmp_path / "run") == [
        {"instance_id": "i0", "attender_span": [1, 2], "receiver_span": [4, 5]}
    ]


def test_non_overwrite_resume_and_duplicate_shards(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, {"x": 1}, "command", {"select": "abc"})
    with pytest.raises(ArtifactError, match="already exists"):
        initialize_run(run, {"x": 1}, "command", {"select": "abc"})
    initialize_run(run, {"x": 1}, "command", {"select": "abc"}, resume=True)
    write_shard(run, 0, [{"x": 1}])
    write_shard(run, 0, [{"x": 1}])
    with pytest.raises(ArtifactError, match="different content"):
        write_shard(run, 0, [{"x": 2}])
    assert merge_shards(run) == [{"x": 1}]


def test_equivalent_list_and_array_shard_rewrite_verifies_but_change_fails(tmp_path):
    run = tmp_path / "run"
    write_shard(run, 0, [{"span": [1, 2], "count": 3}])
    write_shard(
        run,
        0,
        [{"span": np.array([1, 2], dtype=np.int64), "count": np.int64(3)}],
    )
    with pytest.raises(ArtifactError, match="different content"):
        write_shard(run, 0, [{"span": np.array([1, 3]), "count": 3}])


def test_incomplete_run_resumes_and_finalizes_parquet_arrays(tmp_path):
    run = tmp_path / "run"
    config = {
        "model": {"id": "fake", "revision": "r1"},
        "experiment": {"seeds": [42, 43, 44]},
        "runtime": {"resume": False, "dry_run": False, "results_root": "first"},
    }
    initialize_run(run, config, "initial", {"select": "select-hash"})
    checkpoint = run / "checkpoints" / "loaded.parquet"
    pd.DataFrame([{"instance_id": "i0", "attender_span": [1, 2]}]).to_parquet(
        checkpoint, index=False
    )
    resumed = deepcopy(config)
    resumed["runtime"].update({"resume": True, "results_root": "moved"})
    initialize_run(run, resumed, "resume", {"select": "select-hash"}, resume=True)

    write_frames(
        run,
        raw=pd.read_parquet(checkpoint),
        exclusions=pd.DataFrame(),
    )

    assert merge_shards(run) == [{"instance_id": "i0", "attender_span": [1, 2]}]


def test_validator_fails_incomplete_run(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, {"x": 1}, "command", {})
    result = validate_run(run)
    assert not result["valid"]
    assert any("missing files" in error for error in result["errors"])


def test_validator_detects_manifest_reference_tampering(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, {"runtime": {}}, "command", {"select": "original"})
    (run / "manifest_refs.json").write_text(
        json.dumps({"select": "changed"}), encoding="utf-8"
    )

    result = validate_run(run)

    assert "manifest references hash mismatch" in result["errors"]

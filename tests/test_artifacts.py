import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dlmrel.artifacts import (
    ArtifactError,
    atomic_json,
    canonical_hash,
    final_artifact_hashes,
    initialize_run,
    merge_shards,
    selection_lock_hash,
    validate_run,
    write_shard,
)
from dlmrel.config import RunConfig
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


def test_nullable_table_values_are_explicit_json_null_not_nan(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_frames(
        run,
        raw=pd.DataFrame([{"instance_id": "i0", "matched_attention_mass": np.nan}]),
        exclusions=pd.DataFrame(),
    )
    assert merge_shards(run) == [{"instance_id": "i0", "matched_attention_mass": None}]


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


ROOT = Path(__file__).parents[1]


def _complete_valid_run(tmp_path):
    cfg = RunConfig.load_files(
        ROOT / "configs/models/fake.yaml",
        ROOT / "configs/datasets/ewt.yaml",
        ROOT / "configs/experiments/attention_entropy.yaml",
    )
    run = tmp_path / "complete-run"
    initialize_run(
        run,
        cfg.to_dict(),
        "dlmrel synthetic validation",
        {"select": "s", "dev": "d", "test": "t"},
    )
    raw = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "seed": seed,
                "timestep": round(progress * 63),
                "normalized_progress": progress,
                "layer": 0,
                "head": 0,
                "entropy": 0.0,
                "entropy_normalized": 0.0,
                "entropy_no_bos": 0.0,
                "bos_sink_mass": 1.0,
            }
            for seed in (42, 43, 44)
            for progress in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
        ]
    )
    write_frames(run, raw=raw, exclusions=pd.DataFrame())
    pd.DataFrame([{"seed": seed, "accuracy": 1.0} for seed in (42, 43, 44)]).to_csv(
        run / "per_seed_metrics.csv", index=False
    )
    pd.DataFrame([{"accuracy": 1.0}]).to_csv(run / "metrics.csv", index=False)
    atomic_json(
        run / "summary.json",
        {
            "schema_version": "dlmrel-run-v1",
            "completion_status": "complete",
            "capabilities": cfg.to_dict()["model"]["capabilities"],
            "n_rows": len(raw),
        },
    )
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["completion_status"] = "complete"
    metadata["final_artifact_hashes"] = final_artifact_hashes(run)
    atomic_json(run / "run_metadata.json", metadata)
    return run


def test_validator_accepts_consistent_complete_artifacts(tmp_path):
    run = _complete_valid_run(tmp_path)
    assert validate_run(run)["valid"] is True


def test_validator_detects_modified_instances_and_duplicate_rows(tmp_path):
    run = _complete_valid_run(tmp_path)
    instances = pd.read_parquet(run / "instances.parquet")
    pd.concat([instances, instances], ignore_index=True).to_parquet(
        run / "instances.parquet", index=False
    )

    result = validate_run(run)

    assert "instances artifact contains duplicate observations" in result["errors"]
    assert "instance Parquet and JSON shards differ" in result["errors"]


def test_validator_detects_infinite_metric_and_nonfinite_json(tmp_path):
    run = _complete_valid_run(tmp_path)
    pd.DataFrame([{"accuracy": float("inf")}]).to_csv(run / "metrics.csv", index=False)
    result = validate_run(run)
    assert "metrics column accuracy contains infinity" in result["errors"]
    assert "final scientific artifact hashes differ" in result["errors"]

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["bad"] = float("nan")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = validate_run(run)
    assert any(error.startswith("summary is invalid: non-finite") for error in result["errors"])

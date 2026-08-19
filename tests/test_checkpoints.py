from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from dlmrel.artifacts import ArtifactError, atomic_json, canonical_hash, initialize_run
from dlmrel.checkpoints import (
    CheckpointIdentity,
    SentenceCheckpointStore,
    atomic_parquet,
)
from dlmrel.selection import create_selection_lock


def _lock(path: Path, *, head: int = 3, created_at: str = "first") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "dlmrel-selection-lock-v1",
                "model_id": "fake",
                "model_revision": "r1",
                "dataset_id": "ewt",
                "layer": 2,
                "head": head,
                "created_at": created_at,
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(lock: Path | None = None, **runtime_overrides) -> dict:
    runtime = {
        "results_root": "first-results",
        "run_id": "first-run",
        "resume": False,
        "dry_run": False,
        "selection_lock": str(lock) if lock else None,
        **runtime_overrides,
    }
    return {
        "schema_version": "test-v1",
        "model": {"id": "fake", "revision": "r1"},
        "dataset": {"id": "ewt", "revision": "d1"},
        "experiment": {
            "id": "head-search",
            "seeds": [42, 43, 44],
            "normalized_progress": [0.0, 0.5, 1.0],
            "scoring": {"receiver_span": "sum"},
        },
        "runtime": runtime,
    }


def _examples(count: int):
    return [SimpleNamespace(sentence_id=f"sentence-{index:04d}") for index in range(count)]


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        stage="select-all-heads",
        seed=42,
        normalized_progress=0.0,
        timestep=0,
    )


def _rows(chunk, _start):
    return pd.DataFrame(
        {
            "sentence_id": [example.sentence_id for example in chunk],
            "seed": 42,
            "timestep": 0,
            "normalized_progress": 0.0,
            "relation": "object_to_verb",
            "layer": 0,
            "head": 0,
            "correct": [int(index % 2 == 0) for index in range(len(chunk))],
        }
    )


def test_normal_run_resumes_with_operational_flags_changed(tmp_path):
    first_lock = _lock(tmp_path / "first-lock.json", created_at="first")
    equivalent_lock = _lock(tmp_path / "moved-lock.json", created_at="second")
    run = tmp_path / "run"
    manifests = {"select": "select-hash", "dev": "dev-hash", "test": "test-hash"}

    initialize_run(run, _config(first_lock), "initial command", manifests)
    initial = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    initialize_run(
        run,
        _config(
            equivalent_lock,
            results_root="different-results",
            run_id="different-run-id",
            resume=True,
            dry_run=True,
        ),
        "resume command",
        manifests,
        resume=True,
    )
    resumed = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))

    assert resumed["scientific_config_hash"] == initial["scientific_config_hash"]
    assert resumed["started_at"] == initial["started_at"]
    assert "last_resumed_at" in resumed


def test_pre_normalization_run_metadata_is_migrated_when_lock_path_moved(tmp_path):
    original_lock = _lock(tmp_path / "old-lock.json", created_at="old-time")
    moved_lock = _lock(tmp_path / "moved-lock.json", created_at="new-time")
    legacy_config = _config(original_lock)
    manifests = {"select": "select-hash", "dev": "dev-hash"}
    run = tmp_path / "legacy-run"
    run.mkdir()
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump(legacy_config, sort_keys=False), encoding="utf-8"
    )
    atomic_json(run / "manifest_refs.json", manifests)
    atomic_json(
        run / "run_metadata.json",
        {
            "schema_version": "dlmrel-run-v1",
            "config_hash": canonical_hash(legacy_config),
            "started_at": "legacy-start",
            "completion_status": "running",
        },
    )
    original_lock.unlink()

    initialize_run(
        run,
        _config(moved_lock, resume=True),
        "resume",
        manifests,
        resume=True,
    )

    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["started_at"] == "legacy-start"
    assert metadata["scientific_config_hash"] == metadata["config_hash"]
    assert metadata["selection_lock_hash"] is not None


def test_resume_rejects_scientific_config_manifest_or_lock_change(tmp_path):
    original_lock = _lock(tmp_path / "lock.json")
    changed_lock = _lock(tmp_path / "changed-lock.json", head=4)
    run = tmp_path / "run"
    manifests = {"select": "select-hash", "dev": "dev-hash"}
    initialize_run(run, _config(original_lock), "initial", manifests)

    scientific_changes = []
    changed_model = deepcopy(_config(original_lock))
    changed_model["model"]["revision"] = "r2"
    scientific_changes.append(changed_model)
    changed_dataset = deepcopy(_config(original_lock))
    changed_dataset["dataset"]["revision"] = "d2"
    scientific_changes.append(changed_dataset)
    changed_seeds = deepcopy(_config(original_lock))
    changed_seeds["experiment"]["seeds"] = [42, 43, 45]
    scientific_changes.append(changed_seeds)
    changed_progress = deepcopy(_config(original_lock))
    changed_progress["experiment"]["normalized_progress"] = [0.0, 1.0]
    scientific_changes.append(changed_progress)
    changed_scoring = deepcopy(_config(original_lock))
    changed_scoring["experiment"]["scoring"]["receiver_span"] = "mean"
    scientific_changes.append(changed_scoring)
    for changed in scientific_changes:
        with pytest.raises(ArtifactError, match="resume config differs"):
            initialize_run(run, changed, "resume", manifests, resume=True)
    with pytest.raises(ArtifactError, match="resume manifests differ"):
        initialize_run(
            run,
            _config(original_lock),
            "resume",
            {"select": "different", "dev": "dev-hash"},
            resume=True,
        )
    with pytest.raises(ArtifactError, match="resume config differs"):
        initialize_run(run, _config(changed_lock), "resume", manifests, resume=True)


def test_interruption_after_chunk_continues_without_duplicates_or_gaps(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, _config(), "initial", {"select": "hash"})
    examples = _examples(650)
    first_calls = []

    def interrupted(chunk, start):
        first_calls.append(start)
        if start == 300:
            raise RuntimeError("simulated interruption")
        return _rows(chunk, start)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        SentenceCheckpointStore(run).run(examples, _identity(), interrupted)
    assert first_calls == [0, 300]

    resumed_calls = []

    def resumed(chunk, start):
        resumed_calls.append(start)
        return _rows(chunk, start)

    frame = SentenceCheckpointStore(run).run(examples, _identity(), resumed)

    assert resumed_calls == [300, 600]
    assert len(frame) == 650
    assert frame["sentence_id"].is_unique
    assert frame["sentence_id"].tolist() == [example.sentence_id for example in examples]


def test_existing_whole_seed_checkpoint_is_reused_and_annotated(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, _config(), "initial", {"select": "hash"})
    examples = _examples(12)
    legacy = run / "checkpoints" / "select-all-heads-seed42-p0.000000-all.parquet"
    expected = _rows(examples, 0)
    atomic_parquet(legacy, expected)

    def must_not_run(_chunk, _start):
        raise AssertionError("legacy whole-seed checkpoint was not reused")

    actual = SentenceCheckpointStore(run).run(
        examples,
        _identity(),
        must_not_run,
        legacy_path=legacy,
    )

    pd.testing.assert_frame_equal(actual, expected)
    metadata = json.loads(legacy.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["legacy_whole_seed"] is True
    assert metadata["sentence_start"] == 0
    assert metadata["sentence_end"] == len(examples)


def test_head_search_reuses_legacy_seed_42_and_43_files(tmp_path, monkeypatch):
    from dlmrel.experiments import shared

    run = tmp_path / "run"
    initialize_run(run, _config(), "initial", {"select": "hash"})
    examples = _examples(12)
    checkpoint_dir = run / "checkpoints"
    for seed in (42, 43):
        frame = _rows(examples, 0)
        frame["seed"] = seed
        atomic_parquet(
            checkpoint_dir / f"select-all-heads-seed{seed}-p0.000000-all.parquet",
            frame,
        )

    calls = []

    def compute(_model, _tokenizer, chunk, _cfg, *, seed, **_kwargs):
        calls.append(seed)
        frame = _rows(chunk, 0)
        frame["seed"] = seed
        return frame

    monkeypatch.setattr(shared, "score_attention_heads", compute)
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            seeds=[42, 43, 44],
            steps=64,
            scoring=SimpleNamespace(primary_visibility="both_masked"),
        )
    )
    combined = shared.score_over_seeds(
        object(),
        object(),
        examples,
        cfg,
        role="select",
        checkpoint_dir=checkpoint_dir,
        stage="select-all-heads",
    )

    assert calls == [44]
    assert combined.groupby("seed").size().to_dict() == {42: 12, 43: 12, 44: 12}


def test_incomplete_temporary_and_corrupt_final_chunks_are_recomputed(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, _config(), "initial", {"select": "hash"})
    examples = _examples(10)
    identity = _identity()
    path = run / "checkpoints" / identity.filename(0, 10)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"incomplete")
    calls = []

    def compute(chunk, start):
        calls.append(start)
        return _rows(chunk, start)

    first = SentenceCheckpointStore(run).run(examples, identity, compute)
    assert calls == [0]
    assert not temporary.exists()

    path.write_bytes(b"corrupt parquet")
    calls.clear()
    second = SentenceCheckpointStore(run).run(examples, identity, compute)
    assert calls == [0]
    pd.testing.assert_frame_equal(first, second)

    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["manifest_hashes"] = {"select": "wrong"}
    atomic_json(metadata_path, metadata)
    calls.clear()
    third = SentenceCheckpointStore(run).run(examples, identity, compute)
    assert calls == [0]
    pd.testing.assert_frame_equal(first, third)


def test_checkpoint_rejects_rows_outside_the_input_sentence_range(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, _config(), "initial", {"select": "hash"})
    examples = _examples(10)

    def wrong_rows(_chunk, _start):
        frame = _rows(examples, 0).iloc[:1].copy()
        frame["sentence_id"] = "not-in-this-chunk"
        return frame

    with pytest.raises(ArtifactError, match="outside its sentence range"):
        SentenceCheckpointStore(run).run(examples, _identity(), wrong_rows)


def test_cached_chunk_with_wrong_sentence_range_is_recomputed(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, _config(), "initial", {"select": "hash"})
    examples = _examples(10)
    store = SentenceCheckpointStore(run)
    store.run(examples, _identity(), _rows)
    path = run / "checkpoints" / _identity().filename(0, 10)
    frame = pd.read_parquet(path)
    frame.loc[0, "sentence_id"] = "foreign-sentence"
    frame.to_parquet(path, index=False)
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["parquet_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(metadata_path, metadata)
    calls = []

    def recompute(chunk, start):
        calls.append(start)
        return _rows(chunk, start)

    restored = SentenceCheckpointStore(run).run(examples, _identity(), recompute)

    assert calls == [0]
    assert set(restored["sentence_id"]) == {example.sentence_id for example in examples}


def test_legacy_checkpoint_with_foreign_sentence_is_not_reused(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, _config(), "initial", {"select": "hash"})
    examples = _examples(4)
    legacy = run / "checkpoints" / "legacy.parquet"
    wrong = _rows(examples, 0)
    wrong.loc[0, "sentence_id"] = "foreign-sentence"
    atomic_parquet(legacy, wrong)
    calls = []

    def recompute(chunk, start):
        calls.append(start)
        return _rows(chunk, start)

    restored = SentenceCheckpointStore(run).run(
        examples, _identity(), recompute, legacy_path=legacy
    )

    assert calls == [0]
    assert "foreign-sentence" not in set(restored["sentence_id"])


def test_resumed_and_uninterrupted_final_outputs_are_identical(tmp_path):
    examples = _examples(650)
    frames = {}
    for name in ("uninterrupted", "resumed"):
        run = tmp_path / name
        initialize_run(run, _config(), "initial", {"select": "hash"})
        store = SentenceCheckpointStore(run)
        if name == "resumed":
            def stop_after_first(chunk, start):
                if start == 300:
                    raise RuntimeError("stop")
                return _rows(chunk, start)

            with pytest.raises(RuntimeError, match="stop"):
                store.run(examples, _identity(), stop_after_first)
            store = SentenceCheckpointStore(run)
        frames[name] = store.run(examples, _identity(), _rows)
        _write_final_outputs(run, frames[name])

    columns = ["sentence_id", "seed", "layer", "head"]
    for filename in ("instances.parquet",):
        left = pd.read_parquet(tmp_path / "uninterrupted" / filename).sort_values(columns)
        right = pd.read_parquet(tmp_path / "resumed" / filename).sort_values(columns)
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
    for filename in ("per_seed_metrics.csv", "metrics.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "uninterrupted" / filename),
            pd.read_csv(tmp_path / "resumed" / filename),
        )
    for filename in ("selection_lock.json", "summary.json"):
        left = json.loads((tmp_path / "uninterrupted" / filename).read_text(encoding="utf-8"))
        right = json.loads((tmp_path / "resumed" / filename).read_text(encoding="utf-8"))
        assert left == right


def _write_final_outputs(run: Path, frame: pd.DataFrame) -> None:
    ordered = frame.sort_values(
        ["sentence_id", "seed", "relation", "layer", "head"]
    ).reset_index(drop=True)
    ordered.to_parquet(run / "instances.parquet", index=False)
    per_seed = ordered.groupby(["seed", "layer", "head"], as_index=False).agg(
        accuracy=("correct", "mean"), n_instances=("correct", "size")
    )
    per_seed.to_csv(run / "per_seed_metrics.csv", index=False)
    per_seed.groupby(["layer", "head"], as_index=False).agg(
        accuracy=("accuracy", "mean"), n_seeds=("seed", "nunique")
    ).to_csv(run / "metrics.csv", index=False)
    scores = ordered.groupby(["relation", "layer", "head"], as_index=False).agg(
        accuracy=("correct", "mean"), n_total=("correct", "size")
    )
    lock, _, _ = create_selection_lock(
        scores,
        scores,
        relation="object_to_verb",
        top_k=1,
        track="confirmatory_ewt",
        model_id="fake",
        model_revision="r1",
        dataset_id="ewt",
        config_hash="same",
        select_manifest_hash="select",
        dev_manifest_hash="dev",
        created_at="fixed-start",
    )
    lock.write_once(run / "selection_lock.json")
    atomic_json(
        run / "summary.json",
        {
            "completion_status": "complete",
            "n_rows": len(ordered),
            "selected_layer": lock.layer,
            "selected_head": lock.head,
        },
    )

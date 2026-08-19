from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from dlmrel import cli
from dlmrel import head_search_recovery as recovery
from dlmrel.artifacts import ArtifactError, atomic_json, initialize_run
from dlmrel.checkpoints import SentenceCheckpointStore
from dlmrel.cli import main
from dlmrel.config import RELATION_NAMES, RunConfig
from dlmrel.fake_run import _all_head_rows, run_fake

ROOT = Path(__file__).parents[1]
MANIFESTS = {"select": "select-fixture", "dev": "dev-fixture", "test": "test-fixture"}


@pytest.fixture(scope="module")
def completed_fake_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("head-search-recovery-reference")
    run = root / "reference"
    cfg = RunConfig.load_files(
        ROOT / "configs/models/fake.yaml",
        ROOT / "configs/datasets/ewt.yaml",
        ROOT / "configs/experiments/head_search.yaml",
    )
    initialize_run(run, cfg.to_dict(), "fake reference", MANIFESTS)
    run_fake(cfg, run)
    return run


def _copy_run(source: Path, target: Path, *, keep_all_head: bool) -> Path:
    shutil.copytree(source, target)
    for name in (
        "metrics.csv",
        "per_seed_metrics.csv",
        "structural_slices.csv",
        "selection_permutation.json",
        "selection_permutation_results.csv",
        "summary.json",
        "validation.json",
        "relation_locks.resolved.json",
    ):
        (target / name).unlink(missing_ok=True)
    for directory in ("permutations", "permutation-checkpoints"):
        shutil.rmtree(target / directory, ignore_errors=True)
    if not keep_all_head:
        (target / recovery.TEST_ALL_HEAD_EVIDENCE).unlink(missing_ok=True)
        (target / recovery.TEST_ALL_HEAD_METADATA).unlink(missing_ok=True)
    metadata_path = target / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completion_status"] = "running"
    for field in ("ended_at", "final_artifact_hashes", "recovery_test_grid"):
        metadata.pop(field, None)
    atomic_json(metadata_path, metadata)
    return target


def _cfg(run: Path) -> RunConfig:
    return recovery.load_saved_head_search_config(run)


def _mock_test_examples(monkeypatch, cfg):
    all_rows = _all_head_rows(cfg, "test")
    sentence_ids = list(dict.fromkeys(all_rows["sentence_id"].astype(str)))
    examples = [SimpleNamespace(sentence_id=sentence_id) for sentence_id in sentence_ids]
    roles = []
    calls = []

    def load_examples(_cfg, _tokenizer, role):
        roles.append(role)
        if role != "test":
            raise AssertionError("recovery attempted to load select/dev examples")
        return examples, pd.DataFrame()

    def score(_model, _tokenizer, chunk, _cfg, *, role, seed, **_kwargs):
        roles.append(f"score:{role}")
        ids = {example.sentence_id for example in chunk}
        calls.append((seed, tuple(sorted(ids))))
        return all_rows[
            all_rows["sentence_id"].isin(ids) & all_rows["seed"].eq(seed)
        ].copy()

    monkeypatch.setattr(recovery, "load_manifest_examples", load_examples)
    monkeypatch.setattr(recovery, "score_attention_heads", score)
    return all_rows, examples, roles, calls, score


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_locked_head_evidence_is_reusable_but_permutation_incomplete(
    completed_fake_run, tmp_path
):
    run = _copy_run(completed_fake_run, tmp_path / "legacy", keep_all_head=False)
    status = recovery.recovery_status(run, _cfg(run))

    assert status["legacy_locked_head_evidence"] == "valid_and_reusable"
    assert status["locked_head_evidence_sufficient_for_metrics"] is True
    assert status["locked_head_evidence_sufficient_for_selection_aware_permutation"] is False
    assert status["missing_test_grid_required"] is True
    with pytest.raises(ArtifactError, match="locked-head-only evidence"):
        recovery.complete_cpu_finalization(_cfg(run), run, n_permutations=10)


def test_missing_grid_runs_test_only_and_resumes_without_duplicates(
    completed_fake_run, tmp_path, monkeypatch
):
    run = _copy_run(completed_fake_run, tmp_path / "grid-resume", keep_all_head=False)
    cfg = _cfg(run)
    expected, _examples, roles, calls, ordinary_score = _mock_test_examples(monkeypatch, cfg)
    immutable_hashes = {
        name: _sha256(run / name)
        for name in ("select_instances.parquet", "dev_instances.parquet")
    }
    legacy = run / "checkpoints" / "test-locked-head__seed-42__legacy.parquet"
    legacy.write_bytes(b"legacy locked-head checkpoint remains separate")
    legacy_hash = _sha256(legacy)

    real_store = SentenceCheckpointStore
    monkeypatch.setattr(
        recovery,
        "SentenceCheckpointStore",
        lambda path: real_store(path, chunk_size=20),
    )
    interrupted = False

    def interrupt_second_chunk(*args, **kwargs):
        nonlocal interrupted
        if len(calls) == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated GPU interruption")
        return ordinary_score(*args, **kwargs)

    monkeypatch.setattr(recovery, "score_attention_heads", interrupt_second_chunk)
    with pytest.raises(RuntimeError, match="simulated GPU interruption"):
        recovery.score_missing_test_grid(object(), object(), cfg, run)
    assert len(calls) == 1
    assert not (run / recovery.TEST_ALL_HEAD_METADATA).exists()

    monkeypatch.setattr(recovery, "score_attention_heads", ordinary_score)
    report, locked, exclusions = recovery.score_missing_test_grid(
        object(),
        object(),
        cfg,
        run,
        model_metadata={"checkpoint": "fake", "revision": "local-v1"},
    )

    actual = pd.read_parquet(run / recovery.TEST_ALL_HEAD_EVIDENCE)
    identity = ["sentence_id", "instance_id", "seed", "layer", "head"]
    assert report["select_forward_calls"] == 0
    assert report["dev_forward_calls"] == 0
    assert report["stage"] == recovery.TEST_ALL_HEAD_STAGE
    assert report["estimated_test_forward_calls"] == 180
    assert locked.empty and exclusions.empty
    assert not actual.duplicated(identity).any()
    assert len(actual) == len(expected)
    expected_narrow = expected.loc[:, recovery.PERMUTATION_COLUMNS].sort_values(
        list(recovery.PERMUTATION_SORT), kind="mergesort"
    ).reset_index(drop=True)
    actual_narrow = actual.sort_values(
        list(recovery.PERMUTATION_SORT), kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_narrow, expected_narrow, check_dtype=False)
    assert len(calls) == 9  # one completed chunk was reused; eight chunks were recomputed
    assert set(roles) == {"test", "score:test"}
    assert _sha256(legacy) == legacy_hash
    checkpoint_names = [path.name for path in (run / "checkpoints").glob("*.parquet")]
    assert any(name.startswith(recovery.TEST_ALL_HEAD_STAGE) for name in checkpoint_names)
    assert any(name.startswith("test-locked-head") for name in checkpoint_names)
    assert {
        name: _sha256(run / name)
        for name in ("select_instances.parquet", "dev_instances.parquet")
    } == immutable_hashes


def test_cpu_finalizer_never_loads_a_model(completed_fake_run, tmp_path, monkeypatch, capsys):
    run = _copy_run(completed_fake_run, tmp_path / "cpu-only", keep_all_head=True)
    monkeypatch.setattr(
        cli,
        "load_adapter",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("model must not load")),
    )

    assert main(["finalize-head-search", "--run-dir", str(run)]) == 0
    assert '"valid": true' in capsys.readouterr().out.lower()


def test_completed_test_grid_is_idempotent_and_skips_model_load(
    completed_fake_run, tmp_path, monkeypatch, capsys
):
    run = _copy_run(completed_fake_run, tmp_path / "grid-idempotent", keep_all_head=True)
    monkeypatch.setattr(
        cli,
        "load_adapter",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("model must not load")),
    )

    assert main(["recover-head-search-test-grid", "--run-dir", str(run)]) == 0
    output = capsys.readouterr().out.lower()
    assert '"model_loaded": false' in output
    assert '"all_head_test_evidence": "complete"' in output


def test_relation_by_relation_finalizer_matches_uninterrupted_fake_run(
    completed_fake_run, tmp_path, monkeypatch
):
    run = _copy_run(completed_fake_run, tmp_path / "relation-wise", keep_all_head=True)
    cfg = _cfg(run)
    reads = []
    original = recovery._read_relation_parquet

    def recording_read(path, relation, columns):
        reads.append((Path(path).name, relation, tuple(columns)))
        return original(path, relation, columns)

    monkeypatch.setattr(recovery, "_read_relation_parquet", recording_read)
    result = recovery.complete_cpu_finalization(cfg, run, n_permutations=10, checkpoint_interval=5)

    assert result["validation"]["valid"] is True
    assert len(reads) == len(RELATION_NAMES) * 5
    for relation in RELATION_NAMES:
        relation_reads = [item for item in reads if item[1] == relation]
        assert len(relation_reads) == 5
        assert all(
            set(item[2]).issubset(
                set(recovery.PERMUTATION_COLUMNS) | set(recovery.LOCKED_METRIC_COLUMNS)
            )
            for item in relation_reads
        )
    _assert_final_science_equal(completed_fake_run, run)


def test_partial_cpu_finalization_resumes_and_matches_reference(
    completed_fake_run, tmp_path, monkeypatch
):
    run = _copy_run(completed_fake_run, tmp_path / "partial-finalizer", keep_all_head=True)
    cfg = _cfg(run)
    original = recovery.selection_aware_permutation
    interrupted = False

    def interrupt_after_primary(*args, relation, **kwargs):
        nonlocal interrupted
        if relation == RELATION_NAMES[1] and not interrupted:
            interrupted = True
            raise RuntimeError("simulated CPU interruption")
        return original(*args, relation=relation, **kwargs)

    monkeypatch.setattr(recovery, "selection_aware_permutation", interrupt_after_primary)
    with pytest.raises(RuntimeError, match="simulated CPU interruption"):
        recovery.complete_cpu_finalization(cfg, run, n_permutations=10, checkpoint_interval=5)
    primary_checkpoint = run / "permutation-checkpoints" / f"{RELATION_NAMES[0]}.json"
    saved = json.loads(primary_checkpoint.read_text(encoding="utf-8"))
    assert saved["completion_status"] == "complete"

    monkeypatch.setattr(recovery, "selection_aware_permutation", original)
    resumed = recovery.complete_cpu_finalization(cfg, run, n_permutations=10, checkpoint_interval=5)
    assert resumed["validation"]["valid"] is True
    instances = pd.read_parquet(run / "instances.parquet")
    assert not instances.duplicated(["sentence_id", "instance_id", "seed", "relation"]).any()
    _assert_final_science_equal(completed_fake_run, run)


@pytest.mark.parametrize("corruption", ["evidence_hash", "scientific_config"])
def test_recovery_rejects_corrupt_or_incompatible_evidence(
    completed_fake_run, tmp_path, corruption
):
    run = _copy_run(
        completed_fake_run,
        tmp_path / f"corrupt-{corruption}",
        keep_all_head=True,
    )
    if corruption == "evidence_hash":
        metadata_path = run / recovery.TEST_ALL_HEAD_METADATA
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["parquet_sha256"] = "0" * 64
        atomic_json(metadata_path, metadata)
        expected = "evidence hash mismatch"
    else:
        config_path = run / "config.resolved.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["dataset"]["revision"] = "scientifically-different-revision"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        expected = "scientific configuration differs"

    with pytest.raises(ArtifactError, match=expected):
        recovery.complete_cpu_finalization(_cfg(run), run, n_permutations=10)


def _assert_final_science_equal(reference: Path, recovered: Path) -> None:
    for name in ("instances.parquet",):
        left = pd.read_parquet(reference / name).sort_values(
            ["sentence_id", "instance_id", "seed", "relation", "layer", "head"],
            kind="mergesort",
        ).reset_index(drop=True)
        right = pd.read_parquet(recovered / name).sort_values(
            ["sentence_id", "instance_id", "seed", "relation", "layer", "head"],
            kind="mergesort",
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
    for name in ("metrics.csv", "per_seed_metrics.csv", "selection_permutation_results.csv"):
        left = pd.read_csv(reference / name).sort_values(
            list(pd.read_csv(reference / name).columns), kind="mergesort"
        ).reset_index(drop=True)
        right = pd.read_csv(recovered / name).sort_values(
            list(pd.read_csv(recovered / name).columns), kind="mergesort"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
    assert json.loads((reference / "selection_permutation.json").read_text(encoding="utf-8")) == json.loads(
        (recovered / "selection_permutation.json").read_text(encoding="utf-8")
    )
    assert json.loads((reference / "summary.json").read_text(encoding="utf-8")) == json.loads(
        (recovered / "summary.json").read_text(encoding="utf-8")
    )
    for relation in RELATION_NAMES:
        left_lock = reference / "relation-selection" / "locks" / f"{relation}.json"
        right_lock = recovered / "relation-selection" / "locks" / f"{relation}.json"
        assert left_lock.read_bytes() == right_lock.read_bytes()
    for directory in (reference, recovered):
        validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
        assert validation["valid"] is True
        assert validation["errors"] == []

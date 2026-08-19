from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from runners.modal_app import (
    RUN_RESULT_SCHEMA,
    RUN_SPEC_SCHEMA,
    ProcessOutcome,
    RunResult,
    RunSpec,
    SpecError,
    _sanitized_environment,
    build_cli_args,
    classify_failure,
    execute_spec,
    normalized_failure_signature,
    redact_secrets,
    route_for_spec,
    stream_process,
)

COMMIT = "e0f80baf4db57881849f926a0dae631260ffe558"
SCIENCE_HASH = "a" * 64
MANIFESTS = {"select": "b" * 64, "dev": "c" * 64, "test": "d" * 64}


def make_spec(**changes) -> RunSpec:
    values = {
        "schema_version": RUN_SPEC_SCHEMA,
        "git_commit": COMMIT,
        "operation": "experiment-run",
        "model_config": "configs/models/fake.yaml",
        "dataset_config": "configs/datasets/ewt.yaml",
        "experiment_config": "configs/experiments/head_search.yaml",
        "track": "confirmatory_ewt",
        "run_id": "safe-run-v1",
        "result_namespace": "scratch",
        "selection_lock_source": None,
        "resume": False,
        "dry_run": True,
        "resource_profile": "cpu-small",
        "timeout_seconds": 3_600,
        "attempt": 1,
        "maximum_repair_attempts": 3,
        "expected_scientific_config_hash": None,
        "expected_manifest_hashes": {},
        "source_result_identity": None,
        "max_estimated_cost_usd": 20.0,
    }
    values.update(changes)
    return RunSpec(**values)


def make_result(**changes) -> RunResult:
    values = {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": "success",
        "run_call_id": "fc-test",
        "attempt": 1,
        "git_commit": COMMIT,
        "operation": "cpu-test",
        "sanitized_cli_args": ["python", "-m", "pytest", "-q"],
        "scientific_config_hash": None,
        "manifest_hashes": {},
        "selection_lock_hash": None,
        "started_at": "2026-08-18T00:00:00+00:00",
        "ended_at": "2026-08-18T00:00:01+00:00",
        "duration_seconds": 1.0,
        "modal_reference": "modal-function-call:fc-test",
        "exit_code": 0,
        "failure_category": None,
        "exception_class": None,
        "failure_signature": None,
        "stdout_tail": "passed",
        "stderr_tail": "",
        "full_log_artifact": "scratch/safe-run-v1/attempt-1.log",
        "last_completed_checkpoint": None,
        "validation_result": None,
        "result_directory": None,
        "scientific_artifacts_written": False,
        "recommended_next_action": "done",
    }
    values.update(changes)
    return RunResult(**values)


def test_run_spec_is_strict_serializable_and_rejects_unknown_or_missing_fields():
    spec = make_spec()
    assert RunSpec.from_json(spec.to_json()) == spec
    extra = spec.to_dict() | {"command": "rm -rf /"}
    with pytest.raises(SpecError, match="unknown"):
        RunSpec.from_dict(extra)
    missing = spec.to_dict()
    missing.pop("git_commit")
    with pytest.raises(SpecError, match="missing"):
        RunSpec.from_dict(missing)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"run_id": "bad;touch-pwned"}, "unsafe"),
        ({"git_commit": "main"}, "40-character"),
        ({"model_config": "../model.yaml"}, "allowlist"),
        ({"source_result_identity": "../official/run"}, "allowed root"),
        ({"selection_lock_source": "/tmp/lock.json"}, "allowed root"),
        ({"result_namespace": "official"}, "scratch"),
        ({"timeout_seconds": 100_000}, "timeout"),
        ({"max_estimated_cost_usd": 0.01}, "cost ceiling"),
    ],
)
def test_run_spec_rejects_shell_injection_path_escape_and_ceiling_changes(change, message):
    with pytest.raises(SpecError, match=message):
        make_spec(**change)


def test_safe_cli_mapping_and_cpu_gpu_routes():
    fake = make_spec()
    assert route_for_spec(fake) == "cpu"
    args = build_cli_args(fake, mount=Path("/safe/results"))
    assert args[:2] == ["dlmrel", "run"]
    assert "--dry-run" in args
    assert all(";" not in value for value in args)

    dream = make_spec(
        operation="model-smoke-test",
        model_config="configs/models/dream_7b.yaml",
        dry_run=False,
        resource_profile="dream-a100-80gb",
        timeout_seconds=7_200,
    )
    assert route_for_spec(dream) == "dream"
    diffullama = replace(
        dream,
        model_config="configs/models/diffullama_7b.yaml",
        resource_profile="diffullama-a100-80gb",
    )
    assert route_for_spec(diffullama) == "diffullama"


def test_time_curve_requires_isolated_lock():
    with pytest.raises(SpecError, match="selection-lock"):
        make_spec(
            experiment_config="configs/experiments/time_curve.yaml",
            dry_run=False,
            model_config="configs/models/dream_7b.yaml",
            resource_profile="dream-a100-80gb",
        )


def test_run_result_round_trip_is_strict():
    result = make_result()
    assert RunResult.from_dict(json.loads(result.to_json())) == result
    invalid = result.to_dict() | {"secret_environment": {}}
    with pytest.raises(SpecError, match="unknown"):
        RunResult.from_dict(invalid)
    malformed = result.to_dict() | {"attempt": "1"}
    with pytest.raises(SpecError, match="attempt"):
        RunResult.from_dict(malformed)
    inconsistent = result.to_dict() | {
        "status": "failed",
        "failure_category": None,
        "failure_signature": None,
    }
    with pytest.raises(SpecError, match="requires classified"):
        RunResult.from_dict(inconsistent)


def test_secret_redaction_and_signature_are_stable_but_distinguish_failures():
    raw = (
        "OPENAI_API_KEY=sk-proj-abcdefghijk hf_abcdefghijk "
        "Bearer ghp_abcdefghijk"
    )
    assert "abcdefghijk" not in redact_secrets(raw)
    first = normalized_failure_signature(
        exception_class="ValueError",
        message="2026-08-18T01:02:03Z /tmp/job-123/x.py 100/200 boom",
        stage="test",
        exit_code=1,
    )
    same = normalized_failure_signature(
        exception_class="ValueError",
        message="2026-08-19T04:05:06Z /tmp/job-999/y.py 150/300 boom",
        stage="test",
        exit_code=1,
    )
    distinct = normalized_failure_signature(
        exception_class="TypeError", message="different", stage="test", exit_code=1
    )
    assert first == same
    assert first != distinct


@pytest.mark.parametrize(
    ("message", "code", "category"),
    [
        ("temporary failure: HTTP 503", 1, "transient_infrastructure"),
        ("fatal: could not resolve host", 1, "transient_infrastructure"),
        ("CUDA out of memory", 137, "resource_exhaustion"),
        ("operation exceeded timeout of 3600 seconds", 124, "resource_exhaustion"),
        ("ModuleNotFoundError: transformers", 1, "dependency_environment_incompatibility"),
        ("checkpoint mismatch", 1, "artifact_checkpoint_mismatch"),
        ("scientific config differs", 2, "scientific_configuration_mismatch"),
        ("manifest checksum failed", 2, "data_revision_checksum_failure"),
        ("Traceback ValueError", 1, "deterministic_implementation_failure"),
    ],
)
def test_failure_classification(message, code, category):
    assert classify_failure(message, exit_code=code) == category


def test_subprocess_runner_always_uses_shell_false(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        stdout = io.StringIO("ok\n")
        stderr = io.StringIO("")

        def wait(self, timeout=None):
            return 0

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("runners.modal_app.subprocess.Popen", fake_popen)
    result = stream_process(
        ["dlmrel", "validate"],
        cwd=tmp_path,
        env={},
        timeout=10,
        log_path=tmp_path / "run.log",
    )
    assert result.exit_code == 0
    assert captured["shell"] is False
    assert captured["args"] == ["dlmrel", "validate"]


def test_child_environment_excludes_all_controller_and_model_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk_secret")
    monkeypatch.setenv("GH_TOKEN", "ghp_secret")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "modal_secret")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    env = _sanitized_environment(tmp_path)
    assert not {"OPENAI_API_KEY", "GH_TOKEN", "MODAL_TOKEN_SECRET", "HF_TOKEN"}.intersection(env)
    assert env["HF_HOME"].startswith(str(tmp_path))


def test_fake_adapter_wrapper_execution_uses_injected_local_stub_and_collects_artifacts(tmp_path):
    spec = make_spec()
    results = tmp_path / "results"
    logs = tmp_path / "logs"
    cache = tmp_path / "cache"
    run_dir = spec.run_directory(results)

    def runner(args, **kwargs):
        assert args[0:2] == ["dlmrel", "run"]
        assert kwargs["cwd"] == Path.cwd()
        run_dir.mkdir(parents=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"scientific_config_hash": SCIENCE_HASH}), encoding="utf-8"
        )
        (run_dir / "manifest_refs.json").write_text(json.dumps(MANIFESTS), encoding="utf-8")
        (run_dir / "validation.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
        return ProcessOutcome(0, "fake completed", "")

    result = execute_spec(
        spec,
        repository=Path.cwd(),
        results_mount=results,
        cache_mount=cache,
        logs_mount=logs,
        process_runner=runner,
        call_id="local-stub",
    )
    assert result.status == "success"
    assert result.scientific_config_hash == SCIENCE_HASH
    assert result.manifest_hashes == MANIFESTS
    assert result.validation_result == {"valid": True}
    assert result.scientific_artifacts_written


def test_completed_validated_run_is_never_overwritten(tmp_path):
    spec = make_spec()
    run_dir = spec.run_directory(tmp_path / "results")
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"completion_status": "complete"}), encoding="utf-8"
    )
    (run_dir / "validation.json").write_text(json.dumps({"valid": True}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("completed run must not execute")

    result = execute_spec(
        spec,
        repository=Path.cwd(),
        results_mount=tmp_path / "results",
        cache_mount=tmp_path / "cache",
        logs_mount=tmp_path / "logs",
        process_runner=forbidden,
    )
    assert result.status == "stopped"
    assert result.failure_category == "protected_scientific_behavior"


def test_run_identity_is_attempt_and_commit_independent_for_concurrency():
    first = make_spec()
    second = replace(first, attempt=2, git_commit="1" * 40)
    assert first.run_identity == second.run_identity
    assert first.results_root() != second.results_root()
    validation = replace(
        first,
        operation="validation",
        dry_run=False,
        resource_profile="cpu-small",
        resume=True,
        expected_scientific_config_hash=SCIENCE_HASH,
        expected_manifest_hashes=MANIFESTS,
        source_result_identity="scratch/source/run",
    )
    assert first.run_identity == validation.run_identity
    assert first.results_root() != validation.results_root()

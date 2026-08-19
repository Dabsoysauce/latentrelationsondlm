"""Thin, allowlisted Modal execution wrapper for the existing ``dlmrel`` CLI.

The pure-Python contract and execution helpers deliberately work without the
Modal package.  Modal objects are defined only when the optional SDK is
installed, which keeps local policy tests credential-free.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:  # The local dry-run and tests intentionally do not require Modal.
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised by import in the test environment
    modal = None


RUN_SPEC_SCHEMA = "dlmrel-modal-run-spec-v1"
RUN_RESULT_SCHEMA = "dlmrel-modal-run-result-v1"
REPOSITORY_URL = "https://github.com/Dabsoysauce/latentrelationsondlm.git"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPOSITORY = Path("/workspace/repository")
RESULTS_MOUNT = Path("/mnt/results")
CACHE_MOUNT = Path("/mnt/model-cache")
LOGS_MOUNT = Path("/mnt/attempt-logs")
TAIL_LIMIT = 16_384

OPERATIONS = {
    "cpu-test",
    "model-smoke-test",
    "experiment-run",
    "missing-test-grid-recovery",
    "cpu-finalization",
    "validation",
}
MODEL_CONFIGS = {
    "configs/models/fake.yaml": ("fake", "fake"),
    "configs/models/dream_7b.yaml": ("dream_7b", "dream"),
    "configs/models/diffullama_7b.yaml": ("diffullama_7b", "diffullama"),
}
DATASET_CONFIGS = {
    "configs/datasets/ewt.yaml": "ewt",
    "configs/datasets/de_gsd.yaml": "de_gsd",
    "configs/datasets/ja_gsd.yaml": "ja_gsd",
}
EXPERIMENT_CONFIGS = {
    "configs/experiments/head_search.yaml": ("confirmatory_head_search", "confirmatory_ewt"),
    "configs/experiments/time_curve.yaml": ("confirmatory_time_curve", "confirmatory_ewt"),
    "configs/experiments/attention_entropy.yaml": (
        "attention_entropy_over_time",
        "exploratory_extensions",
    ),
    "configs/experiments/logit_lens.yaml": ("rank_logit_lens", "exploratory_extensions"),
    "configs/experiments/pos_probe.yaml": ("masked_pos_probe", "exploratory_extensions"),
    "configs/experiments/external_transfer.yaml": (
        "ewt_locked_transfer",
        "external_treebank_transfer",
    ),
}
TRACKS = {value[1] for value in EXPERIMENT_CONFIGS.values()}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
FAILURE_CATEGORIES = {
    "protected_scientific_behavior",
    "scientific_configuration_mismatch",
    "data_revision_checksum_failure",
    "artifact_checkpoint_mismatch",
    "resource_exhaustion",
    "dependency_environment_incompatibility",
    "transient_infrastructure",
    "deterministic_implementation_failure",
    "unknown",
}


class SpecError(ValueError):
    """A submitted run could escape the allowlisted operational surface."""


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    route: str
    cpu: float
    memory_mib: int
    gpu: str | None
    maximum_timeout_seconds: int
    estimated_rate_usd_per_second: float

    def estimated_cost(self, timeout_seconds: int) -> float:
        return round(self.estimated_rate_usd_per_second * timeout_seconds, 2)


# Rates are an operational guardrail snapshot from Modal's public pricing page
# on 2026-08-18.  They are deliberately not scientific configuration.
RESOURCE_PROFILES = {
    "cpu-small": ResourceProfile(
        "cpu-small", "cpu", 4.0, 16_384, None, 2 * 60 * 60, 0.00008792
    ),
    "cpu-large": ResourceProfile(
        "cpu-large", "cpu", 8.0, 65_536, None, 12 * 60 * 60, 0.00024688
    ),
    "dream-a100-80gb": ResourceProfile(
        "dream-a100-80gb", "dream", 4.0, 32_768, "A100-80GB", 6 * 60 * 60, 0.00081744
    ),
    "diffullama-a100-80gb": ResourceProfile(
        "diffullama-a100-80gb",
        "diffullama",
        4.0,
        32_768,
        "A100-80GB",
        6 * 60 * 60,
        0.00081744,
    ),
}


def _strict_keys(value: dict[str, Any], cls: type, label: str) -> None:
    expected = {item.name for item in fields(cls)}
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise SpecError(f"{label} fields differ; missing={missing}, unknown={unknown}")


def _safe_relative_path(value: str, *, label: str, allow_empty: bool = False) -> str:
    if not value and allow_empty:
        return value
    if not value or "\\" in value:
        raise SpecError(f"{label} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SpecError(f"{label} must remain inside its allowed root")
    if any(not SAFE_ID.fullmatch(part) for part in path.parts):
        raise SpecError(f"{label} contains unsafe characters")
    return path.as_posix()


@dataclass(frozen=True)
class RunSpec:
    schema_version: str
    git_commit: str
    operation: str
    model_config: str | None
    dataset_config: str | None
    experiment_config: str | None
    track: str
    run_id: str
    result_namespace: str
    selection_lock_source: str | None
    resume: bool
    dry_run: bool
    resource_profile: str
    timeout_seconds: int
    attempt: int
    maximum_repair_attempts: int
    expected_scientific_config_hash: str | None
    expected_manifest_hashes: dict[str, str] = field(default_factory=dict)
    source_result_identity: str | None = None
    max_estimated_cost_usd: float = 20.0

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunSpec:
        if not isinstance(value, dict):
            raise SpecError("RunSpec must be a JSON object")
        _strict_keys(value, cls, "RunSpec")
        return cls(**value)

    @classmethod
    def from_json(cls, value: str) -> RunSpec:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise SpecError("RunSpec is not valid JSON") from error
        return cls.from_dict(decoded)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def model_id(self) -> str:
        return MODEL_CONFIGS[self.model_config][0] if self.model_config else "none"

    @property
    def model_family(self) -> str:
        return MODEL_CONFIGS[self.model_config][1] if self.model_config else "none"

    @property
    def dataset_id(self) -> str:
        return DATASET_CONFIGS[self.dataset_config] if self.dataset_config else "none"

    @property
    def experiment_id(self) -> str:
        return EXPERIMENT_CONFIGS[self.experiment_config][0] if self.experiment_config else "none"

    @property
    def profile(self) -> ResourceProfile:
        return RESOURCE_PROFILES[self.resource_profile]

    @property
    def run_identity(self) -> str:
        value = {
            "track": self.track,
            "model": self.model_id,
            "dataset": self.dataset_id,
            "experiment": self.experiment_id,
            "run_id": self.run_id,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def results_root(self, mount: Path = RESULTS_MOUNT) -> Path:
        return (
            mount
            / self.result_namespace
            / self.run_id
            / self.operation
            / f"attempt-{self.attempt}"
        )

    def run_directory(self, mount: Path = RESULTS_MOUNT) -> Path:
        return (
            self.results_root(mount)
            / self.track
            / self.model_id
            / self.dataset_id
            / self.experiment_id
            / self.run_id
        )

    def validate(self) -> None:
        if self.schema_version != RUN_SPEC_SCHEMA:
            raise SpecError(f"schema_version must be {RUN_SPEC_SCHEMA}")
        if not isinstance(self.git_commit, str) or not COMMIT_SHA.fullmatch(self.git_commit):
            raise SpecError("git_commit must be a full lowercase 40-character SHA")
        if self.operation not in OPERATIONS:
            raise SpecError(f"unsupported operation: {self.operation!r}")
        if not SAFE_ID.fullmatch(self.run_id):
            raise SpecError("run_id contains unsafe characters")
        if self.track not in TRACKS:
            raise SpecError("track is not active")
        _safe_relative_path(self.result_namespace, label="result_namespace")
        if PurePosixPath(self.result_namespace).parts[0] != "scratch":
            raise SpecError("result_namespace must be inside the isolated scratch namespace")
        for value, allowed, label in (
            (self.model_config, MODEL_CONFIGS, "model_config"),
            (self.dataset_config, DATASET_CONFIGS, "dataset_config"),
            (self.experiment_config, EXPERIMENT_CONFIGS, "experiment_config"),
        ):
            if value is not None and value not in allowed:
                raise SpecError(f"{label} is outside the active allowlist")
        needs_configs = self.operation != "cpu-test"
        if needs_configs and not all(
            (self.model_config, self.dataset_config, self.experiment_config)
        ):
            raise SpecError(f"{self.operation} requires model, dataset, and experiment configs")
        if self.experiment_config and EXPERIMENT_CONFIGS[self.experiment_config][1] != self.track:
            raise SpecError("track does not match the selected experiment")
        if self.experiment_config == "configs/experiments/external_transfer.yaml":
            if self.dataset_config == "configs/datasets/ewt.yaml":
                raise SpecError("external transfer requires German or Japanese data")
        elif self.dataset_config and self.dataset_config != "configs/datasets/ewt.yaml":
            raise SpecError("non-transfer experiments require EWT")
        if self.selection_lock_source is not None:
            _safe_relative_path(self.selection_lock_source, label="selection_lock_source")
            if self.selection_lock_source.startswith("official"):
                raise SpecError("official result storage cannot be mounted read-write")
        lock_required = self.experiment_config in {
            "configs/experiments/time_curve.yaml",
            "configs/experiments/external_transfer.yaml",
        }
        if self.operation == "experiment-run" and lock_required and not self.selection_lock_source:
            raise SpecError("this experiment requires an isolated selection-lock source")
        if self.source_result_identity is not None:
            _safe_relative_path(self.source_result_identity, label="source_result_identity")
            if self.source_result_identity.startswith("official"):
                raise SpecError("recovery must use a staged copy, not official results")
        if self.operation in {"missing-test-grid-recovery", "cpu-finalization", "validation"}:
            if not self.source_result_identity:
                raise SpecError(f"{self.operation} requires source_result_identity")
        if self.operation == "missing-test-grid-recovery":
            if (
                self.model_family != "dream"
                or self.dataset_id != "ewt"
                or self.experiment_config != "configs/experiments/head_search.yaml"
            ):
                raise SpecError("missing-test-grid recovery is only the frozen Dream EWT head search")
            if self.dry_run:
                raise SpecError("the recovery CLI has no dry-run mode")
        if not isinstance(self.resume, bool) or not isinstance(self.dry_run, bool):
            raise SpecError("resume and dry_run must be booleans")
        if self.resume or self.operation in {
            "missing-test-grid-recovery",
            "cpu-finalization",
            "validation",
        }:
            if not self.expected_scientific_config_hash or not self.expected_manifest_hashes:
                raise SpecError("resume/recovery requires expected scientific and manifest hashes")
        if self.expected_scientific_config_hash is not None and not HASH.fullmatch(
            self.expected_scientific_config_hash
        ):
            raise SpecError("expected scientific configuration hash is invalid")
        if not isinstance(self.expected_manifest_hashes, dict) or any(
            not isinstance(key, str)
            or not SAFE_ID.fullmatch(key)
            or not isinstance(value, str)
            or not HASH.fullmatch(value)
            for key, value in self.expected_manifest_hashes.items()
        ):
            raise SpecError("expected manifest hashes are invalid")
        if self.expected_manifest_hashes and set(self.expected_manifest_hashes) != {
            "select",
            "dev",
            "test",
        }:
            raise SpecError("expected manifest hashes must contain exactly select/dev/test")
        if self.resource_profile not in RESOURCE_PROFILES:
            raise SpecError("resource_profile is not allowlisted")
        expected_route = route_for_spec(self)
        if self.profile.route != expected_route:
            raise SpecError(
                f"resource profile route {self.profile.route!r} does not match {expected_route!r}"
            )
        if not isinstance(self.timeout_seconds, int) or not (
            1 <= self.timeout_seconds <= self.profile.maximum_timeout_seconds
        ):
            raise SpecError("timeout exceeds the selected resource-profile ceiling")
        if not isinstance(self.attempt, int) or self.attempt < 1 or self.attempt > 4:
            raise SpecError("attempt must be between 1 and 4")
        if not isinstance(self.maximum_repair_attempts, int) or not (
            0 <= self.maximum_repair_attempts <= 3
        ):
            raise SpecError("maximum_repair_attempts must be between 0 and 3")
        if self.attempt - 1 > self.maximum_repair_attempts:
            raise SpecError("attempt exceeds the configured repair limit")
        if not isinstance(self.max_estimated_cost_usd, (int, float)) or not (
            0 < self.max_estimated_cost_usd <= 100
        ):
            raise SpecError("max_estimated_cost_usd must be in (0, 100]")
        if self.profile.estimated_cost(self.timeout_seconds) > self.max_estimated_cost_usd:
            raise SpecError("resource timeout exceeds the estimated cost ceiling")


def route_for_spec(spec: RunSpec) -> str:
    if spec.dry_run or spec.operation in {"cpu-test", "cpu-finalization", "validation"}:
        return "cpu"
    if spec.model_family == "fake":
        return "cpu"
    if spec.model_family in {"dream", "diffullama"}:
        return spec.model_family
    raise SpecError("no execution route for this model and operation")


def _mounted_source(value: str, mount: Path = RESULTS_MOUNT) -> Path:
    relative = _safe_relative_path(value, label="mounted source")
    candidate = (mount / relative).resolve()
    root = mount.resolve()
    if candidate != root and root not in candidate.parents:
        raise SpecError("mounted source escaped the results volume")
    return candidate


def build_cli_args(spec: RunSpec, *, mount: Path = RESULTS_MOUNT) -> list[str]:
    """Map a validated operation to one predetermined argv (never a shell string)."""
    spec.validate()
    if spec.operation == "cpu-test":
        return [sys.executable, "-m", "pytest", "-q"]
    if spec.operation == "model-smoke-test":
        args = [
            "dlmrel",
            "smoke-test",
            "--model",
            spec.model_config,
            "--dataset",
            spec.dataset_config,
            "--experiment",
            spec.experiment_config,
        ]
        if spec.dry_run:
            args.append("--dry-run")
        return args
    if spec.operation == "experiment-run":
        args = [
            "dlmrel",
            "run",
            "--model",
            spec.model_config,
            "--dataset",
            spec.dataset_config,
            "--experiment",
            spec.experiment_config,
            "--results",
            str(spec.results_root(mount)),
            "--run-id",
            spec.run_id,
        ]
        if spec.resume:
            args.append("--resume")
        if spec.dry_run:
            args.append("--dry-run")
        if spec.selection_lock_source:
            args.extend(
                ["--selection-lock", str(_mounted_source(spec.selection_lock_source, mount))]
            )
        return args
    run_dir = str(spec.run_directory(mount))
    command = {
        "missing-test-grid-recovery": "recover-head-search-test-grid",
        "cpu-finalization": "finalize-head-search",
        "validation": "validate",
    }[spec.operation]
    return ["dlmrel", command, "--run-dir", run_dir]


def redact_secrets(value: str) -> str:
    patterns = (
        (
            r"\b(?:sk[-_][A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_-]{8,}|"
            r"github_pat_[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9_-]{8,})\b",
            "[REDACTED_TOKEN]",
        ),
        (r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1[REDACTED_TOKEN]"),
        (
            r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY)[A-Z0-9_]*\s*[=:]\s*)"
            r"[^\s,;]+",
            r"\1[REDACTED]",
        ),
    )
    redacted = value
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def normalized_failure_signature(
    *, exception_class: str | None, message: str, stage: str | None, exit_code: int | None
) -> str:
    normalized = redact_secrets(message.lower())
    normalized = re.sub(r"[a-z]:\\[^\s:]+|/(?:tmp|workspace|root)/[^\s:]+", "<path>", normalized)
    normalized = re.sub(r"\b\d{4}-\d\d-\d\d[t ][0-9:.+z-]+\b", "<timestamp>", normalized)
    normalized = re.sub(r"0x[0-9a-f]+", "<address>", normalized)
    normalized = re.sub(r"\b\d+\s*/\s*\d+\b|\b\d+%", "<progress>", normalized)
    normalized = re.sub(r"\bcontainer[-_ ]?[0-9a-f]{8,}\b", "<container>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()[-2_000:]
    payload = {
        "exception": exception_class or "",
        "message": normalized,
        "stage": stage or "",
        "exit_code": exit_code,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def classify_failure(message: str, *, exit_code: int | None = None) -> str:
    value = message.lower()
    if "protected scientific" in value:
        return "protected_scientific_behavior"
    if "scientific" in value and ("config" in value or "identity" in value):
        return "scientific_configuration_mismatch"
    if any(piece in value for piece in ("checksum", "revision mismatch", "manifest")):
        return "data_revision_checksum_failure"
    if any(piece in value for piece in ("checkpoint", "artifact mismatch", "corrupt parquet")):
        return "artifact_checkpoint_mismatch"
    if exit_code in {-9, 137} or any(
        piece in value
        for piece in (
            "out of memory",
            "cuda oom",
            "cuda out of memory",
            "oom killed",
            "operation exceeded timeout",
        )
    ):
        return "resource_exhaustion"
    if any(
        piece in value
        for piece in ("modulenotfounderror", "dependency conflict", "resolutionimpossible")
    ):
        return "dependency_environment_incompatibility"
    if any(
        piece in value
        for piece in (
            "connection reset",
            "temporary failure",
            "service unavailable",
            "gateway timeout",
            "image build failed transiently",
            "could not resolve host",
            "connection timed out",
            "tls handshake timeout",
            "remote end hung up",
            "http 429",
            "http 503",
        )
    ):
        return "transient_infrastructure"
    if any(piece in value for piece in ("traceback", "assertionerror", "typeerror", "valueerror")):
        return "deterministic_implementation_failure"
    return "unknown"


@dataclass(frozen=True)
class RunResult:
    schema_version: str
    status: str
    run_call_id: str | None
    attempt: int
    git_commit: str
    operation: str
    sanitized_cli_args: list[str]
    scientific_config_hash: str | None
    manifest_hashes: dict[str, str]
    selection_lock_hash: str | None
    started_at: str
    ended_at: str
    duration_seconds: float
    modal_reference: str | None
    exit_code: int | None
    failure_category: str | None
    exception_class: str | None
    failure_signature: str | None
    stdout_tail: str
    stderr_tail: str
    full_log_artifact: str | None
    last_completed_checkpoint: str | None
    validation_result: dict[str, Any] | None
    result_directory: str | None
    scientific_artifacts_written: bool
    recommended_next_action: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunResult:
        if not isinstance(value, dict):
            raise SpecError("RunResult must be a JSON object")
        _strict_keys(value, cls, "RunResult")
        result = cls(**value)
        if result.schema_version != RUN_RESULT_SCHEMA:
            raise SpecError("unsupported RunResult schema")
        if result.status not in {"success", "failed", "stopped", "infrastructure_retry"}:
            raise SpecError("unsupported RunResult status")
        if (
            not isinstance(result.attempt, int)
            or isinstance(result.attempt, bool)
            or not 1 <= result.attempt <= 4
        ):
            raise SpecError("RunResult attempt is invalid")
        if not isinstance(result.git_commit, str) or not COMMIT_SHA.fullmatch(result.git_commit):
            raise SpecError("RunResult git_commit is invalid")
        if result.operation not in OPERATIONS:
            raise SpecError("RunResult operation is invalid")
        if not isinstance(result.sanitized_cli_args, list) or any(
            not isinstance(item, str) for item in result.sanitized_cli_args
        ):
            raise SpecError("RunResult sanitized_cli_args are invalid")
        for name, digest in (
            ("scientific_config_hash", result.scientific_config_hash),
            ("selection_lock_hash", result.selection_lock_hash),
            ("failure_signature", result.failure_signature),
        ):
            if digest is not None and (
                not isinstance(digest, str) or not HASH.fullmatch(digest)
            ):
                raise SpecError(f"RunResult {name} is invalid")
        if not isinstance(result.manifest_hashes, dict) or (
            result.manifest_hashes
            and (
                set(result.manifest_hashes) != {"select", "dev", "test"}
                or any(
                    not isinstance(digest, str) or not HASH.fullmatch(digest)
                    for digest in result.manifest_hashes.values()
                )
            )
        ):
            raise SpecError("RunResult manifest_hashes are invalid")
        if result.failure_category is not None and result.failure_category not in FAILURE_CATEGORIES:
            raise SpecError("RunResult failure_category is invalid")
        if result.status == "success" and (
            result.failure_category is not None or result.failure_signature is not None
        ):
            raise SpecError("successful RunResult cannot contain failure fields")
        if result.status != "success" and (
            result.failure_category is None or result.failure_signature is None
        ):
            raise SpecError("unsuccessful RunResult requires classified failure fields")
        if (
            not isinstance(result.duration_seconds, (int, float))
            or isinstance(result.duration_seconds, bool)
            or result.duration_seconds < 0
        ):
            raise SpecError("RunResult duration is invalid")
        if result.exit_code is not None and (
            not isinstance(result.exit_code, int) or isinstance(result.exit_code, bool)
        ):
            raise SpecError("RunResult exit_code is invalid")
        if not isinstance(result.stdout_tail, str) or not isinstance(result.stderr_tail, str):
            raise SpecError("RunResult output tails are invalid")
        if len(result.stdout_tail) > TAIL_LIMIT or len(result.stderr_tail) > TAIL_LIMIT:
            raise SpecError("RunResult output tail exceeds its bound")
        for label, path in (
            ("full_log_artifact", result.full_log_artifact),
            ("result_directory", result.result_directory),
        ):
            if path is not None:
                _safe_relative_path(path, label=f"RunResult {label}")
        if result.validation_result is not None and not isinstance(
            result.validation_result, dict
        ):
            raise SpecError("RunResult validation_result is invalid")
        if not isinstance(result.scientific_artifacts_written, bool):
            raise SpecError("RunResult scientific_artifacts_written is invalid")
        if not isinstance(result.recommended_next_action, str):
            raise SpecError("RunResult recommended_next_action is invalid")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProcessOutcome:
    exit_code: int
    stdout: str
    stderr: str


def _sanitized_environment(cache_mount: Path) -> dict[str, str]:
    allowed = {
        "PATH",
        "PYTHONPATH",
        "LANG",
        "LC_ALL",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(
        {
            "HF_HOME": str(cache_mount / "huggingface"),
            "TRANSFORMERS_CACHE": str(cache_mount / "huggingface" / "transformers"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def stream_process(
    args: list[str], *, cwd: Path, env: dict[str, str], timeout: int, log_path: Path
) -> ProcessOutcome:
    """Run one argv without a shell while streaming and retaining bounded evidence."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_tail: deque[str] = deque()
    stderr_tail: deque[str] = deque()
    tail_sizes = {"stdout": 0, "stderr": 0}
    lock = threading.Lock()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            errors="replace",
        )

        def pump(stream, label: str, output, target: deque[str]) -> None:
            assert stream is not None
            for line in iter(stream.readline, ""):
                clean = redact_secrets(line)
                output.write(clean)
                output.flush()
                with lock:
                    log.write(f"[{label}] {clean}")
                    log.flush()
                    target.append(clean)
                    tail_sizes[label] += len(clean)
                    while tail_sizes[label] > TAIL_LIMIT and target:
                        tail_sizes[label] -= len(target.popleft())

        threads = [
            threading.Thread(
                target=pump, args=(process.stdout, "stdout", sys.stdout, stdout_tail), daemon=True
            ),
            threading.Thread(
                target=pump, args=(process.stderr, "stderr", sys.stderr, stderr_tail), daemon=True
            ),
        ]
        for thread in threads:
            thread.start()
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = 124
            stderr_tail.append(f"operation exceeded timeout of {timeout} seconds\n")
        for thread in threads:
            thread.join(timeout=10)
    return ProcessOutcome(exit_code, "".join(stdout_tail)[-TAIL_LIMIT:], "".join(stderr_tail)[-TAIL_LIMIT:])


def checkout_exact_commit(commit: str, destination: Path) -> Path:
    if not COMMIT_SHA.fullmatch(commit):
        raise SpecError("refusing to checkout a non-immutable commit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY_URL, str(destination)],
        check=True,
        shell=False,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit],
        check=True,
        shell=False,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
        shell=False,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise SpecError("checked-out repository commit differs from RunSpec")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(destination)],
        check=True,
        shell=False,
    )
    return destination


def _atomic_copy_source(spec: RunSpec, *, mount: Path) -> None:
    if not spec.source_result_identity:
        return
    source = _mounted_source(spec.source_result_identity, mount)
    target = spec.run_directory(mount)
    if source.resolve() == target.resolve():
        raise SpecError("source result and attempt target must be different directories")
    if target.exists():
        return
    if not source.is_dir():
        raise FileNotFoundError(f"staged source result does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    os.replace(temporary, target)


def _completed_validated(path: Path) -> bool:
    summary = path / "summary.json"
    validation = path / "validation.json"
    if not summary.is_file() or not validation.is_file():
        return False
    try:
        return (
            json.loads(summary.read_text(encoding="utf-8")).get("completion_status") == "complete"
            and json.loads(validation.read_text(encoding="utf-8")).get("valid") is True
        )
    except (OSError, json.JSONDecodeError):
        return False


def _snapshot(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): (item.stat().st_size, item.stat().st_mtime_ns)
        for item in path.rglob("*")
        if item.is_file() and not item.name.endswith(".tmp")
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _last_checkpoint(run_dir: Path) -> str | None:
    records = []
    for path in (run_dir / "checkpoints").glob("*.meta.json"):
        metadata = _read_json(path)
        if metadata:
            records.append(
                (
                    path.stat().st_mtime_ns,
                    int(metadata.get("sentence_end", -1)),
                    str(metadata.get("stage", "")),
                    path.name,
                )
            )
    return max(records)[3] if records else None


def _runtime_versions(repository: Path, spec: RunSpec) -> dict[str, Any]:
    packages = {}
    for name in (
        "dlmrel",
        "torch",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "numpy",
        "pandas",
        "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        torch_runtime = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except (ImportError, RuntimeError):
        torch_runtime = None
    try:
        gpu_runtime = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu_runtime = None
    model_identity = None
    if spec.model_config:
        import yaml

        model = yaml.safe_load((repository / spec.model_config).read_text(encoding="utf-8"))
        model_identity = {
            key: model.get(key)
            for key in (
                "id",
                "name",
                "revision",
                "tokenizer_revision",
                "remote_code_revision",
                "dtype",
                "attn_implementation",
            )
        }
    return {
        "python": sys.version,
        "packages": packages,
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_identity": model_identity,
        "torch_runtime": torch_runtime,
        "nvidia_smi": gpu_runtime,
    }


def _validate_expected_identity(spec: RunSpec, run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    metadata = _read_json(run_dir / "run_metadata.json") or {}
    manifests = _read_json(run_dir / "manifest_refs.json") or {}
    scientific_hash = metadata.get("scientific_config_hash") or metadata.get("config_hash")
    if spec.expected_scientific_config_hash and scientific_hash != spec.expected_scientific_config_hash:
        legacy_migration = (
            spec.operation == "missing-test-grid-recovery"
            and metadata.get("legacy_scientific_config_hash")
            == spec.expected_scientific_config_hash
        )
        if not legacy_migration:
            raise SpecError("scientific configuration identity differs from RunSpec")
    if spec.expected_manifest_hashes and manifests != spec.expected_manifest_hashes:
        raise SpecError("manifest hashes differ from RunSpec")
    return metadata, manifests


def execute_spec(
    spec: RunSpec,
    *,
    repository: Path,
    results_mount: Path,
    cache_mount: Path,
    logs_mount: Path,
    process_runner: Callable[..., ProcessOutcome] = stream_process,
    call_id: str | None = None,
    commit_results: Callable[[], None] | None = None,
    commit_logs: Callable[[], None] | None = None,
) -> RunResult:
    """Execute exactly one existing CLI operation and return structured evidence."""
    spec.validate()
    started = datetime.now(timezone.utc)
    run_dir = spec.run_directory(results_mount)
    log_relative = PurePosixPath(
        spec.result_namespace,
        spec.run_id,
        spec.operation,
        f"attempt-{spec.attempt}.log",
    ).as_posix()
    log_path = logs_mount / log_relative
    args = build_cli_args(spec, mount=results_mount)
    stdout_tail = ""
    stderr_tail = ""
    exit_code: int | None = None
    exception_class: str | None = None
    failure_category: str | None = None
    signature: str | None = None
    status = "failed"
    recommendation = "stop_for_human_review"
    before = _snapshot(run_dir)
    try:
        _atomic_copy_source(spec, mount=results_mount)
        if commit_results is not None:
            commit_results()
        if _completed_validated(run_dir):
            if spec.operation == "validation":
                status = "success"
                recommendation = "no_action_completed_run_unchanged"
            else:
                status = "stopped"
                failure_category = "protected_scientific_behavior"
                stderr_tail = "completed validated run will not be overwritten"
                recommendation = "use_a_new_run_id_or_human_approved_copy"
        else:
            if run_dir.exists() and spec.expected_scientific_config_hash:
                _validate_expected_identity(spec, run_dir)
            outcome = process_runner(
                args,
                cwd=repository,
                env=_sanitized_environment(cache_mount),
                timeout=spec.timeout_seconds,
                log_path=log_path,
            )
            exit_code = outcome.exit_code
            stdout_tail = redact_secrets(outcome.stdout)[-TAIL_LIMIT:]
            stderr_tail = redact_secrets(outcome.stderr)[-TAIL_LIMIT:]
            if exit_code == 0:
                status = "success"
                recommendation = "validate_and_request_human_promotion"
            else:
                failure_category = classify_failure(stderr_tail or stdout_tail, exit_code=exit_code)
                recommendation = {
                    "transient_infrastructure": "retry_same_commit_once",
                    "resource_exhaustion": "review_resource_ceiling",
                    "deterministic_implementation_failure": "bounded_repair",
                }.get(failure_category, "stop_for_human_review")
        if run_dir.exists() and spec.expected_scientific_config_hash:
            _validate_expected_identity(spec, run_dir)
    except Exception as error:  # Ordinary worker failures must become structured evidence.
        status = "failed"
        exception_class = type(error).__name__
        stderr_tail = redact_secrets(
            (stderr_tail + "\n" + "".join(traceback.format_exception(error)))[-TAIL_LIMIT:]
        )
        failure_category = classify_failure(stderr_tail, exit_code=exit_code)
        recommendation = (
            "retry_same_commit_once"
            if failure_category == "transient_infrastructure"
            else "stop_for_human_review"
        )
    finally:
        if commit_results is not None:
            commit_results()
        if commit_logs is not None:
            commit_logs()
    ended = datetime.now(timezone.utc)
    metadata = _read_json(run_dir / "run_metadata.json") or {}
    manifests = _read_json(run_dir / "manifest_refs.json") or {}
    validation = _read_json(run_dir / "validation.json")
    summary = _read_json(run_dir / "summary.json") or {}
    if failure_category:
        signature = normalized_failure_signature(
            exception_class=exception_class,
            message=stderr_tail or stdout_tail,
            stage=(summary.get("last_completed_stage") or _last_checkpoint(run_dir)),
            exit_code=exit_code,
        )
    return RunResult(
        schema_version=RUN_RESULT_SCHEMA,
        status=status,
        run_call_id=call_id,
        attempt=spec.attempt,
        git_commit=spec.git_commit,
        operation=spec.operation,
        sanitized_cli_args=[redact_secrets(str(value)) for value in args],
        scientific_config_hash=(
            metadata.get("scientific_config_hash") or metadata.get("config_hash")
        ),
        manifest_hashes=manifests,
        selection_lock_hash=metadata.get("selection_lock_hash"),
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
        duration_seconds=round((ended - started).total_seconds(), 3),
        modal_reference=f"modal-function-call:{call_id}" if call_id else None,
        exit_code=exit_code,
        failure_category=failure_category,
        exception_class=exception_class,
        failure_signature=signature,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        full_log_artifact=log_relative,
        last_completed_checkpoint=_last_checkpoint(run_dir),
        validation_result=validation,
        result_directory=run_dir.relative_to(results_mount).as_posix() if run_dir.exists() else None,
        scientific_artifacts_written=_snapshot(run_dir) != before,
        recommended_next_action=recommendation,
    )


def _failed_before_worker(spec: RunSpec, error: Exception) -> RunResult:
    now = datetime.now(timezone.utc).isoformat()
    evidence = redact_secrets("".join(traceback.format_exception(error))[-TAIL_LIMIT:])
    category = classify_failure(evidence)
    return RunResult(
        RUN_RESULT_SCHEMA,
        "failed",
        None,
        spec.attempt,
        spec.git_commit,
        spec.operation,
        [],
        None,
        {},
        None,
        now,
        now,
        0.0,
        None,
        None,
        category,
        type(error).__name__,
        normalized_failure_signature(
            exception_class=type(error).__name__, message=evidence, stage="checkout", exit_code=None
        ),
        "",
        evidence,
        None,
        None,
        None,
        None,
        False,
        "stop_for_human_review",
    )


if modal is not None:  # pragma: no cover - definitions are inspected, not invoked, in CPU tests
    _base_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install_from_pyproject(str(REPOSITORY_ROOT / "pyproject.toml"), optional_dependencies=["dev"])
    )
    _cpu_image = _base_image
    _dream_image = _base_image.pip_install_from_requirements(
        str(REPOSITORY_ROOT / "requirements" / "dream.txt")
    )
    _diffullama_image = _base_image.pip_install_from_requirements(
        str(REPOSITORY_ROOT / "requirements" / "diffullama.txt")
    )
    app = modal.App("dlmrel-guarded-runner", include_source=True)
    _cache_volume = modal.Volume.from_name("dlmrel-model-cache", create_if_missing=True)
    _results_volume = modal.Volume.from_name("dlmrel-scratch-results", create_if_missing=True)
    _logs_volume = modal.Volume.from_name("dlmrel-attempt-logs", create_if_missing=True)
    _run_locks = modal.Dict.from_name("dlmrel-run-identity-locks", create_if_missing=True)

    def _remote_worker(spec_json: str, lock_token: str) -> str:
        spec = RunSpec.from_json(spec_json)
        if _run_locks.get(spec.run_identity) != lock_token:
            raise SpecError("missing or stale run-identity lock")
        _results_volume.reload()
        _logs_volume.reload()
        try:
            with tempfile.TemporaryDirectory(prefix="dlmrel-source-") as temporary:
                repository = checkout_exact_commit(spec.git_commit, Path(temporary) / "repository")
                call_id = modal.current_function_call_id()
                versions = _runtime_versions(repository, spec)
                print(json.dumps({"runtime": versions}, sort_keys=True))
                result = execute_spec(
                    spec,
                    repository=repository,
                    results_mount=RESULTS_MOUNT,
                    cache_mount=CACHE_MOUNT,
                    logs_mount=LOGS_MOUNT,
                    call_id=call_id,
                    commit_results=_results_volume.commit,
                    commit_logs=_logs_volume.commit,
                )
                return result.to_json()
        except Exception as error:
            return _failed_before_worker(spec, error).to_json()

    _volume_mounts = {
        str(CACHE_MOUNT): _cache_volume,
        str(RESULTS_MOUNT): _results_volume,
        str(LOGS_MOUNT): _logs_volume,
    }

    @app.function(image=_cpu_image, volumes=_volume_mounts, timeout=86_400)
    def cpu_job(spec_json: str, lock_token: str) -> str:
        return _remote_worker(spec_json, lock_token)

    @app.function(image=_dream_image, volumes=_volume_mounts, timeout=86_400)
    def dream_gpu_job(spec_json: str, lock_token: str) -> str:
        return _remote_worker(spec_json, lock_token)

    @app.function(image=_diffullama_image, volumes=_volume_mounts, timeout=86_400)
    def diffullama_gpu_job(spec_json: str, lock_token: str) -> str:
        return _remote_worker(spec_json, lock_token)

    @app.function(image=_cpu_image, timeout=86_400)
    def submit_job(spec_json: str) -> str:
        spec = RunSpec.from_json(spec_json)
        token = uuid.uuid4().hex
        if not _run_locks.put(spec.run_identity, token, skip_if_exists=True):
            raise SpecError("another controller already holds this run identity")
        try:
            function = {
                "cpu": cpu_job,
                "dream": dream_gpu_job,
                "diffullama": diffullama_gpu_job,
            }[route_for_spec(spec)]
            options = {
                "cpu": spec.profile.cpu,
                "memory": spec.profile.memory_mib,
                "timeout": spec.timeout_seconds,
            }
            if spec.profile.gpu:
                options["gpu"] = spec.profile.gpu
            return function.with_options(**options).remote(spec_json, token)
        finally:
            if _run_locks.get(spec.run_identity) == token:
                _run_locks.pop(spec.run_identity)

    @app.local_entrypoint()
    def main(spec_json: str) -> str:
        """Submit one validated ephemeral call; this does not deploy a persistent App."""
        RunSpec.from_json(spec_json)
        return submit_job.remote(spec_json)
else:
    app = None

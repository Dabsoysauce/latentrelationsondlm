"""Versioned, non-overwriting run artifacts and immutable selection locks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RUN_SCHEMA = "dlmrel-run-v1"
LOCK_SCHEMA = "dlmrel-selection-lock-v1"
REQUIRED_RUN_FILES = {
    "config.resolved.yaml",
    "command.txt",
    "run_metadata.json",
    "environment.json",
    "manifest_refs.json",
    "exclusions.parquet",
    "instances.parquet",
    "per_seed_metrics.csv",
    "metrics.csv",
    "summary.json",
    "validation.json",
}


class ArtifactError(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class SelectionLock:
    schema_version: str
    track: str
    model_id: str
    model_revision: str
    dataset_id: str
    relation: str
    layer: int
    head: int
    top_k: int
    metric: str
    decision_rule: str
    tie_break: str
    config_hash: str
    select_manifest_hash: str
    dev_manifest_hash: str
    candidate_scores_hash: str
    frozen_settings: dict[str, Any]
    created_at: str

    @classmethod
    def create(cls, **values: Any) -> SelectionLock:
        return cls(schema_version=LOCK_SCHEMA, created_at=_now(), **values)

    def write_once(self, path: str | Path) -> None:
        path = Path(path)
        payload = asdict(self)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise ArtifactError(f"refusing to overwrite immutable selection lock: {path}")
            return
        atomic_json(path, payload)


def load_selection_lock(
    path: str | Path,
    *,
    config_hash: str,
    select_manifest_hash: str,
    dev_manifest_hash: str,
) -> SelectionLock:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != {item.name for item in SelectionLock.__dataclass_fields__.values()}:
        raise ArtifactError("selection lock schema fields do not match")
    lock = SelectionLock(**raw)
    if lock.schema_version != LOCK_SCHEMA:
        raise ArtifactError("unsupported selection lock schema")
    expected = (config_hash, select_manifest_hash, dev_manifest_hash)
    actual = (lock.config_hash, lock.select_manifest_hash, lock.dev_manifest_hash)
    if actual != expected:
        raise ArtifactError("selection lock hash mismatch")
    return lock


def run_directory(
    root: str | Path, track: str, model: str, dataset: str, experiment: str, run_id: str
) -> Path:
    return Path(root) / track / model / dataset / experiment / run_id


def initialize_run(
    path: str | Path,
    config: dict[str, Any],
    command: str,
    manifests: dict[str, str],
    *,
    resume: bool = False,
) -> Path:
    path = Path(path)
    if path.exists() and not resume:
        raise ArtifactError(f"run directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if (path / "summary.json").exists():
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("completion_status") == "complete":
            raise ArtifactError("completed runs cannot be resumed or overwritten")
    config_hash = canonical_hash(config)
    resolved = path / "config.resolved.yaml"
    if resolved.exists():
        existing = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if canonical_hash(existing) != config_hash:
            raise ArtifactError("resume config differs from the existing run")
    else:
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (path / "command.txt").write_text(command.rstrip() + "\n", encoding="utf-8")
    atomic_json(path / "manifest_refs.json", manifests)
    atomic_json(path / "environment.json", environment_record())
    atomic_json(
        path / "run_metadata.json",
        {
            "schema_version": RUN_SCHEMA,
            "config_hash": config_hash,
            "started_at": _now(),
            "completion_status": "running",
        },
    )
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    return path


def write_shard(run_dir: str | Path, shard_id: int, payload: list[dict[str, Any]]) -> Path:
    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard-{shard_id:06d}.json"
    value = {
        "schema_version": RUN_SCHEMA,
        "shard_id": shard_id,
        "rows": payload,
        "rows_hash": canonical_hash(payload),
    }
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise ArtifactError(f"duplicate shard id with different content: {shard_id}")
        return path
    atomic_json(path, value)
    return path


def merge_shards(run_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted((Path(run_dir) / "checkpoints").glob("shard-*.json")):
        shard = json.loads(path.read_text(encoding="utf-8"))
        shard_id = int(shard["shard_id"])
        if shard_id in seen:
            raise ArtifactError(f"duplicate shard id: {shard_id}")
        if canonical_hash(shard["rows"]) != shard["rows_hash"]:
            raise ArtifactError(f"corrupt shard: {path}")
        seen.add(shard_id)
        rows.extend(shard["rows"])
    return rows


def validate_run(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    missing = sorted(name for name in REQUIRED_RUN_FILES if not (path / name).exists())
    errors: list[str] = []
    if missing:
        errors.append("missing files: " + ", ".join(missing))
    metadata_path = path / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != RUN_SCHEMA:
            errors.append("run metadata schema mismatch")
        config_path = path / "config.resolved.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if canonical_hash(config) != metadata.get("config_hash"):
                errors.append("resolved config hash mismatch")
        summary_path = path / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("completion_status") != metadata.get("completion_status"):
                errors.append("summary and metadata completion status differ")
            claimed = summary.get("capabilities", {})
            if config_path.exists():
                declared = config.get("model", {}).get("capabilities", {})
                unsupported = sorted(
                    name for name, value in claimed.items() if value and not declared.get(name, False)
                )
                if unsupported:
                    errors.append("unsupported capability claim: " + ", ".join(unsupported))
    validation = {
        "schema_version": RUN_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "validated_at": _now(),
    }
    if path.exists():
        atomic_json(path / "validation.json", validation)
    return validation


def environment_record() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unknown", "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": sha,
        "git_dirty": dirty,
        "cuda": "not_probed_by_cpu_core",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

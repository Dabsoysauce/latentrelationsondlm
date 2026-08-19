"""Versioned, non-overwriting run artifacts and immutable selection locks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
}


class ArtifactError(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def json_compatible(value: Any, *, location: str = "$") -> Any:
    """Recursively normalize NumPy/Parquet values into strict JSON values."""
    if isinstance(value, np.ndarray):
        return [json_compatible(item, location=f"{location}[]") for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_compatible(value.item(), location=location)
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            normalized_key = json_compatible(key, location=f"{location}.<key>")
            if not isinstance(normalized_key, (str, int, float, bool, type(None))):
                raise ArtifactError(f"non-JSON mapping key at {location}: {key!r}")
            normalized[normalized_key] = json_compatible(item, location=f"{location}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [json_compatible(item, location=f"{location}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"non-finite float at {location}: {value!r}")
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert nullable table cells to JSON null while still rejecting infinities."""
    nullable = frame.astype(object).where(pd.notna(frame), None)
    return json_compatible(nullable.to_dict("records"))


def selection_lock_scientific_dict(value: SelectionLock | dict[str, Any]) -> dict[str, Any]:
    """Return lock contents without operational provenance timestamps."""
    payload = asdict(value) if isinstance(value, SelectionLock) else deepcopy(value)
    payload.pop("created_at", None)
    return payload


def selection_lock_hash(value: SelectionLock | dict[str, Any]) -> str:
    return canonical_hash(selection_lock_scientific_dict(value))


def selection_source_hash(path: str | Path) -> str:
    """Hash a legacy object lock or the scientific identities in a six-lock manifest."""
    source = Path(path)
    if source.is_dir() and (source / "relation-selection").is_dir():
        source = source / "relation-selection"
    manifest = (
        source / "relation_selection_bundle.json"
        if source.is_dir()
        else source
    )
    if manifest.name == "relation_selection_bundle.json":
        bundle = json.loads(manifest.read_text(encoding="utf-8"))
        relations = bundle.get("relations", {})
        identities = {
            relation: record.get("selection_lock_hash")
            for relation, record in relations.items()
        }
        if not identities or any(value is None for value in identities.values()):
            raise ArtifactError("relation-lock manifest has incomplete scientific identities")
        return canonical_hash(
            {
                "schema_version": bundle.get("schema_version"),
                "primary_relation": bundle.get("primary_relation"),
                "relation_lock_hashes": identities,
            }
        )
    return selection_lock_hash(json.loads(source.read_text(encoding="utf-8")))


def scientific_configuration(
    config: dict[str, Any], *, resolved_selection_lock_hash: str | None = None
) -> dict[str, Any]:
    """Remove execution-only runtime state while retaining source-lock identity."""
    value = deepcopy(config)
    runtime = value.pop("runtime", {}) or {}
    lock_path = runtime.get("selection_lock")
    if lock_path:
        digest = resolved_selection_lock_hash
        if digest is None:
            digest = selection_source_hash(lock_path)
        value["selection_lock_hash"] = digest
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_compatible(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
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
        created_at = values.pop("created_at", None) or _now()
        return cls(schema_version=LOCK_SCHEMA, created_at=created_at, **values)

    def write_once(self, path: str | Path) -> None:
        path = Path(path)
        payload = json_compatible(asdict(self))
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
    lock_path = (config.get("runtime") or {}).get("selection_lock")
    current_lock_hash = None
    if lock_path:
        current_lock_hash = selection_source_hash(lock_path)
    scientific = scientific_configuration(
        config, resolved_selection_lock_hash=current_lock_hash
    )
    config_hash = canonical_hash(scientific)
    resolved = path / "config.resolved.yaml"
    metadata_path = path / "run_metadata.json"
    existing_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    if resolved.exists():
        existing = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        stored_lock_hash = existing_metadata.get("selection_lock_hash")
        existing_lock_path = (existing.get("runtime") or {}).get("selection_lock")
        if stored_lock_hash is None and existing_lock_path:
            try:
                stored_lock_hash = selection_source_hash(existing_lock_path)
            except FileNotFoundError:
                # Pre-normalization metadata did not retain this digest. If the
                # original path moved, the incoming lock is the only identity
                # available for a one-time migration.
                stored_lock_hash = current_lock_hash
        existing_scientific = scientific_configuration(
            existing, resolved_selection_lock_hash=stored_lock_hash
        )
        if canonical_hash(existing_scientific) != config_hash:
            raise ArtifactError("resume config differs from the existing run")
    else:
        resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest_path = path / "manifest_refs.json"
    if manifest_path.exists():
        existing_manifests = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifests != manifests:
            raise ArtifactError("resume manifests differ from the existing run")
    else:
        atomic_json(manifest_path, manifests)
    (path / "command.txt").write_text(command.rstrip() + "\n", encoding="utf-8")
    atomic_json(path / "environment.json", environment_record())
    metadata = existing_metadata
    metadata.setdefault("started_at", _now())
    if existing_metadata:
        metadata["last_resumed_at"] = _now()
    metadata.update(
        {
            "schema_version": RUN_SCHEMA,
            "config_hash": config_hash,
            "scientific_config_hash": config_hash,
            "selection_lock_hash": current_lock_hash,
            "manifest_hashes_hash": canonical_hash(manifests),
            "completion_status": "running",
        }
    )
    atomic_json(
        metadata_path,
        metadata,
    )
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    return path


def write_shard(run_dir: str | Path, shard_id: int, payload: list[dict[str, Any]]) -> Path:
    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard-{shard_id:06d}.json"
    normalized_payload = json_compatible(payload)
    value = {
        "schema_version": RUN_SCHEMA,
        "shard_id": shard_id,
        "rows": normalized_payload,
        "rows_hash": canonical_hash(normalized_payload),
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


def final_artifact_hashes(run_dir: str | Path) -> dict[str, str]:
    """Hash finalized scientific outputs while excluding operational/mutable files."""
    root = Path(run_dir)
    excluded_files = {
        "command.txt",
        "config.resolved.yaml",
        "environment.json",
        "manifest_refs.json",
        "run_metadata.json",
        "validation.json",
    }
    hashes = {}
    for candidate in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = candidate.relative_to(root)
        if (
            relative.as_posix() in excluded_files
            or relative.parts[0] in {"checkpoints", "figures"}
            or candidate.name.endswith(".tmp")
        ):
            continue
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[relative.as_posix()] = digest.hexdigest()
    return hashes


def validate_run(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    missing = sorted(name for name in REQUIRED_RUN_FILES if not (path / name).exists())
    errors: list[str] = []
    if missing:
        errors.append("missing files: " + ", ".join(missing))

    metadata = _validated_json(path / "run_metadata.json", "run metadata", errors)
    manifests = _validated_json(path / "manifest_refs.json", "manifest references", errors)
    summary = _validated_json(path / "summary.json", "summary", errors)
    environment = _validated_json(path / "environment.json", "environment", errors)
    config = _validated_yaml(path / "config.resolved.yaml", "resolved config", errors)

    metadata_path = path / "run_metadata.json"
    if metadata_path.exists() and metadata is not None:
        if metadata.get("schema_version") != RUN_SCHEMA:
            errors.append("run metadata schema mismatch")
        if manifests is not None and metadata.get("manifest_hashes_hash") is not None:
            if canonical_hash(manifests) != metadata.get("manifest_hashes_hash"):
                errors.append("manifest references hash mismatch")
        if config is not None:
            recorded = metadata.get("scientific_config_hash")
            if recorded is not None:
                scientific = scientific_configuration(
                    config,
                    resolved_selection_lock_hash=metadata.get("selection_lock_hash"),
                )
                if canonical_hash(scientific) != recorded:
                    errors.append("resolved scientific config hash mismatch")
            elif canonical_hash(config) != metadata.get("config_hash"):
                errors.append("resolved config hash mismatch")
        if summary is not None:
            if summary.get("completion_status") != metadata.get("completion_status"):
                errors.append("summary and metadata completion status differ")
            if summary.get("completion_status") != "complete":
                errors.append("run is not marked complete")
            claimed = summary.get("capabilities", {})
            if config is not None:
                declared = config.get("model", {}).get("capabilities", {})
                unsupported = sorted(
                    name for name, value in claimed.items() if value and not declared.get(name, False)
                )
                if unsupported:
                    errors.append("unsupported capability claim: " + ", ".join(unsupported))
        recorded_artifacts = metadata.get("final_artifact_hashes")
        if recorded_artifacts is not None and recorded_artifacts != final_artifact_hashes(path):
            errors.append("final scientific artifact hashes differ")

    instances = _validated_table(path / "instances.parquet", "instances", errors)
    exclusions = _validated_table(path / "exclusions.parquet", "exclusions", errors)
    per_seed = _validated_table(path / "per_seed_metrics.csv", "per-seed metrics", errors)
    metrics = _validated_table(path / "metrics.csv", "metrics", errors)
    if instances is not None:
        if instances.empty:
            errors.append("instances artifact is empty")
        _validate_finite_table(instances, "instances", errors)
        base_identity = "instance_id" if "instance_id" in instances else "sentence_id"
        observation_columns = [
            column
            for column in (
                base_identity,
                "word_index",
                "seed",
                "timestep",
                "normalized_progress",
                "relation",
                "layer",
                "head",
                "depth",
                "position",
            )
            if column in instances
        ]
        if observation_columns and instances.duplicated(observation_columns).any():
            errors.append("instances artifact contains duplicate observations")
        try:
            shard_rows = merge_shards(path)
            if not shard_rows:
                errors.append("instance JSON shards are missing")
            elif canonical_hash(shard_rows) != canonical_hash(dataframe_records(instances)):
                errors.append("instance Parquet and JSON shards differ")
        except (ArtifactError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"instance JSON shards are invalid: {error}")
        if summary is not None:
            if "n_rows" in summary and int(summary["n_rows"]) != len(instances):
                errors.append("summary row count differs from instances")
            for count_name in ("n_instances", "n_test_instances"):
                if count_name in summary and "instance_id" in instances:
                    if int(summary[count_name]) != instances["instance_id"].nunique():
                        errors.append(f"summary {count_name} differs from instances")
    if exclusions is not None:
        expected = {"sentence_id", "instance_id", "role", "reason"}
        if not expected.issubset(exclusions):
            errors.append("exclusions artifact schema is incomplete")
    for label, frame in (("per-seed metrics", per_seed), ("metrics", metrics)):
        if frame is not None:
            if frame.empty:
                errors.append(f"{label} artifact is empty")
            _validate_finite_table(frame, label, errors)
    if environment is not None and not isinstance(environment, dict):
        errors.append("environment must be a JSON object")

    command_path = path / "command.txt"
    if command_path.exists() and not command_path.read_text(encoding="utf-8").strip():
        errors.append("command artifact is empty")

    if config is not None and not missing:
        try:
            from .config import RunConfig

            resolved_config = RunConfig.from_dict(config)
            _validate_experiment_artifacts(path, resolved_config, instances, errors)
            relation_bundle = path / "relation-selection"
            if relation_bundle.is_dir():
                from .relation_selection import load_relation_locks

                load_relation_locks(relation_bundle, resolved_config)
        except (ArtifactError, TypeError, ValueError) as error:
            errors.append(f"resolved run science or relation locks are invalid: {error}")
    validation = {
        "schema_version": RUN_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "validated_at": _now(),
    }
    if path.exists():
        atomic_json(path / "validation.json", validation)
    return validation


def _validated_json(path: Path, label: str, errors: list[str]) -> Any | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        canonical_hash(value)
        return value
    except (ArtifactError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{label} is invalid: {error}")
        return None


def _validated_yaml(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("expected a mapping")
        canonical_hash(value)
        return value
    except (ArtifactError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        errors.append(f"{label} is invalid: {error}")
        return None


def _validated_table(path: Path, label: str, errors: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"{label} artifact is unreadable: {error}")
        return None


def _validate_finite_table(frame: pd.DataFrame, label: str, errors: list[str]) -> None:
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            errors.append(f"{label} column {column} contains infinity")
    required_finite = {
        "seed",
        "timestep",
        "normalized_progress",
        "layer",
        "head",
        "correct",
        "accuracy",
        "top1",
        "top5",
        "rank",
        "mrr",
        "entropy",
        "entropy_normalized",
        "entropy_no_bos",
        "bos_sink_mass",
        "selected_c",
    }
    for column in sorted(required_finite.intersection(frame.columns)):
        if frame[column].isna().any():
            errors.append(f"{label} column {column} contains missing/non-finite values")


def _validate_experiment_artifacts(path, cfg, instances, errors: list[str]) -> None:
    if instances is None:
        return
    required_columns = {
        "head_search": {
            "sentence_id",
            "instance_id",
            "seed",
            "relation",
            "layer",
            "head",
            "correct",
        },
        "time_curve": {
            "sentence_id",
            "instance_id",
            "seed",
            "relation",
            "layer",
            "head",
            "timestep",
            "normalized_progress",
            "visibility",
            "correct",
        },
        "attention_entropy": {
            "sentence_id",
            "seed",
            "layer",
            "head",
            "timestep",
            "normalized_progress",
            "entropy",
            "entropy_normalized",
            "entropy_no_bos",
            "bos_sink_mass",
        },
        "logit_lens": {
            "sentence_id",
            "seed",
            "depth",
            "position",
            "position_state",
            "rank",
            "top1",
            "top5",
            "mrr",
        },
        "pos_probe": {
            "sentence_id",
            "word_index",
            "seed",
            "gold_upos",
            "prediction",
        },
    }[cfg.experiment.type]
    if missing := required_columns - set(instances):
        errors.append(f"{cfg.experiment.type} instances missing columns: {sorted(missing)}")
    if "seed" in instances and set(instances["seed"].dropna().astype(int)) != set(
        cfg.experiment.seeds
    ):
        errors.append("instances do not contain exactly the configured seeds")
    if cfg.experiment.type in {"time_curve", "attention_entropy", "logit_lens"}:
        progress = sorted(instances["normalized_progress"].dropna().astype(float).unique())
        if progress != sorted(cfg.experiment.normalized_progress):
            errors.append("instances do not contain every configured progress point")
        expected_times = sorted(
            {round(value * (cfg.experiment.steps - 1)) for value in cfg.experiment.normalized_progress}
        )
        if sorted(instances["timestep"].dropna().astype(int).unique()) != expected_times:
            errors.append("instances use incorrect or incomplete timesteps")

    lock_required = cfg.experiment.type == "time_curve" or cfg.experiment.type == "head_search"
    if lock_required:
        for name in ("selection_lock.json", "relation_locks.resolved.json"):
            if not (path / name).is_file():
                errors.append(f"locked experiment is missing {name}")
        resolved_path = path / "relation_locks.resolved.json"
        if resolved_path.is_file():
            try:
                from .relation_selection import load_relation_locks

                resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
                source_locks = load_relation_locks(resolved["source"], cfg)
                expected = {
                    relation: {
                        "layer": lock.layer,
                        "head": lock.head,
                        "selection_lock_hash": selection_lock_hash(lock),
                    }
                    for relation, lock in source_locks.locks.items()
                }
                if resolved.get("relations") != expected:
                    errors.append("resolved relation locks differ from their immutable source")
                primary = source_locks.resolve("object_to_verb")
                embedded_primary = json.loads(
                    (path / "selection_lock.json").read_text(encoding="utf-8")
                )
                if selection_lock_hash(embedded_primary) != selection_lock_hash(primary):
                    errors.append("embedded primary lock differs from its immutable source")
            except (ArtifactError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"resolved relation-lock provenance is invalid: {error}")
    if cfg.experiment.type == "head_search" and cfg.track == "confirmatory_ewt":
        for name in (
            "relation-selection/relation_selection_bundle.json",
            "selection_permutation.json",
            "selection_permutation_results.csv",
        ):
            if not (path / name).is_file():
                errors.append(f"confirmatory head search is missing {name}")


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

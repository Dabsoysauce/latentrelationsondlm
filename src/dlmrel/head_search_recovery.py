"""Phased recovery of interrupted confirmatory EWT head-search runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .artifacts import (
    ArtifactError,
    atomic_json,
    canonical_hash,
    final_artifact_hashes,
    scientific_configuration,
    validate_run,
)
from .checkpoints import CheckpointIdentity, SentenceCheckpointStore
from .config import RELATION_NAMES, RunConfig
from .data import load_manifest_examples
from .experiments.shared import (
    locked_metrics,
    per_seed_metrics,
    score_attention_heads,
    selection_progress,
    structural_slices,
)
from .permutation import selection_aware_permutation
from .relation_selection import (
    MINIMUM_DENOMINATOR,
    PRIMARY_RELATION,
    SECONDARY_RELATIONS,
    RelationLockSet,
    filter_relation_locked_rows,
    load_relation_locks,
    write_resolved_lock_manifest,
)

RECOVERY_SCHEMA = "dlmrel-head-search-recovery-v1"
TEST_ALL_HEAD_STAGE = "test-all-heads-selection-permutation"
TEST_ALL_HEAD_EVIDENCE = "test_all_head_permutation_evidence.parquet"
TEST_ALL_HEAD_METADATA = "test_all_head_permutation_evidence.meta.json"
PERMUTATION_COLUMNS = (
    "sentence_id",
    "instance_id",
    "role",
    "relation",
    "seed",
    "timestep",
    "normalized_progress",
    "layer",
    "head",
    "predicted_word_idx",
    "gold_receiver_word_idx",
    "attender_word_idx",
    "sentence_length_words",
    "n_candidate_words",
    "correct",
)
PERMUTATION_SORT = ("sentence_id", "instance_id", "seed", "layer", "head")
LOCKED_METRIC_COLUMNS = (
    "sentence_id",
    "instance_id",
    "role",
    "treebank",
    "relation",
    "seed",
    "timestep",
    "normalized_progress",
    "visibility",
    "layer",
    "head",
    "correct",
    "attender_word_idx",
    "receiver_word_idx",
    "nearest_correct",
    "uniform_correct",
    "previous_correct",
    "next_correct",
    "oracle_pos_correct",
    "wrong_same_pos_correct",
    "gold_attention_mass",
    "matched_attention_mass",
    "matched_gold_greater",
    "signed_distance",
    "direction",
    "coordinated",
    "embedded_clause",
    "relative_clause",
    "passive_voice",
    "punctuation_between",
)
SELECTION_SOURCE_FILES = (
    "config.resolved.yaml",
    "manifest_refs.json",
    "select_all_head_scores.csv",
    "dev_all_head_scores.csv",
    "select_instances.parquet",
    "dev_instances.parquet",
)


def load_saved_head_search_config(run_dir: str | Path) -> RunConfig:
    """Load the immutable resolved config from an existing EWT head-search run."""
    path = Path(run_dir) / "config.resolved.yaml"
    if not path.is_file():
        raise ArtifactError(f"recovery run is missing {path.name}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ArtifactError("recovery resolved config is unreadable") from error
    if not isinstance(raw, dict):
        raise ArtifactError("recovery resolved config must be a mapping")
    cfg = RunConfig.from_dict(raw)
    if cfg.track != "confirmatory_ewt" or cfg.dataset.id != "ewt":
        raise ArtifactError("head-search recovery requires the confirmatory EWT run")
    if cfg.experiment.type != "head_search":
        raise ArtifactError("head-search recovery requires a head_search experiment")
    return cfg


def validate_recovery_identity(
    run_dir: str | Path, cfg: RunConfig, *, migrate_legacy_hash: bool = True
) -> tuple[dict[str, Any], dict[str, str], str]:
    """Validate saved science/manifests and migrate only the old operational hash form."""
    run = Path(run_dir)
    try:
        raw_config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        manifests = json.loads((run / "manifest_refs.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ArtifactError("recovery run identity artifacts are unreadable") from error
    if not isinstance(raw_config, dict) or not isinstance(metadata, dict) or not isinstance(
        manifests, dict
    ):
        raise ArtifactError("recovery run identity artifacts have invalid types")
    if set(manifests) != {"select", "dev", "test"}:
        raise ArtifactError("recovery manifest references must contain select/dev/test")
    if metadata.get("manifest_hashes_hash") not in {None, canonical_hash(manifests)}:
        raise ArtifactError("recovery manifest references disagree with run metadata")

    normalized_hash = canonical_hash(scientific_configuration(raw_config))
    legacy_hash = canonical_hash(raw_config)
    recorded_hash = metadata.get("scientific_config_hash") or metadata.get("config_hash")
    if recorded_hash not in {normalized_hash, legacy_hash}:
        raise ArtifactError("recovery scientific configuration differs from run metadata")
    if recorded_hash == legacy_hash and recorded_hash != normalized_hash and migrate_legacy_hash:
        metadata.setdefault("legacy_scientific_config_hash", recorded_hash)
        metadata["scientific_config_hash"] = normalized_hash
        metadata["config_hash"] = normalized_hash
        atomic_json(run / "run_metadata.json", metadata)
        recorded_hash = normalized_hash
    if recorded_hash != normalized_hash:
        raise ArtifactError("legacy recovery identity must be normalized before new work")
    return metadata, manifests, normalized_hash


def validate_recovery_selection_inputs(
    run_dir: str | Path, cfg: RunConfig
) -> RelationLockSet:
    """Verify immutable select/dev evidence and the complete six-lock bundle."""
    run = Path(run_dir)
    bundle_dir = run / "relation-selection"
    source_path = bundle_dir / "source_artifacts.json"
    if not source_path.is_file():
        raise ArtifactError("recovery requires a validated relation-selection bundle")
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError("relation-selection source identity is unreadable") from error
    stable_hashes = source.get("stable_selection_artifact_hashes", {})
    if set(stable_hashes) != set(SELECTION_SOURCE_FILES):
        raise ArtifactError("relation-selection source hash inventory is incomplete")
    for relative in SELECTION_SOURCE_FILES:
        path = run / relative
        if not path.is_file() or _file_sha256(path) != stable_hashes[relative]:
            raise ArtifactError(f"saved selection evidence differs at {relative}")
    if canonical_hash(stable_hashes) != source.get("stable_selection_artifacts_hash"):
        raise ArtifactError("relation-selection source hashes are internally inconsistent")

    locks = load_relation_locks(bundle_dir, cfg)
    if set(locks.locks) != set(RELATION_NAMES):
        raise ArtifactError("recovery requires complete locks for all six relations")
    return locks


def recovery_status(run_dir: str | Path, cfg: RunConfig | None = None) -> dict[str, Any]:
    """Classify reusable locked evidence separately from permutation readiness."""
    run = Path(run_dir)
    cfg = cfg or load_saved_head_search_config(run)
    validate_recovery_identity(run, cfg)
    locks = validate_recovery_selection_inputs(run, cfg)
    locked_path = run / "instances.parquet"
    if not locked_path.is_file():
        raise ArtifactError("recovery requires the existing locked-head instances.parquet")
    _validate_locked_evidence_sentence_file(locked_path, cfg, locks)
    all_head_path = run / TEST_ALL_HEAD_EVIDENCE
    all_head_ready = all_head_path.is_file() and (run / TEST_ALL_HEAD_METADATA).is_file()
    if all_head_ready:
        saved = validate_test_all_head_metadata(run, cfg)
        expected_heads = {
            (int(layer), int(head)) for layer, head in saved.get("heads", [])
        }
        _validate_test_grid_relations(all_head_path, cfg, expected_heads)
    return {
        "schema_version": RECOVERY_SCHEMA,
        "run_id": run.name,
        "legacy_locked_head_evidence": "valid_and_reusable",
        "locked_head_evidence_sufficient_for_metrics": True,
        "locked_head_evidence_sufficient_for_selection_aware_permutation": False,
        "all_head_test_evidence": "complete" if all_head_ready else "missing",
        "missing_test_grid_required": not all_head_ready,
        "cpu_finalization_ready": all_head_ready,
    }


def score_missing_test_grid(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: str | Path,
    *,
    model_metadata: dict[str, Any] | None = None,
    collect_locked_rows: bool = False,
    reuse_existing_locked: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Score only the missing all-head EWT test grid and atomically consolidate it."""
    run = Path(run_dir)
    metadata, manifests, scientific_hash = validate_recovery_identity(run, cfg)
    locks = validate_recovery_selection_inputs(run, cfg)
    if reuse_existing_locked:
        _validate_locked_evidence_sentence_file(run / "instances.parquet", cfg, locks)

    examples, test_exclusions = load_manifest_examples(cfg, tokenizer, "test")
    example_ids = [str(example.sentence_id) for example in examples]
    if reuse_existing_locked:
        locked_ids = set(
            pd.read_parquet(run / "instances.parquet", columns=["sentence_id"])["sentence_id"]
            .astype(str)
            .unique()
        )
        if set(example_ids) != locked_ids:
            raise ArtifactError("saved locked-head evidence and current test examples differ")
    progress = selection_progress(cfg)
    timestep = round(progress * (cfg.experiment.steps - 1))
    if progress != 0.0 or timestep != 0:
        raise ArtifactError("recovery test grid requires frozen progress 0 and timestep 0")

    store = SentenceCheckpointStore(run)
    state: dict[str, Any] = {"row_count": 0, "heads": set(), "sentence_ids": set()}
    locked_frames: list[pd.DataFrame] = []

    def chunks() -> Iterable[pd.DataFrame]:
        for seed in cfg.experiment.seeds:
            identity = CheckpointIdentity(
                stage=TEST_ALL_HEAD_STAGE,
                seed=seed,
                normalized_progress=progress,
                timestep=timestep,
                heads=None,
            )
            for frame in store.iter_chunks(
                examples,
                identity,
                lambda chunk, start, current_seed=seed: score_attention_heads(
                    model,
                    tokenizer,
                    list(chunk),
                    cfg,
                    role="test",
                    heads=None,
                    normalized_progress=progress,
                    seed=current_seed,
                    sentence_offset=start,
                    total_sentences=len(examples),
                ),
            ):
                narrow = _narrow_permutation_evidence(frame, role="test", cfg=cfg)
                state["row_count"] += len(narrow)
                state["heads"].update(
                    narrow[["layer", "head"]].itertuples(index=False, name=None)
                )
                state["sentence_ids"].update(narrow["sentence_id"].astype(str))
                if collect_locked_rows:
                    locked_frames.append(filter_relation_locked_rows(frame, locks))
                yield narrow

    evidence_path = run / TEST_ALL_HEAD_EVIDENCE
    _atomic_stream_parquet(evidence_path, chunks())
    if state["sentence_ids"] != set(example_ids):
        raise ArtifactError("all-head test evidence omitted one or more test sentences")
    evidence_metadata = {
        "schema_version": RECOVERY_SCHEMA,
        "completion_status": "complete",
        "stage": TEST_ALL_HEAD_STAGE,
        "role": "test",
        "scientific_config_hash": scientific_hash,
        "manifest_hashes": manifests,
        "seeds": list(cfg.experiment.seeds),
        "normalized_progress": progress,
        "timestep": timestep,
        "heads": [list(head) for head in sorted(state["heads"])],
        "sentence_count": len(example_ids),
        "sentence_ids_hash": canonical_hash(sorted(example_ids)),
        "row_count": state["row_count"],
        "parquet_sha256": _file_sha256(evidence_path),
    }
    _validate_test_grid_relations(evidence_path, cfg, set(state["heads"]))
    atomic_json(run / TEST_ALL_HEAD_METADATA, evidence_metadata)

    metadata["recovery_test_grid"] = {
        "schema_version": RECOVERY_SCHEMA,
        "completion_status": "complete",
        "stage": TEST_ALL_HEAD_STAGE,
        "evidence": TEST_ALL_HEAD_EVIDENCE,
        "evidence_sha256": evidence_metadata["parquet_sha256"],
        "model_metadata": model_metadata,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata["completion_status"] = "running"
    metadata.pop("final_artifact_hashes", None)
    atomic_json(run / "run_metadata.json", metadata)
    report = {
        "schema_version": RECOVERY_SCHEMA,
        "completion_status": "complete",
        "stage": TEST_ALL_HEAD_STAGE,
        "select_forward_calls": 0,
        "dev_forward_calls": 0,
        "test_sentences": len(example_ids),
        "seeds": list(cfg.experiment.seeds),
        "estimated_test_forward_calls": len(example_ids) * len(cfg.experiment.seeds),
        "checkpoint_chunk_size": store.chunk_size,
        "all_head_rows": state["row_count"],
        "head_count": len(state["heads"]),
        "evidence": str(evidence_path),
        "legacy_locked_head_evidence_reused_for_metrics": True,
        "legacy_locked_head_evidence_used_for_permutations": False,
    }
    locked = pd.concat(locked_frames, ignore_index=True) if locked_frames else pd.DataFrame()
    return report, locked, test_exclusions


def validate_test_all_head_metadata(
    run_dir: str | Path, cfg: RunConfig
) -> dict[str, Any]:
    """Validate the consolidated narrow all-head test evidence without loading it all."""
    run = Path(run_dir)
    _metadata, manifests, scientific_hash = validate_recovery_identity(run, cfg)
    path = run / TEST_ALL_HEAD_EVIDENCE
    metadata_path = run / TEST_ALL_HEAD_METADATA
    if not path.is_file() or not metadata_path.is_file():
        raise ArtifactError(
            "locked-head-only evidence cannot run the selection-aware permutation; "
            f"missing {TEST_ALL_HEAD_EVIDENCE}"
        )
    try:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError("all-head test evidence metadata is unreadable") from error
    expected = {
        "schema_version": RECOVERY_SCHEMA,
        "completion_status": "complete",
        "stage": TEST_ALL_HEAD_STAGE,
        "role": "test",
        "scientific_config_hash": scientific_hash,
        "manifest_hashes": manifests,
        "seeds": list(cfg.experiment.seeds),
        "normalized_progress": 0.0,
        "timestep": 0,
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ArtifactError("all-head test evidence metadata is scientifically incompatible")
    if saved.get("parquet_sha256") != _file_sha256(path):
        raise ArtifactError("all-head test evidence hash mismatch")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != saved.get("row_count"):
        raise ArtifactError("all-head test evidence row count mismatch")
    _require_parquet_columns(path, set(PERMUTATION_COLUMNS))
    return saved


def finalize_head_search_artifacts(
    cfg: RunConfig,
    run_dir: str | Path,
    *,
    n_permutations: int = 1000,
    checkpoint_interval: int = 50,
) -> dict[str, Any]:
    """Finalize one relation at a time using only narrow on-disk evidence."""
    run = Path(run_dir)
    metadata, _manifests, scientific_hash = validate_recovery_identity(run, cfg)
    locks = validate_recovery_selection_inputs(run, cfg)
    test_metadata = validate_test_all_head_metadata(run, cfg)
    _validate_locked_evidence_sentence_file(run / "instances.parquet", cfg, locks)
    if not (run / "exclusions.parquet").is_file():
        raise ArtifactError("recovery requires the saved exclusions.parquet")

    metadata["completion_status"] = "running"
    metadata.pop("final_artifact_hashes", None)
    atomic_json(run / "run_metadata.json", metadata)

    metrics_parts = []
    per_seed_parts = []
    structural_parts = []
    permutation_rows: list[dict[str, Any]] = []
    permutation_results: dict[str, dict[str, Any]] = {}
    select_sentences: set[str] = set()
    dev_sentences: set[str] = set()
    test_sentences: set[str] = set()
    test_instances: set[str] = set()

    for relation in RELATION_NAMES:
        select = _read_relation_parquet(
            run / "select_instances.parquet", relation, PERMUTATION_COLUMNS
        )
        dev = _read_relation_parquet(run / "dev_instances.parquet", relation, PERMUTATION_COLUMNS)
        test = _read_relation_parquet(run / TEST_ALL_HEAD_EVIDENCE, relation, PERMUTATION_COLUMNS)
        select_heads = _validate_relation_grid(select, cfg, relation=relation, role="select")
        dev_heads = _validate_relation_grid(dev, cfg, relation=relation, role="dev")
        test_heads = _validate_relation_grid(test, cfg, relation=relation, role="test")
        if select_heads != dev_heads or select_heads != test_heads:
            raise ArtifactError(f"all-head evidence grid differs across splits for {relation!r}")
        select_sentences.update(select["sentence_id"].astype(str))
        dev_sentences.update(dev["sentence_id"].astype(str))
        test_sentences.update(test["sentence_id"].astype(str))

        result = selection_aware_permutation(
            select,
            dev,
            test,
            relation=relation,
            top_k=cfg.experiment.scoring.top_k,
            n_permutations=n_permutations,
            seed=42,
            scientific_config_hash=scientific_hash,
            minimum_denominator=MINIMUM_DENOMINATOR,
            checkpoint_path=run / "permutation-checkpoints" / f"{relation}.json",
            resume=True,
            checkpoint_interval=checkpoint_interval,
        )
        if result["completion_status"] != "complete":
            raise ArtifactError(f"permutation run did not complete for relation {relation!r}")
        lock = locks.resolve(relation)
        if (result["observed_selected_layer"], result["observed_selected_head"]) != (
            lock.layer,
            lock.head,
        ):
            raise ArtifactError(
                f"permutation observed protocol disagrees with frozen lock for {relation!r}"
            )
        result["family"] = (
            "primary_separate" if relation == PRIMARY_RELATION else "five_secondaries"
        )
        atomic_json(run / "permutations" / f"{relation}.json", result)
        permutation_results[relation] = result
        permutation_rows.append(
            {
                "relation": relation,
                "family": result["family"],
                "observed_test_accuracy": result["observed_test_accuracy"],
                "raw_p_value": result["p_value"],
                "holm_adjusted_p_value": None,
                "n_permutations": result["n_permutations"],
                "null_mean": result["null_mean"],
                "null_std": result["null_std"],
                "null_definition": result["null_definition"],
                "checkpoint": f"permutation-checkpoints/{relation}.json",
            }
        )

        locked = _read_relation_parquet(run / "instances.parquet", relation, LOCKED_METRIC_COLUMNS)
        _validate_locked_relation_rows(locked, cfg, locks, relation)
        key_columns = ["sentence_id", "instance_id", "seed"]
        all_head_keys = set(test[key_columns].itertuples(index=False, name=None))
        locked_keys = set(locked[key_columns].itertuples(index=False, name=None))
        if locked_keys != all_head_keys:
            raise ArtifactError(
                f"locked and all-head test instances differ for relation {relation!r}"
            )
        test_instances.update(locked["instance_id"].astype(str))
        fixed_offset = {relation: lock.frozen_settings.get("fixed_offset")}
        metrics_parts.append(locked_metrics(locked, fixed_offset))
        per_seed_parts.append(per_seed_metrics(locked))
        structural_parts.append(structural_slices(locked))

        del select, dev, test, locked

    _apply_secondary_holm(permutation_rows)
    if canonical_hash(sorted(test_sentences)) != test_metadata.get("sentence_ids_hash"):
        raise ArtifactError("all-head test sentence identity differs from its metadata")
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_table = _relation_sort(permutation_table)
    _atomic_csv(run / "selection_permutation_results.csv", permutation_table)

    primary = dict(permutation_results[PRIMARY_RELATION])
    primary["holm_adjusted_p_value"] = None
    atomic_json(run / "selection_permutation.json", primary)

    metrics = pd.concat(metrics_parts, ignore_index=True)
    metrics = metrics.merge(
        permutation_table[["relation", "raw_p_value", "holm_adjusted_p_value"]],
        on="relation",
        how="left",
        validate="many_to_one",
    )
    metrics = _deterministic_sort(
        metrics, ("treebank", "relation", "layer", "head", "visibility")
    )
    seed_metrics = _deterministic_sort(
        pd.concat(per_seed_parts, ignore_index=True),
        ("seed", "treebank", "relation", "layer", "head", "visibility"),
    )
    slices = _deterministic_sort(
        pd.concat(structural_parts, ignore_index=True),
        (
            "treebank",
            "relation",
            "layer",
            "head",
            "visibility",
            "slice_dimension",
            "slice_value",
        ),
    )
    _atomic_csv(run / "metrics.csv", metrics)
    _atomic_csv(run / "per_seed_metrics.csv", seed_metrics)
    _atomic_csv(run / "structural_slices.csv", slices)
    write_resolved_lock_manifest(run, locks)
    return {
        "selection_lock": asdict(locks.resolve(PRIMARY_RELATION)),
        "n_select_sentences": len(select_sentences),
        "n_dev_sentences": len(dev_sentences),
        "n_test_sentences": test_metadata["sentence_count"],
        "n_test_instances": len(test_instances),
        "test_heads_exposed": len(locks.heads),
        "test_relation_locks_applied": len(locks.locks),
        "relations": list(RELATION_NAMES),
        "relation_selection_bundle": "relation-selection/relation_selection_bundle.json",
        "secondary_relations": "predefined_and_evaluated_with_their_own_frozen_heads",
        "permutation_null": "within_instance_valid_receiver_full_select_dev_test_protocol",
    }


def complete_cpu_finalization(
    cfg: RunConfig,
    run_dir: str | Path,
    *,
    n_permutations: int = 1000,
    checkpoint_interval: int = 50,
) -> dict[str, Any]:
    """Complete and validate a recovered run without importing or loading a model."""
    run = Path(run_dir)
    if (run / "summary.json").is_file():
        try:
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactError("existing recovery summary is unreadable") from error
        if summary.get("completion_status") == "complete":
            validation = validate_run(run)
            if not validation["valid"]:
                raise ArtifactError("completed recovery run is invalid: " + "; ".join(validation["errors"]))
            return {"details": summary, "validation": validation, "already_complete": True}

    details = finalize_head_search_artifacts(
        cfg,
        run,
        n_permutations=n_permutations,
        checkpoint_interval=checkpoint_interval,
    )
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": "dlmrel-run-v1",
        "completion_status": "complete",
        "capabilities": asdict(cfg.model.capabilities),
        **details,
    }
    if cfg.model.family == "fake":
        summary["fake_cpu_only"] = True
    else:
        model_metadata = (metadata.get("recovery_test_grid") or {}).get("model_metadata")
        if not isinstance(model_metadata, dict):
            raise ArtifactError("recovery test-grid metadata lacks the loaded model identity")
        summary["model_metadata"] = model_metadata
    atomic_json(run / "summary.json", summary)
    metadata.update(
        {
            "completion_status": "complete",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "model_revision": cfg.model.revision,
            "tokenizer_revision": cfg.model.tokenizer_revision,
            "remote_code_revision": cfg.model.remote_code_revision,
        }
    )
    atomic_json(run / "run_metadata.json", metadata)
    metadata["final_artifact_hashes"] = final_artifact_hashes(run)
    atomic_json(run / "run_metadata.json", metadata)
    validation = validate_run(run)
    if not validation["valid"]:
        raise ArtifactError("recovered run failed validation: " + "; ".join(validation["errors"]))
    return {"details": details, "validation": validation, "already_complete": False}


def write_test_all_head_evidence(
    run_dir: str | Path,
    cfg: RunConfig,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Write deterministic fake/test all-head evidence with the production contract."""
    run = Path(run_dir)
    _metadata, manifests, scientific_hash = validate_recovery_identity(run, cfg)
    narrow = _narrow_permutation_evidence(frame, role="test", cfg=cfg)
    path = run / TEST_ALL_HEAD_EVIDENCE
    _atomic_stream_parquet(path, [narrow])
    heads = sorted(narrow[["layer", "head"]].drop_duplicates().itertuples(index=False, name=None))
    sentence_ids = sorted(narrow["sentence_id"].astype(str).unique())
    _validate_test_grid_relations(path, cfg, set(heads))
    metadata = {
        "schema_version": RECOVERY_SCHEMA,
        "completion_status": "complete",
        "stage": TEST_ALL_HEAD_STAGE,
        "role": "test",
        "scientific_config_hash": scientific_hash,
        "manifest_hashes": manifests,
        "seeds": list(cfg.experiment.seeds),
        "normalized_progress": 0.0,
        "timestep": 0,
        "heads": [list(head) for head in heads],
        "sentence_count": len(sentence_ids),
        "sentence_ids_hash": canonical_hash(sentence_ids),
        "row_count": len(narrow),
        "parquet_sha256": _file_sha256(path),
    }
    atomic_json(run / TEST_ALL_HEAD_METADATA, metadata)
    return metadata


def _narrow_permutation_evidence(
    frame: pd.DataFrame, *, role: str, cfg: RunConfig
) -> pd.DataFrame:
    if missing := set(PERMUTATION_COLUMNS) - set(frame):
        raise ArtifactError(f"{role} all-head evidence missing columns: {sorted(missing)}")
    narrow = frame.loc[:, PERMUTATION_COLUMNS].copy()
    if narrow.empty or narrow.isna().any().any():
        raise ArtifactError(f"{role} all-head permutation evidence is empty or contains missing values")
    for column in ("sentence_id", "instance_id", "role", "relation"):
        narrow[column] = narrow[column].astype(str)
    for column in (
        "seed",
        "timestep",
        "layer",
        "head",
        "predicted_word_idx",
        "gold_receiver_word_idx",
        "attender_word_idx",
        "sentence_length_words",
        "n_candidate_words",
        "correct",
    ):
        narrow[column] = narrow[column].astype("int64")
    narrow["normalized_progress"] = narrow["normalized_progress"].astype("float64")
    if set(narrow["role"]) != {role}:
        raise ArtifactError(f"{role} all-head evidence contains another split role")
    expected_progress = selection_progress(cfg)
    if set(narrow["normalized_progress"]) != {expected_progress}:
        raise ArtifactError(f"{role} all-head evidence uses an incompatible progress point")
    if set(narrow["timestep"]) != {round(expected_progress * (cfg.experiment.steps - 1))}:
        raise ArtifactError(f"{role} all-head evidence uses an incompatible timestep")
    return narrow.sort_values(list(PERMUTATION_SORT), kind="mergesort").reset_index(drop=True)


def _read_relation_parquet(
    path: Path, relation: str, columns: Iterable[str]
) -> pd.DataFrame:
    requested = list(dict.fromkeys(columns))
    _require_parquet_columns(path, set(requested))
    try:
        frame = pd.read_parquet(
            path,
            columns=requested,
            filters=[("relation", "==", relation)],
        )
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactError(f"could not read relation evidence from {path.name}") from error
    if frame.empty:
        raise ArtifactError(f"{path.name} has no evidence for relation {relation!r}")
    return frame


def _validate_relation_grid(
    frame: pd.DataFrame, cfg: RunConfig, *, relation: str, role: str
) -> set[tuple[int, int]]:
    if set(frame["relation"].astype(str)) != {relation}:
        raise ArtifactError(f"{role} evidence mixes relations while processing {relation!r}")
    if set(frame["role"].astype(str)) != {role}:
        raise ArtifactError(f"{role} evidence contains another split role")
    if set(frame["seed"].astype(int)) != set(cfg.experiment.seeds):
        raise ArtifactError(f"{role} evidence does not contain exactly the frozen seeds")
    progress = selection_progress(cfg)
    if set(frame["normalized_progress"].astype(float)) != {progress}:
        raise ArtifactError(f"{role} evidence uses an incompatible progress point")
    if set(frame["timestep"].astype(int)) != {round(progress * (cfg.experiment.steps - 1))}:
        raise ArtifactError(f"{role} evidence uses an incompatible timestep")
    identity = ["sentence_id", "instance_id", "seed", "layer", "head"]
    if frame.duplicated(identity).any():
        raise ArtifactError(f"{role} evidence contains duplicate all-head rows")
    heads = set(frame[["layer", "head"]].itertuples(index=False, name=None))
    if not heads:
        raise ArtifactError(f"{role} evidence contains no model heads")
    expected_per_seed = len(heads)
    per_seed = frame.groupby(["sentence_id", "instance_id", "seed"], observed=True).size()
    if not per_seed.eq(expected_per_seed).all():
        raise ArtifactError(f"{role} evidence omits one or more model heads")
    seed_counts = frame.groupby(["sentence_id", "instance_id"], observed=True)["seed"].nunique()
    if not seed_counts.eq(len(cfg.experiment.seeds)).all():
        raise ArtifactError(f"{role} evidence omits one or more seeds")
    return {(int(layer), int(head)) for layer, head in heads}


def _validate_test_grid_relations(
    path: Path, cfg: RunConfig, expected_heads: set[tuple[int, int]]
) -> None:
    for relation in RELATION_NAMES:
        frame = _read_relation_parquet(path, relation, PERMUTATION_COLUMNS)
        actual = _validate_relation_grid(frame, cfg, relation=relation, role="test")
        if actual != expected_heads:
            raise ArtifactError(f"test all-head grid differs for relation {relation!r}")


def _validate_locked_evidence_sentence_file(
    path: Path, cfg: RunConfig, locks: RelationLockSet
) -> None:
    _require_parquet_columns(
        path,
        {
            "sentence_id",
            "instance_id",
            "role",
            "relation",
            "seed",
            "timestep",
            "normalized_progress",
            "layer",
            "head",
        },
    )
    for relation in RELATION_NAMES:
        columns = (
            "sentence_id",
            "instance_id",
            "role",
            "relation",
            "seed",
            "timestep",
            "normalized_progress",
            "layer",
            "head",
        )
        frame = _read_relation_parquet(path, relation, columns)
        _validate_locked_relation_rows(frame, cfg, locks, relation)


def _validate_locked_relation_rows(
    frame: pd.DataFrame, cfg: RunConfig, locks: RelationLockSet, relation: str
) -> None:
    if set(frame["role"].astype(str)) != {"test"}:
        raise ArtifactError("locked-head evidence contains a non-test split")
    if set(frame["seed"].astype(int)) != set(cfg.experiment.seeds):
        raise ArtifactError("locked-head evidence does not contain the frozen seeds")
    if set(frame["normalized_progress"].astype(float)) != {0.0} or set(
        frame["timestep"].astype(int)
    ) != {0}:
        raise ArtifactError("locked-head evidence is not the frozen progress-0 test")
    lock = locks.resolve(relation)
    heads = set(frame[["layer", "head"]].itertuples(index=False, name=None))
    if heads != {(lock.layer, lock.head)}:
        raise ArtifactError(f"locked test rows do not match the frozen lock for {relation!r}")
    identity = ["sentence_id", "instance_id", "seed", "relation", "layer", "head"]
    if frame.duplicated(identity).any():
        raise ArtifactError("locked-head evidence contains duplicate rows")


def _apply_secondary_holm(rows: list[dict[str, Any]]) -> None:
    secondaries = [row for row in rows if row["relation"] in SECONDARY_RELATIONS]
    if {row["relation"] for row in secondaries} != set(SECONDARY_RELATIONS):
        raise ArtifactError("Holm family does not contain the five predefined secondaries")
    secondaries.sort(
        key=lambda row: (row["raw_p_value"], SECONDARY_RELATIONS.index(row["relation"]))
    )
    running = 0.0
    family_size = len(SECONDARY_RELATIONS)
    for rank, row in enumerate(secondaries, start=1):
        running = max(
            running,
            min(1.0, (family_size - rank + 1) * float(row["raw_p_value"])),
        )
        row["holm_adjusted_p_value"] = running


def _relation_sort(frame: pd.DataFrame) -> pd.DataFrame:
    order = {relation: index for index, relation in enumerate(RELATION_NAMES)}
    return (
        frame.assign(_relation_order=frame["relation"].map(order))
        .sort_values("_relation_order", kind="mergesort")
        .drop(columns="_relation_order")
        .reset_index(drop=True)
    )


def _deterministic_sort(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame]
    return frame.sort_values(available, kind="mergesort").reset_index(drop=True)


def _atomic_stream_parquet(path: Path, frames: Iterable[pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for frame in frames:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
        if writer is None:
            raise ArtifactError("all-head evidence produced no rows")
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _require_parquet_columns(path: Path, required: set[str]) -> None:
    if not path.is_file():
        raise ArtifactError(f"missing required recovery evidence: {path.name}")
    try:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, pa.ArrowException) as error:
        raise ArtifactError(f"recovery evidence is not valid Parquet: {path.name}") from error
    if missing := required - columns:
        raise ArtifactError(f"{path.name} is missing required columns: {sorted(missing)}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

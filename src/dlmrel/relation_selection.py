"""Leakage-safe, relation-specific head-selection bundles from select/dev evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .artifacts import (
    LOCK_SCHEMA,
    RUN_SCHEMA,
    ArtifactError,
    SelectionLock,
    atomic_json,
    canonical_hash,
    scientific_configuration,
)
from .config import RELATION_NAMES, RunConfig
from .controls import fit_fixed_offset
from .experiments.shared import (
    aggregate_head_scores,
    selection_aware_permutation,
    selection_progress,
)

BUNDLE_SCHEMA = "dlmrel-relation-selection-bundle-v1"
VALIDATION_SCHEMA = "dlmrel-relation-selection-validation-v1"
PROTOCOL_SCHEMA = "dlmrel-relation-head-selection-v1"
PRIMARY_RELATION = "object_to_verb"
SECONDARY_RELATIONS = tuple(relation for relation in RELATION_NAMES if relation != PRIMARY_RELATION)
REQUIRED_SEEDS = (42, 43, 44)
TOP_K = 5
MINIMUM_DENOMINATOR = 25
PERMUTATION_SEED = 42
DEFAULT_PERMUTATIONS = 1000

RAW_REQUIRED_COLUMNS = {
    "sentence_id",
    "instance_id",
    "role",
    "seed",
    "relation",
    "layer",
    "head",
    "predicted_word_idx",
    "gold_receiver_word_idx",
    "correct",
    "normalized_progress",
    "timestep",
    "treebank",
    "signed_distance",
}
SCORE_REQUIRED_COLUMNS = {"relation", "layer", "head", "accuracy", "n_total", "n_correct"}
STABLE_SOURCE_FILES = (
    "config.resolved.yaml",
    "manifest_refs.json",
    "select_all_head_scores.csv",
    "dev_all_head_scores.csv",
    "select_instances.parquet",
    "dev_instances.parquet",
)
FORBIDDEN_SOURCE_READS = (
    "instances.parquet",
    "metrics.csv",
    "per_seed_metrics.csv",
    "summary.json",
    "selection_permutation.json",
    "structural_slices.csv",
)


@dataclass(frozen=True)
class SourceEvidence:
    run_dir: Path
    config: RunConfig
    config_raw: dict[str, Any]
    metadata: dict[str, Any]
    manifests: dict[str, str]
    select_scores: pd.DataFrame
    dev_scores: pd.DataFrame
    select_rows: pd.DataFrame
    dev_rows: pd.DataFrame
    stable_hashes: dict[str, str]
    all_read_hashes: dict[str, str]
    scientific_config_hash: str


@dataclass(frozen=True)
class BundleBuild:
    output_dir: Path
    bundle: dict[str, Any]
    primary_lock: SelectionLock | None
    primary_select_candidates: pd.DataFrame
    primary_dev_candidates: pd.DataFrame


def derive_relation_selection_bundle(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    require_complete: bool = True,
    allow_source_output: bool = False,
    allow_existing: bool = False,
    n_permutations: int = DEFAULT_PERMUTATIONS,
) -> BundleBuild:
    """Derive six independent locks without reading any locked-test artifact."""
    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    _validate_paths(source, output, allow_source_output=allow_source_output)
    evidence = _load_and_validate_source(source, require_complete=require_complete)
    source_hash = canonical_hash(evidence.stable_hashes)
    if output.exists():
        if not allow_existing:
            raise ArtifactError(f"refusing to overwrite relation-selection output: {output}")
        return _load_existing_bundle(output, evidence, source_hash)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        built = _write_bundle(temporary, evidence, source_hash, n_permutations=n_permutations)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return BundleBuild(
        output_dir=output,
        bundle=built.bundle,
        primary_lock=built.primary_lock,
        primary_select_candidates=built.primary_select_candidates,
        primary_dev_candidates=built.primary_dev_candidates,
    )


def install_primary_aliases(run_dir: str | Path, build: BundleBuild) -> None:
    """Install legacy primary files atomically without changing their contents."""
    if build.primary_lock is None:
        raise ArtifactError("primary relation has insufficient evidence; no test head may be exposed")
    run = Path(run_dir)
    build.primary_lock.write_once(run / "selection_lock.json")
    _write_csv_once(run / "select_candidates.csv", build.primary_select_candidates)
    _write_csv_once(run / "dev_candidates.csv", build.primary_dev_candidates)
    bundle_alias = build.output_dir / "selection_lock.json"
    if (run / "selection_lock.json").read_bytes() != bundle_alias.read_bytes():
        raise ArtifactError("primary selection-lock aliases are not byte-equivalent")


def _write_bundle(
    output: Path,
    evidence: SourceEvidence,
    source_hash: str,
    *,
    n_permutations: int,
) -> BundleBuild:
    (output / "locks").mkdir(parents=True)
    (output / "candidates").mkdir()
    relation_records: dict[str, dict[str, Any]] = {}
    locks: dict[str, SelectionLock] = {}
    candidate_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for relation in RELATION_NAMES:
        select_candidates, dev_candidates, result = _select_relation(evidence, relation)
        select_path = output / "candidates" / f"{relation}_select.csv"
        dev_path = output / "candidates" / f"{relation}_dev.csv"
        select_candidates.to_csv(select_path, index=False)
        dev_candidates.to_csv(dev_path, index=False)
        candidate_hash = canonical_hash(
            {"select_csv_sha256": _sha256(select_path), "dev_csv_sha256": _sha256(dev_path)}
        )
        candidate_frames[relation] = (select_candidates, dev_candidates)
        record = {
            **result,
            "role": _relation_role(relation),
            "select_candidates": f"candidates/{relation}_select.csv",
            "dev_candidates": f"candidates/{relation}_dev.csv",
            "candidate_table_hash": candidate_hash,
            "lock": None,
        }
        if result["status"] == "selected":
            lock = _make_lock(evidence, relation, result, candidate_hash)
            lock_path = output / "locks" / f"{relation}.json"
            lock.write_once(lock_path)
            locks[relation] = lock
            record["lock"] = f"locks/{relation}.json"
            record["selection_lock_hash"] = canonical_hash(asdict(lock))
        relation_records[relation] = record

    primary_lock = locks.get(PRIMARY_RELATION)
    if primary_lock is not None:
        primary_lock.write_once(output / "selection_lock.json")
        if (output / "selection_lock.json").read_bytes() != (
            output / "locks" / f"{PRIMARY_RELATION}.json"
        ).read_bytes():
            raise ArtifactError("bundled primary alias is not byte-equivalent to its relation lock")

    permutation_rows = _permutation_results(
        evidence,
        relation_records,
        n_permutations=n_permutations,
    )
    pd.DataFrame(permutation_rows).to_csv(output / "permutation_results.csv", index=False)
    source_record = {
        "schema_version": BUNDLE_SCHEMA,
        "source_run_identity": _source_identity(evidence),
        "stable_selection_artifact_hashes": evidence.stable_hashes,
        "stable_selection_artifacts_hash": source_hash,
        "all_allowed_read_artifact_hashes": evidence.all_read_hashes,
        "forbidden_artifacts_read": [],
        "forbidden_source_artifacts": list(FORBIDDEN_SOURCE_READS),
        "test_outcomes_used": False,
    }
    atomic_json(output / "source_artifacts.json", source_record)
    shutil.copyfile(evidence.run_dir / "config.resolved.yaml", output / "config.resolved.yaml")
    protocol = _protocol(evidence.config, n_permutations)
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "primary_relation": PRIMARY_RELATION,
        "secondary_relations": list(SECONDARY_RELATIONS),
        "relations": relation_records,
        "selection_protocol": protocol,
        "permutation_results": "permutation_results.csv",
        "source_artifacts": "source_artifacts.json",
        "source_selection_artifacts_hash": source_hash,
        "primary_alias": "selection_lock.json" if primary_lock is not None else None,
        "secondary_claim_status": "predefined_selected_not_tested",
        "test_outcomes_used": False,
    }
    atomic_json(output / "relation_selection_bundle.json", bundle)
    selected = [relation for relation, record in relation_records.items() if record["status"] == "selected"]
    insufficient = [
        relation for relation, record in relation_records.items() if record["status"] != "selected"
    ]
    atomic_json(
        output / "run_metadata.json",
        {
            "schema_version": BUNDLE_SCHEMA,
            "completion_status": "complete",
            "derived_at": _now(),
            "source_run_identity": _source_identity(evidence),
            "source_selection_artifacts_hash": source_hash,
            "scientific_config_hash": evidence.scientific_config_hash,
        },
    )
    atomic_json(
        output / "summary.json",
        {
            "schema_version": BUNDLE_SCHEMA,
            "completion_status": "complete",
            "selected_relations": selected,
            "insufficient_evidence_relations": insufficient,
            "primary_relation": PRIMARY_RELATION,
            "primary_status": relation_records[PRIMARY_RELATION]["status"],
            "secondary_relations_are_not_confirmatory_test_results": True,
            "model_inference_performed": False,
            "test_artifacts_read": False,
        },
    )
    validation = _validate_written_bundle(output, source_hash)
    atomic_json(output / "validation.json", validation)
    if not validation["valid"]:
        raise ArtifactError(
            "generated relation-selection bundle is invalid: " + "; ".join(validation["errors"])
        )
    primary_frames = candidate_frames[PRIMARY_RELATION]
    return BundleBuild(output, bundle, primary_lock, primary_frames[0], primary_frames[1])


def _load_and_validate_source(source: Path, *, require_complete: bool) -> SourceEvidence:
    if not source.is_dir():
        raise ArtifactError(f"source run directory does not exist: {source}")
    required = list(STABLE_SOURCE_FILES) + ["run_metadata.json"]
    if require_complete:
        required.append("validation.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ArtifactError("source run is missing required selection evidence: " + ", ".join(missing))

    config_path = source / "config.resolved.yaml"
    config_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_raw, dict):
        raise ArtifactError("source resolved config must be a mapping")
    config = RunConfig.from_dict(config_raw)
    metadata = _read_json(source / "run_metadata.json")
    manifests = _read_json(source / "manifest_refs.json")
    if metadata.get("schema_version") != RUN_SCHEMA or not isinstance(
        metadata.get("started_at"), str
    ):
        raise ArtifactError("source run metadata schema or start time is invalid")
    if require_complete:
        validation = _read_json(source / "validation.json")
        if (
            metadata.get("completion_status") != "complete"
            or validation.get("schema_version") != RUN_SCHEMA
            or not validation.get("valid")
        ):
            raise ArtifactError("offline derivation requires a complete, validated source run")
        recorded_revisions = {
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "remote_code_revision": config.model.remote_code_revision,
        }
        if any(metadata.get(key) != value for key, value in recorded_revisions.items()):
            raise ArtifactError("source run metadata revisions do not match the resolved config")
    _validate_protocol_config(config)
    _validate_manifest_refs(metadata, manifests)
    scientific_hash = canonical_hash(
        scientific_configuration(
            config_raw,
            resolved_selection_lock_hash=metadata.get("selection_lock_hash"),
        )
    )
    recorded_hash = metadata.get("scientific_config_hash", metadata.get("config_hash"))
    if recorded_hash != scientific_hash:
        raise ArtifactError("source scientific configuration hash does not match its resolved config")

    select_scores = pd.read_csv(source / "select_all_head_scores.csv")
    dev_scores = pd.read_csv(source / "dev_all_head_scores.csv")
    select_rows = pd.read_parquet(source / "select_instances.parquet")
    dev_rows = pd.read_parquet(source / "dev_instances.parquet")
    _validate_evidence_frames(config, select_scores, dev_scores, select_rows, dev_rows)

    stable_hashes = {name: _sha256(source / name) for name in STABLE_SOURCE_FILES}
    allowed = list(STABLE_SOURCE_FILES) + ["run_metadata.json"]
    if require_complete:
        allowed.append("validation.json")
    all_read_hashes = {name: _sha256(source / name) for name in allowed}
    return SourceEvidence(
        run_dir=source,
        config=config,
        config_raw=config_raw,
        metadata=metadata,
        manifests=manifests,
        select_scores=select_scores,
        dev_scores=dev_scores,
        select_rows=select_rows,
        dev_rows=dev_rows,
        stable_hashes=stable_hashes,
        all_read_hashes=all_read_hashes,
        scientific_config_hash=scientific_hash,
    )


def _validate_paths(source: Path, output: Path, *, allow_source_output: bool) -> None:
    if source == output:
        raise ArtifactError("relation-selection output must differ from the source run")
    inside_source = source in output.parents
    expected_internal = output == source / "relation-selection"
    if inside_source and not (allow_source_output and expected_internal):
        raise ArtifactError("offline relation-selection output must be outside the completed source run")


def _validate_protocol_config(config: RunConfig) -> None:
    experiment = config.experiment
    scoring = experiment.scoring
    if config.track != "confirmatory_ewt" or config.dataset.id != "ewt":
        raise ArtifactError("relation-specific selection requires the confirmatory EWT source run")
    if experiment.type != "head_search":
        raise ArtifactError("source experiment must be a head search")
    if tuple(experiment.seeds) != REQUIRED_SEEDS:
        raise ArtifactError(f"source seeds must be exactly {list(REQUIRED_SEEDS)}")
    if scoring.top_k != TOP_K or scoring.primary_relation != PRIMARY_RELATION:
        raise ArtifactError("source scoring must use top_k=5 and object_to_verb as primary")
    for label, revision in (
        ("model", config.model.revision),
        ("tokenizer", config.model.tokenizer_revision),
    ):
        if not revision or revision in {"main", "master", "unknown"}:
            raise ArtifactError(f"source {label} revision is not immutable")
    if config.model.remote_code_revision in {"main", "master", "unknown"}:
        raise ArtifactError("source remote-code revision is not immutable")


def _validate_manifest_refs(metadata: dict[str, Any], manifests: dict[str, str]) -> None:
    if set(manifests) != {"select", "dev", "test"}:
        raise ArtifactError("source manifest references must contain exactly select, dev, and test")
    if any(not isinstance(value, str) or not value for value in manifests.values()):
        raise ArtifactError("source manifest hashes must be non-empty strings")
    if len(set(manifests.values())) != 3:
        raise ArtifactError("source select/dev/test manifest hashes must be distinct")
    if metadata.get("manifest_hashes_hash") != canonical_hash(manifests):
        raise ArtifactError("source manifest reference hash is inconsistent with run metadata")


def _validate_evidence_frames(
    config: RunConfig,
    select_scores: pd.DataFrame,
    dev_scores: pd.DataFrame,
    select_rows: pd.DataFrame,
    dev_rows: pd.DataFrame,
) -> None:
    for role, scores, rows in (
        ("select", select_scores, select_rows),
        ("dev", dev_scores, dev_rows),
    ):
        missing_scores = SCORE_REQUIRED_COLUMNS - set(scores)
        missing_rows = RAW_REQUIRED_COLUMNS - set(rows)
        if missing_scores or missing_rows:
            raise ArtifactError(
                f"{role} evidence missing columns: "
                f"scores={sorted(missing_scores)}, instances={sorted(missing_rows)}"
            )
        if rows.empty:
            raise ArtifactError(f"{role} instance evidence is empty")
        if set(rows["role"].dropna().astype(str)) != {role}:
            raise ArtifactError(f"{role} evidence contains the wrong role label")
        _validate_relation_labels(rows, role)
        duplicate_keys = ["sentence_id", "instance_id", "seed", "relation", "layer", "head"]
        if rows.duplicated(duplicate_keys).any():
            raise ArtifactError(f"{role} evidence has duplicate instance/seed/relation/head rows")
        _validate_seed_and_head_grid(rows, role)
        expected_correct = (rows["predicted_word_idx"] == rows["gold_receiver_word_idx"]).astype(int)
        if not np.array_equal(rows["correct"].astype(int).to_numpy(), expected_correct.to_numpy()):
            raise ArtifactError(f"{role} correct labels disagree with predicted and gold receivers")
        consistency = rows.groupby(["sentence_id", "instance_id", "relation"], observed=True).agg(
            gold_values=("gold_receiver_word_idx", "nunique"),
            distance_values=("signed_distance", "nunique"),
        )
        if (consistency > 1).any().any():
            raise ArtifactError(f"{role} relation-instance metadata changes across heads or seeds")
        if set(rows["treebank"].dropna().astype(str)) != {config.dataset.treebank}:
            raise ArtifactError(f"{role} evidence treebank does not match the source config")
        expected_progress = selection_progress(config)
        expected_timestep = round(expected_progress * (config.experiment.steps - 1))
        if not np.allclose(rows["normalized_progress"].astype(float), expected_progress):
            raise ArtifactError(f"{role} evidence uses the wrong normalized selection progress")
        if set(rows["timestep"].astype(int)) != {expected_timestep}:
            raise ArtifactError(f"{role} evidence uses the wrong selection timestep")
        _validate_aggregate(scores, rows, role)
    select_grid = _head_grid(select_rows)
    if select_grid != _head_grid(dev_rows):
        raise ArtifactError("select and dev evidence do not cover the same all-head grid")
    overlap = set(select_rows["sentence_id"].astype(str)) & set(dev_rows["sentence_id"].astype(str))
    if overlap:
        raise ArtifactError("select and dev evidence overlap by sentence_id")


def _validate_relation_labels(rows: pd.DataFrame, role: str) -> None:
    labels = set(rows["relation"].dropna().astype(str))
    unexpected = labels - set(RELATION_NAMES)
    if unexpected:
        raise ArtifactError(f"{role} evidence contains unexpected relations: {sorted(unexpected)}")
    for relation in labels:
        relation_seeds = set(rows.loc[rows["relation"] == relation, "seed"].astype(int))
        if relation_seeds != set(REQUIRED_SEEDS):
            raise ArtifactError(f"{role} relation {relation!r} does not contain all three frozen seeds")


def _validate_seed_and_head_grid(rows: pd.DataFrame, role: str) -> None:
    if set(rows["seed"].astype(int)) != set(REQUIRED_SEEDS):
        raise ArtifactError(f"{role} evidence seeds must be exactly {list(REQUIRED_SEEDS)}")
    for column in ("layer", "head"):
        numeric = pd.to_numeric(rows[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ArtifactError(f"{role} evidence has invalid {column} identifiers")
    grid = _head_grid(rows)
    layers = sorted({layer for layer, _head in grid})
    heads = sorted({head for _layer, head in grid})
    expected = {(layer, head) for layer in range(max(layers) + 1) for head in range(max(heads) + 1)}
    if layers != list(range(max(layers) + 1)) or heads != list(range(max(heads) + 1)) or grid != expected:
        raise ArtifactError(f"{role} evidence is not a complete contiguous model head grid")
    for seed in REQUIRED_SEEDS:
        if _head_grid(rows[rows["seed"] == seed]) != grid:
            raise ArtifactError(f"{role} seed {seed} does not cover the configured all-head grid")
    instance_coverage = rows.groupby(
        ["sentence_id", "instance_id", "seed", "relation"], observed=True
    ).size()
    if not (instance_coverage == len(grid)).all():
        raise ArtifactError(f"{role} relation instances do not each cover the full model head grid")


def _validate_aggregate(scores: pd.DataFrame, rows: pd.DataFrame, role: str) -> None:
    if scores.duplicated(["relation", "layer", "head"]).any():
        raise ArtifactError(f"{role} aggregate head scores contain duplicate heads")
    computed = aggregate_head_scores(rows).sort_values(["relation", "layer", "head"]).reset_index(drop=True)
    recorded = (
        scores[list(computed.columns)]
        .sort_values(["relation", "layer", "head"])
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(
            recorded,
            computed,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ArtifactError(f"{role} aggregate scores are inconsistent with instance evidence") from exc


def _select_relation(
    evidence: SourceEvidence, relation: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    select_all = evidence.select_scores[evidence.select_scores["relation"] == relation].copy()
    select_eligible = select_all[select_all["n_total"] >= MINIMUM_DENOMINATOR].copy()
    select_eligible = select_eligible.sort_values(
        ["accuracy", "n_total", "layer", "head"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    select_candidates = select_eligible.head(TOP_K).copy()
    select_candidates["select_rank"] = range(1, len(select_candidates) + 1)
    select_candidates = _add_per_seed_evidence(
        select_candidates,
        evidence.select_rows[evidence.select_rows["relation"] == relation],
    )
    keys = ["relation", "layer", "head"]
    dev_all = evidence.dev_scores[evidence.dev_scores["relation"] == relation].copy()
    if select_candidates.empty:
        dev_candidates = _empty_dev_candidates(select_candidates)
        return select_candidates, dev_candidates, {
            "status": "insufficient_evidence",
            "reason": "no_select_head_meets_minimum_denominator",
            "counts": _counts(select_all, select_eligible, dev_all, 0, 0),
            "stability": _stability(pd.DataFrame()),
        }

    dev_candidates = select_candidates[keys + ["select_rank"]].merge(
        dev_all,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    dev_candidates["dev_eligible"] = dev_candidates["n_total"].fillna(0) >= MINIMUM_DENOMINATOR
    dev_candidates = _add_per_seed_evidence(
        dev_candidates,
        evidence.dev_rows[evidence.dev_rows["relation"] == relation],
    )
    eligible_dev = dev_candidates[dev_candidates["dev_eligible"]].copy()
    eligible_dev = eligible_dev.sort_values(
        ["accuracy", "n_total", "layer", "head"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible_dev["dev_rank"] = range(1, len(eligible_dev) + 1)
    dev_candidates = dev_candidates.merge(
        eligible_dev[keys + ["dev_rank"]], on=keys, how="left", validate="one_to_one"
    ).sort_values("select_rank", kind="mergesort").reset_index(drop=True)
    if eligible_dev.empty:
        return select_candidates, dev_candidates, {
            "status": "insufficient_evidence",
            "reason": "no_select_top_k_candidate_meets_dev_minimum_denominator",
            "counts": _counts(select_all, select_eligible, dev_all, len(select_candidates), 0),
            "stability": _stability(dev_candidates),
        }
    winner = eligible_dev.iloc[0]
    winner_select = select_candidates[
        (select_candidates["layer"] == winner["layer"])
        & (select_candidates["head"] == winner["head"])
    ].iloc[0]
    return select_candidates, dev_candidates, {
        "status": "selected",
        "reason": None,
        "winner": {
            "layer": int(winner["layer"]),
            "head": int(winner["head"]),
            "select_accuracy": float(winner_select["accuracy"]),
            "select_denominator": int(winner_select["n_total"]),
            "select_rank": int(winner_select["select_rank"]),
            "dev_accuracy": float(winner["accuracy"]),
            "dev_denominator": int(winner["n_total"]),
            "dev_rank": int(winner["dev_rank"]),
        },
        "counts": _counts(
            select_all, select_eligible, dev_all, len(select_candidates), len(eligible_dev)
        ),
        "stability": _stability(dev_candidates),
    }


def _add_per_seed_evidence(candidates: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    if result.empty:
        for seed in REQUIRED_SEEDS:
            result[f"seed_{seed}_accuracy"] = pd.Series(dtype=float)
            result[f"seed_{seed}_n_total"] = pd.Series(dtype="int64")
            result[f"seed_{seed}_n_correct"] = pd.Series(dtype="int64")
        return result
    per_seed = rows.groupby(["relation", "layer", "head", "seed"], as_index=False).agg(
        seed_accuracy=("correct", "mean"),
        seed_n_total=("correct", "size"),
        seed_n_correct=("correct", "sum"),
    )
    keys = ["relation", "layer", "head"]
    for seed in REQUIRED_SEEDS:
        current = per_seed[per_seed["seed"] == seed].drop(columns="seed").rename(
            columns={
                "seed_accuracy": f"seed_{seed}_accuracy",
                "seed_n_total": f"seed_{seed}_n_total",
                "seed_n_correct": f"seed_{seed}_n_correct",
            }
        )
        result = result.merge(current, on=keys, how="left", validate="one_to_one")
    return result


def _empty_dev_candidates(select_candidates: pd.DataFrame) -> pd.DataFrame:
    columns = list(select_candidates.columns) + ["dev_eligible", "dev_rank"]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def _counts(
    select_all: pd.DataFrame,
    select_eligible: pd.DataFrame,
    dev_all: pd.DataFrame,
    select_top_k: int,
    dev_eligible: int,
) -> dict[str, int]:
    return {
        "select_heads_total": int(len(select_all)),
        "select_heads_meeting_minimum": int(len(select_eligible)),
        "select_top_k_count": int(select_top_k),
        "select_max_denominator": int(select_all["n_total"].max()) if not select_all.empty else 0,
        "dev_heads_total_in_source": int(len(dev_all)),
        "dev_candidates_meeting_minimum": int(dev_eligible),
        "dev_max_denominator": int(dev_all["n_total"].max()) if not dev_all.empty else 0,
    }


def _stability(dev_candidates: pd.DataFrame) -> dict[str, Any]:
    if "dev_eligible" not in dev_candidates:
        eligible = pd.DataFrame()
    else:
        eligible = dev_candidates[dev_candidates["dev_eligible"].fillna(False).astype(bool)].copy()
    if eligible.empty:
        return {
            "n_ranked_candidates": 0,
            "same_top_head_on_select_and_dev": None,
            "winner_select_rank": None,
            "select_dev_spearman": None,
        }
    dev_sorted = eligible.sort_values("dev_rank")
    winner_select_rank = int(dev_sorted.iloc[0]["select_rank"])
    correlation = None
    if len(eligible) >= 2:
        value = eligible["select_rank"].corr(eligible["dev_rank"], method="spearman")
        correlation = None if pd.isna(value) else float(value)
    return {
        "n_ranked_candidates": int(len(eligible)),
        "same_top_head_on_select_and_dev": winner_select_rank == 1,
        "winner_select_rank": winner_select_rank,
        "select_dev_spearman": correlation,
    }


def _make_lock(
    evidence: SourceEvidence,
    relation: str,
    result: dict[str, Any],
    candidate_hash: str,
) -> SelectionLock:
    winner = result["winner"]
    relation_rows = evidence.select_rows[evidence.select_rows["relation"] == relation]
    offsets = (
        relation_rows[["sentence_id", "instance_id", "signed_distance"]]
        .drop_duplicates()["signed_distance"]
        .astype(int)
        .tolist()
    )
    config = evidence.config
    settings = {
        "protocol_schema_version": PROTOCOL_SCHEMA,
        "selection_status": "selected",
        "relation_role": _relation_role(relation),
        "metric": "receiver_span_top1_accuracy",
        "top_k": TOP_K,
        "minimum_denominator": MINIMUM_DENOMINATOR,
        "select_accuracy": winner["select_accuracy"],
        "select_denominator": winner["select_denominator"],
        "select_rank": winner["select_rank"],
        "dev_accuracy": winner["dev_accuracy"],
        "dev_denominator": winner["dev_denominator"],
        "dev_rank": winner["dev_rank"],
        "seeds": list(REQUIRED_SEEDS),
        "model_name": config.model.name,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "remote_code_revision": config.model.remote_code_revision,
        "treebank": config.dataset.treebank,
        "dataset_release": config.dataset.release,
        "dataset_revision": config.dataset.revision,
        "source_run_identity": _source_identity(evidence),
        "source_selection_artifact_hashes": evidence.stable_hashes,
        "source_selection_artifacts_hash": canonical_hash(evidence.stable_hashes),
        "candidate_table_hash": candidate_hash,
        "steps": config.experiment.steps,
        "selection_progress": selection_progress(config),
        "selection_timestep": round(
            selection_progress(config) * (config.experiment.steps - 1)
        ),
        "primary_visibility": config.experiment.scoring.primary_visibility,
        "row_aggregation": config.experiment.scoring.attender_rows,
        "span_aggregation": config.experiment.scoring.receiver_span,
        "fixed_offset": fit_fixed_offset(offsets),
        "select_dev_stability": result["stability"],
        "test_outcomes_used": False,
        "forbidden_test_artifacts_read": [],
    }
    return SelectionLock.create(
        track=config.track,
        model_id=config.model.id,
        model_revision=config.model.revision,
        dataset_id=config.dataset.id,
        relation=relation,
        layer=winner["layer"],
        head=winner["head"],
        top_k=TOP_K,
        metric="receiver_span_top1_accuracy",
        decision_rule=(
            "within relation: require denominator >=25; select top 5 by accuracy desc, "
            "denominator desc, layer asc, head asc; choose only among those candidates on dev "
            "by the same ordering"
        ),
        tie_break="denominator descending, then lowest layer, then lowest head",
        config_hash=evidence.scientific_config_hash,
        select_manifest_hash=evidence.manifests["select"],
        dev_manifest_hash=evidence.manifests["dev"],
        candidate_scores_hash=candidate_hash,
        frozen_settings=settings,
        created_at=evidence.metadata.get("started_at") or "source-start-time-unavailable",
    )


def _permutation_results(
    evidence: SourceEvidence,
    relation_records: dict[str, dict[str, Any]],
    *,
    n_permutations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation in RELATION_NAMES:
        record = relation_records[relation]
        base = {
            "relation": relation,
            "role": record["role"],
            "family": "primary_separate" if relation == PRIMARY_RELATION else "five_secondaries",
            "status": record["status"],
            "raw_p_value": np.nan,
            "holm_adjusted_p_value": np.nan,
            "n_permutations": 0,
            "observed_dev_accuracy": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "reason": record["reason"],
        }
        if record["status"] == "selected":
            permutation = selection_aware_permutation(
                evidence.select_rows,
                evidence.dev_rows,
                relation=relation,
                top_k=TOP_K,
                n_permutations=n_permutations,
                seed=PERMUTATION_SEED,
                minimum_denominator=MINIMUM_DENOMINATOR,
            )
            base.update(
                {
                    "raw_p_value": permutation["p_value"],
                    "n_permutations": permutation["n_permutations"],
                    "observed_dev_accuracy": permutation.get("observed_dev_accuracy", np.nan),
                    "null_mean": permutation.get("null_mean", np.nan),
                    "null_std": permutation.get("null_std", np.nan),
                    "reason": permutation.get("reason"),
                }
            )
        rows.append(base)
    _apply_secondary_holm(rows)
    return rows


def _apply_secondary_holm(rows: list[dict[str, Any]]) -> None:
    eligible = [
        row
        for row in rows
        if row["relation"] in SECONDARY_RELATIONS and not pd.isna(row["raw_p_value"])
    ]
    eligible.sort(key=lambda row: (row["raw_p_value"], SECONDARY_RELATIONS.index(row["relation"])))
    running = 0.0
    family_size = len(SECONDARY_RELATIONS)
    for rank, row in enumerate(eligible, start=1):
        adjusted = min(1.0, (family_size - rank + 1) * float(row["raw_p_value"]))
        running = max(running, adjusted)
        row["holm_adjusted_p_value"] = running


def _protocol(config: RunConfig, n_permutations: int) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "relations": list(RELATION_NAMES),
        "primary_relation": PRIMARY_RELATION,
        "secondary_family": list(SECONDARY_RELATIONS),
        "seeds": list(REQUIRED_SEEDS),
        "minimum_denominator": MINIMUM_DENOMINATOR,
        "top_k": TOP_K,
        "select_order": ["accuracy desc", "n_total desc", "layer asc", "head asc"],
        "dev_gate": "select top-K candidates only",
        "dev_order": ["accuracy desc", "n_total desc", "layer asc", "head asc"],
        "insufficient_evidence_policy": "record status and reason; never force a lock",
        "permutations": n_permutations,
        "permutation_seed": PERMUTATION_SEED,
        "multiplicity": "primary raw p separate; Holm across the five predefined secondaries",
        "steps": config.experiment.steps,
        "selection_progress": selection_progress(config),
        "selection_timestep": round(selection_progress(config) * (config.experiment.steps - 1)),
        "row_aggregation": config.experiment.scoring.attender_rows,
        "span_aggregation": config.experiment.scoring.receiver_span,
    }


def _source_identity(evidence: SourceEvidence) -> dict[str, Any]:
    config = evidence.config
    return {
        "run_id": evidence.run_dir.name,
        "started_at": evidence.metadata.get("started_at"),
        "track": config.track,
        "model_id": config.model.id,
        "dataset_id": config.dataset.id,
        "experiment_id": config.experiment.id,
        "scientific_config_hash": evidence.scientific_config_hash,
        "manifest_hashes_hash": canonical_hash(evidence.manifests),
    }


def _relation_role(relation: str) -> str:
    return "primary_confirmatory" if relation == PRIMARY_RELATION else "predefined_secondary"


def _head_grid(rows: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(layer), int(head))
        for layer, head in rows[["layer", "head"]].drop_duplicates().itertuples(index=False)
    }


def _load_existing_bundle(output: Path, evidence: SourceEvidence, source_hash: str) -> BundleBuild:
    saved_validation_path = output / "validation.json"
    if not saved_validation_path.is_file():
        raise ArtifactError("existing relation-selection bundle has no validation record")
    saved_validation = _read_json(saved_validation_path)
    if (
        saved_validation.get("schema_version") != VALIDATION_SCHEMA
        or not saved_validation.get("valid")
        or saved_validation.get("source_selection_artifacts_hash") != source_hash
    ):
        raise ArtifactError("existing relation-selection validation record is invalid")
    bundle = _read_json(output / "relation_selection_bundle.json")
    if bundle.get("source_selection_artifacts_hash") != source_hash:
        raise ArtifactError("existing relation-selection bundle belongs to different source evidence")
    validation = _validate_written_bundle(output, source_hash)
    if not validation["valid"]:
        raise ArtifactError(
            "existing relation-selection bundle is invalid: " + "; ".join(validation["errors"])
        )
    primary = bundle["relations"][PRIMARY_RELATION]
    lock = None
    if primary["status"] == "selected":
        lock = SelectionLock(**_read_json(output / primary["lock"]))
    select = pd.read_csv(output / primary["select_candidates"])
    dev = pd.read_csv(output / primary["dev_candidates"])
    return BundleBuild(output, bundle, lock, select, dev)


def _validate_written_bundle(output: Path, source_hash: str) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "relation_selection_bundle.json",
        "source_artifacts.json",
        "config.resolved.yaml",
        "run_metadata.json",
        "summary.json",
        "permutation_results.csv",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        errors.append("missing files: " + ", ".join(missing))
    bundle: dict[str, Any] = {}
    if not missing:
        bundle = _read_json(output / "relation_selection_bundle.json")
        source_record = _read_json(output / "source_artifacts.json")
        if bundle.get("schema_version") != BUNDLE_SCHEMA:
            errors.append("bundle schema mismatch")
        if bundle.get("source_selection_artifacts_hash") != source_hash:
            errors.append("source selection artifact hash mismatch")
        if source_record.get("stable_selection_artifacts_hash") != source_hash:
            errors.append("source-artifact record hash mismatch")
        stable_hashes = source_record.get("stable_selection_artifact_hashes", {})
        if canonical_hash(stable_hashes) != source_hash:
            errors.append("source-artifact hash map is inconsistent")
        config_hash = stable_hashes.get("config.resolved.yaml")
        if config_hash and _sha256(output / "config.resolved.yaml") != config_hash:
            errors.append("copied resolved config hash mismatch")
        if set(bundle.get("relations", {})) != set(RELATION_NAMES):
            errors.append("bundle does not contain exactly the six canonical relations")
        for relation, record in bundle.get("relations", {}).items():
            if record.get("status") not in {"selected", "insufficient_evidence"}:
                errors.append(f"{relation} has an invalid selection status")
            if record.get("role") != _relation_role(relation):
                errors.append(f"{relation} has an invalid primary/secondary role")
            candidate_files: dict[str, str] = {}
            for field in ("select_candidates", "dev_candidates"):
                relative_path = record.get(field)
                if not isinstance(relative_path, str):
                    errors.append(f"{relation} is missing {field}")
                    continue
                candidate_path = output / relative_path
                if not candidate_path.is_file():
                    errors.append(f"missing {relation} {field}")
                else:
                    candidate_files[field] = _sha256(candidate_path)
            if len(candidate_files) == 2:
                candidate_hash = canonical_hash(
                    {
                        "select_csv_sha256": candidate_files["select_candidates"],
                        "dev_csv_sha256": candidate_files["dev_candidates"],
                    }
                )
                if candidate_hash != record.get("candidate_table_hash"):
                    errors.append(f"{relation} candidate table hash mismatch")
            if record.get("status") == "selected":
                relative_lock = record.get("lock")
                if not isinstance(relative_lock, str):
                    errors.append(f"missing {relation} selection lock mapping")
                    continue
                lock_path = output / relative_lock
                if not lock_path.is_file():
                    errors.append(f"missing {relation} selection lock")
                else:
                    lock = _read_json(lock_path)
                    if set(lock) != set(SelectionLock.__dataclass_fields__):
                        errors.append(f"{relation} lock schema fields do not match")
                    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("relation") != relation:
                        errors.append(f"{relation} lock identity is invalid")
                    if lock.get("candidate_scores_hash") != record.get("candidate_table_hash"):
                        errors.append(f"{relation} lock candidate hash mismatch")
                    settings = lock.get("frozen_settings", {})
                    if settings.get("source_selection_artifacts_hash") != source_hash:
                        errors.append(f"{relation} lock source-artifact hash mismatch")
                    if settings.get("relation_role") != _relation_role(relation):
                        errors.append(f"{relation} lock role mismatch")
            elif record.get("lock") is not None:
                errors.append(f"{relation} insufficient-evidence record has a fabricated lock")
        permutation = pd.read_csv(output / "permutation_results.csv")
        if set(permutation.get("relation", pd.Series(dtype=str))) != set(RELATION_NAMES):
            errors.append("permutation results do not contain exactly the six relations")
        primary = bundle.get("relations", {}).get(PRIMARY_RELATION, {})
        if primary.get("status") == "selected":
            alias = output / "selection_lock.json"
            primary_lock_path = primary.get("lock")
            relation_lock = output / primary_lock_path if isinstance(primary_lock_path, str) else None
            if (
                not alias.is_file()
                or relation_lock is None
                or not relation_lock.is_file()
                or alias.read_bytes() != relation_lock.read_bytes()
            ):
                errors.append("primary lock alias is missing or not byte-equivalent")
    return {
        "schema_version": VALIDATION_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "validated_at": _now(),
        "source_selection_artifacts_hash": source_hash,
    }


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    if path.exists():
        if path.read_bytes() != csv_bytes:
            raise ArtifactError(f"refusing to overwrite immutable candidate table: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(csv_bytes)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

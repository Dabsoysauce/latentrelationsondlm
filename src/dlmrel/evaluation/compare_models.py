"""Validated cross-model comparison on the common valid instance set."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from ..artifacts import ArtifactError, validate_run

OBSERVATION_AXES = (
    "treebank",
    "relation",
    "seed",
    "timestep",
    "normalized_progress",
    "visibility",
    "layer",
    "head",
    "depth",
    "position",
    "position_state",
)


def comparison_scientific_identity(config: dict) -> dict:
    """Scientific settings that must match, excluding only model/runtime identity."""
    identity = deepcopy(config)
    identity.pop("model", None)
    identity.pop("runtime", None)
    identity.pop("selection_lock_hash", None)
    return identity


def compare_runs(run_paths: list[str], output: str | Path) -> tuple[Path, Path]:
    summaries = []
    frames = []
    identity = None
    manifests = None
    for raw in run_paths:
        path = Path(raw)
        validation = validate_run(path)
        if not validation["valid"]:
            raise ArtifactError(f"invalid run {path}: {validation['errors']}")
        config = yaml.safe_load((path / "config.resolved.yaml").read_text(encoding="utf-8"))
        current_identity = comparison_scientific_identity(config)
        current_manifests = json.loads((path / "manifest_refs.json").read_text(encoding="utf-8"))
        if identity is not None and current_identity != identity:
            raise ArtifactError("runs have incompatible non-model scientific configurations")
        if manifests is not None and current_manifests != manifests:
            raise ArtifactError("runs have incompatible manifest hashes")
        identity, manifests = current_identity, current_manifests

        metrics = pd.read_csv(path / "metrics.csv")
        metrics.insert(0, "run_dir", str(path))
        metrics.insert(1, "model", config["model"]["id"])
        summaries.append(metrics)
        instances = pd.read_parquet(path / "instances.parquet")
        instances["model"] = config["model"]["id"]
        frames.append(instances)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.sort_values(list(summary.columns), kind="mergesort").to_csv(output, index=False)
    common_output = output.with_name(output.stem + "_common_instances.csv")
    common_instance_comparison(frames).to_csv(common_output, index=False)
    return output, common_output


def common_instance_comparison(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    for frame in frames:
        required = {"model", "instance_id", "correct"}
        if missing := required - set(frame):
            raise ArtifactError(f"comparison instances missing columns: {sorted(missing)}")
    common = set.intersection(*(set(frame["instance_id"].dropna().astype(str)) for frame in frames))
    rows = []
    for frame in frames:
        if frame["instance_id"].isna().any():
            raise ArtifactError("comparison instances contain a missing instance ID")
        identity_columns = ["instance_id", *[axis for axis in OBSERVATION_AXES if axis in frame]]
        if frame.duplicated(identity_columns).any():
            raise ArtifactError("comparison instances contain duplicate observations")
        subset = frame[frame["instance_id"].astype(str).isin(common)]
        if subset.empty or "correct" not in subset:
            continue
        group_columns = ["model", *[axis for axis in OBSERVATION_AXES if axis in subset]]
        rows.extend(
            subset.groupby(group_columns, as_index=False, observed=True)
            .agg(
                accuracy=("correct", "mean"),
                n_rows=("correct", "size"),
                n_common_instances=("instance_id", "nunique"),
            )
            .to_dict("records")
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            [column for column in ["model", *OBSERVATION_AXES] if column in output],
            kind="mergesort",
        ).reset_index(drop=True)
    return output

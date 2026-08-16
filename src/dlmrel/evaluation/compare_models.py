"""Validated cross-model comparison on the common valid instance set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ..artifacts import ArtifactError, validate_run


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
        current_identity = (
            config["schema_version"],
            config["track"],
            config["dataset"]["id"],
            config["experiment"]["type"],
            config["experiment"]["scoring"],
        )
        current_manifests = json.loads((path / "manifest_refs.json").read_text(encoding="utf-8"))
        if identity is not None and current_identity != identity:
            raise ArtifactError("runs have incompatible datasets, experiments, or scoring rules")
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
    pd.concat(summaries, ignore_index=True).to_csv(output, index=False)
    common_output = output.with_name(output.stem + "_common_instances.csv")
    common_instance_comparison(frames).to_csv(common_output, index=False)
    return output, common_output


def common_instance_comparison(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    common = set.intersection(*(set(frame["instance_id"].dropna().astype(str)) for frame in frames))
    rows = []
    for frame in frames:
        subset = frame[frame["instance_id"].astype(str).isin(common)]
        if subset.empty or "correct" not in subset:
            continue
        rows.extend(
            subset.groupby(["model", "treebank", "relation"], as_index=False)
            .agg(
                accuracy=("correct", "mean"),
                n_rows=("correct", "size"),
                n_common_instances=("instance_id", "nunique"),
            )
            .to_dict("records")
        )
    return pd.DataFrame(rows)

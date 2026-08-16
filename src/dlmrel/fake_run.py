"""Small deterministic run used to test artifact plumbing without a GPU."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from .artifacts import ArtifactError, SelectionLock, atomic_json, canonical_hash, write_shard
from .config import RunConfig
from .models.fake import FakeAdapter
from .selection import create_selection_lock, write_lock_bundle


def run_fake(cfg: RunConfig, run_dir: Path) -> None:
    adapter = FakeAdapter()
    rows = []
    for seed in cfg.experiment.seeds:
        output = adapter.forward(torch.tensor([[1, 3, 5, 7, 9]]), timestep=seed % cfg.experiment.steps)
        for layer, attention in enumerate(output.attentions or ()):
            for head in range(attention.shape[1]):
                prediction = int(attention[0, head, 3].argmax())
                rows.append(
                    {
                        "sentence_id": "synthetic:0",
                        "instance_id": f"synthetic:{seed}",
                        "treebank": cfg.dataset.treebank,
                        "relation": "object_to_verb",
                        "seed": seed,
                        "timestep": seed % cfg.experiment.steps,
                        "normalized_progress": (seed % cfg.experiment.steps)
                        / max(cfg.experiment.steps - 1, 1),
                        "visibility": "both_visible",
                        "layer": layer,
                        "head": head,
                        "prediction": prediction,
                        "correct": int(prediction == 1),
                    }
                )
    scores = pd.DataFrame(rows).groupby(["relation", "layer", "head"], as_index=False).agg(
        accuracy=("correct", "mean"), n_total=("correct", "size")
    )
    lock = _selection_lock(cfg, run_dir, scores)
    selected = [
        row
        for row in rows
        if row["relation"] == lock.relation and row["layer"] == lock.layer and row["head"] == lock.head
    ]
    write_shard(run_dir, 0, selected)
    frame = pd.DataFrame(selected)
    frame.to_parquet(run_dir / "instances.parquet", index=False)
    pd.DataFrame(columns=["sentence_id", "instance_id", "role", "reason"]).to_parquet(
        run_dir / "exclusions.parquet", index=False
    )
    per_seed = frame.groupby(["seed", "relation", "layer", "head"], as_index=False)["correct"].mean()
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(["relation", "layer", "head"], as_index=False).agg(
        accuracy=("correct", "mean"), seed_std=("correct", "std"), n_seeds=("seed", "nunique")
    ).to_csv(run_dir / "metrics.csv", index=False)
    atomic_json(
        run_dir / "summary.json",
        {
            "schema_version": "dlmrel-run-v1",
            "completion_status": "complete",
            "n_instances": len(frame),
            "capabilities": adapter.capabilities.__dict__,
        },
    )
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update({"completion_status": "complete", "ended_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(run_dir / "run_metadata.json", metadata)


def _selection_lock(cfg: RunConfig, run_dir: Path, scores: pd.DataFrame) -> SelectionLock:
    manifests = json.loads((run_dir / "manifest_refs.json").read_text(encoding="utf-8"))
    if cfg.track == "external_treebank_transfer":
        if not cfg.runtime.selection_lock:
            raise ArtifactError("external transfer requires an EWT selection lock")
        lock = SelectionLock(**json.loads(Path(cfg.runtime.selection_lock).read_text(encoding="utf-8")))
        if lock.dataset_id != "ewt" or lock.model_id != cfg.model.id:
            raise ArtifactError("external transfer lock must be EWT and use the same model")
        lock.write_once(run_dir / "selection_lock.json")
        return lock

    dev = scores.copy()
    dev["accuracy"] = dev["accuracy"] + (dev["head"] == 1) * 0.01
    lock, candidates, dev_candidates = create_selection_lock(
        scores,
        dev,
        relation=cfg.experiment.scoring.primary_relation,
        top_k=cfg.experiment.scoring.top_k,
        track=cfg.track,
        model_id=cfg.model.id,
        model_revision=cfg.model.revision,
        dataset_id=cfg.dataset.id,
        config_hash=canonical_hash(cfg.to_dict()),
        select_manifest_hash=manifests["select"],
        dev_manifest_hash=manifests["dev"],
    )
    write_lock_bundle(run_dir, lock, candidates, dev_candidates)
    return lock

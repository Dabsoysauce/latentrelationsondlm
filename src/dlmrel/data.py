"""The single official-split data path used by every active experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from conllu import parse_incr

from .artifacts import ArtifactError, atomic_json
from .config import DatasetConfig, RunConfig
from .relations import Example, build_example
from .splits import build_official_manifests, manifest_hash
from .treebank import acquire_split


def manifest_root(dataset: DatasetConfig) -> Path:
    return Path("data/manifests") / dataset.id / dataset.release


def prepare_manifests(dataset: DatasetConfig, *, download: bool = True) -> dict[str, Any]:
    """Verify pinned UD files and create select/dev/test manifests."""
    sentences = {}
    for split in ("train", "dev", "test"):
        path = acquire_split(dataset, split, download=download)
        with path.open(encoding="utf-8") as stream:
            sentences[split] = list(parse_incr(stream))

    manifests = build_official_manifests(dataset, sentences)
    root = manifest_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for role, rows in manifests.items():
        pd.DataFrame([asdict(row) for row in rows]).to_csv(root / f"{role}.csv", index=False)
        hashes[role] = manifest_hash(rows)

    report = {
        "schema_version": "dlmrel-manifest-v1",
        "dataset": dataset.id,
        "treebank": dataset.treebank,
        "release": dataset.release,
        "revision": dataset.revision,
        "checksums": dataset.checksums,
        "counts": {role: len(rows) for role, rows in manifests.items()},
        "manifest_hashes": hashes,
        "official_boundaries": True,
        "zero_overlap": True,
    }
    atomic_json(root / "audit.json", report)
    return report


def load_manifest_examples(
    cfg: RunConfig, tokenizer, role: str
) -> tuple[list[Example], pd.DataFrame]:
    """Tokenize a frozen manifest without replacing rejected sentences."""
    expected_split = {"select": "train", "dev": "dev", "test": "test"}.get(role)
    if expected_split is None:
        raise ValueError(f"unknown manifest role: {role}")

    path = manifest_root(cfg.dataset) / f"{role}.csv"
    if not path.exists():
        raise ArtifactError(f"missing prepared manifest: {path}")
    manifest = pd.read_csv(path)
    if set(manifest["original_split"]) != {expected_split}:
        raise ArtifactError(f"{role} manifest violates official {expected_split} boundary")

    source_path = acquire_split(cfg.dataset, expected_split, download=False)
    with source_path.open(encoding="utf-8") as stream:
        sentences = list(parse_incr(stream))
    by_sent_id = {
        str(sentence.metadata.get("sent_id")): sentence
        for sentence in sentences
        if sentence.metadata.get("sent_id") is not None
    }

    examples: list[Example] = []
    exclusions: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        sentence = by_sent_id.get(str(row.sent_id))
        if sentence is None:
            exclusions.append(_exclusion(row, role, "sent_id_not_found"))
            continue
        sentence.metadata["source_treebank"] = cfg.dataset.treebank
        sentence.metadata["source_split"] = expected_split
        example = build_example(sentence, tokenizer, cfg.dataset, include_bos=True)
        if example is None:
            exclusions.append(_exclusion(row, role, "tokenization_alignment_or_relation_filter"))
            continue
        example.sentence_id = str(row.sentence_id)
        example.language = cfg.dataset.language
        example.original_split = expected_split
        example.source = cfg.dataset.treebank
        for instance in example.relations:
            instance.instance_id = f"{row.sentence_id}:{instance.instance_id.split(':')[-1]}"
        examples.append(example)
    return examples, pd.DataFrame(exclusions)


def load_audit(dataset: DatasetConfig) -> dict[str, Any]:
    path = manifest_root(dataset) / "audit.json"
    if not path.exists():
        raise ArtifactError(f"missing prepared manifests: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusion(row, role: str, reason: str) -> dict[str, Any]:
    return {"sentence_id": str(row.sentence_id), "instance_id": None, "role": role, "reason": reason}

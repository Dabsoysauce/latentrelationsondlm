"""The single official-split data path used by every active experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from conllu import parse_incr

from .artifacts import ArtifactError, atomic_json
from .config import DatasetConfig, RunConfig
from .relations import Example, build_example
from .splits import (
    ManifestRow,
    assert_zero_overlap,
    build_official_manifests,
    manifest_hash,
    normalize_text,
    stable_sentence_id,
)
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
    manifest = pd.read_csv(path, keep_default_na=False)
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
        try:
            example = build_example(sentence, tokenizer, cfg.dataset, include_bos=True)
        except (IndexError, RuntimeError, TypeError, UnicodeError, ValueError) as error:
            exclusions.append(
                _exclusion(row, role, f"tokenizer_or_alignment_error:{type(error).__name__}")
            )
            continue
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
    """Load and independently verify prepared manifests and pinned source files."""
    path = manifest_root(dataset) / "audit.json"
    if not path.exists():
        raise ArtifactError(f"missing prepared manifests: {path}")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"prepared-manifest audit is unreadable: {path}") from error

    expected_identity = {
        "schema_version": "dlmrel-manifest-v1",
        "dataset": dataset.id,
        "treebank": dataset.treebank,
        "release": dataset.release,
        "revision": dataset.revision,
        "checksums": dataset.checksums,
        "official_boundaries": True,
        "zero_overlap": True,
    }
    for key, expected in expected_identity.items():
        if audit.get(key) != expected:
            raise ArtifactError(f"prepared-manifest audit differs from dataset config at {key}")
    if set(audit.get("manifest_hashes", {})) != {"select", "dev", "test"}:
        raise ArtifactError("prepared-manifest audit must contain select/dev/test hashes")
    if set(audit.get("counts", {})) != {"select", "dev", "test"}:
        raise ArtifactError("prepared-manifest audit must contain select/dev/test counts")

    source_sentences = {}
    for split in ("train", "dev", "test"):
        source_path = acquire_split(dataset, split, download=False)
        with source_path.open(encoding="utf-8") as stream:
            source_sentences[split] = list(parse_incr(stream))
    expected_manifests = build_official_manifests(dataset, source_sentences)

    manifests: dict[str, list[ManifestRow]] = {}
    expected_splits = {"select": "train", "dev": "dev", "test": "test"}
    required_columns = {item.name for item in ManifestRow.__dataclass_fields__.values()}
    for role, original_split in expected_splits.items():
        manifest_path = manifest_root(dataset) / f"{role}.csv"
        if not manifest_path.exists():
            raise ArtifactError(f"missing prepared manifest: {manifest_path}")
        try:
            frame = pd.read_csv(manifest_path, keep_default_na=False)
        except (OSError, ValueError) as error:
            raise ArtifactError(f"prepared manifest is unreadable: {manifest_path}") from error
        if set(frame) != required_columns:
            raise ArtifactError(f"{role} manifest schema fields do not match")
        rows: list[ManifestRow] = []
        for raw in frame.to_dict("records"):
            try:
                raw["n_words"] = int(raw["n_words"])
                row = ManifestRow(**raw)
            except (TypeError, ValueError) as error:
                raise ArtifactError(f"{role} manifest contains an invalid row") from error
            normalized = normalize_text(row.normalized_text)
            text_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            expected_sentence_id = stable_sentence_id(
                dataset.treebank, original_split, row.sent_id, normalized
            )
            if (
                row.role != role
                or row.original_split != original_split
                or row.treebank != dataset.treebank
                or row.language != dataset.language
                or normalized != row.normalized_text
                or row.text_sha256 != text_digest
                or row.sentence_id != expected_sentence_id
                or row.n_words < dataset.min_words
            ):
                raise ArtifactError(f"{role} manifest row identity or boundary is invalid")
            rows.append(row)
        if len(rows) != audit["counts"][role]:
            raise ArtifactError(f"{role} manifest count differs from its audit")
        if manifest_hash(rows) != audit["manifest_hashes"][role]:
            raise ArtifactError(f"{role} manifest hash differs from its audit")
        if rows != expected_manifests[role]:
            raise ArtifactError(f"{role} manifest differs from deterministic preparation")
        manifests[role] = rows
    try:
        assert_zero_overlap(manifests)
    except ValueError as error:
        raise ArtifactError("prepared manifests overlap across official roles") from error
    return audit


def load_paper_manifest_refs(dataset: DatasetConfig, roles: tuple[str, ...]) -> dict[str, Any]:
    """Verify only select/test manifests for the corrected protocol.

    This intentionally never opens ``audit.json``, the official development
    file, or ``dev.csv``.  Development metadata may remain on disk for legacy
    provenance, but it cannot enter a corrected run identity.
    """
    if not roles or not set(roles).issubset({"select", "test"}):
        raise ValueError("paper manifest roles must be a nonempty select/test subset")
    root = manifest_root(dataset)
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    expected_splits = {"select": "train", "test": "test"}
    required_columns = {item.name for item in ManifestRow.__dataclass_fields__.values()}
    for role in roles:
        path = root / f"{role}.csv"
        if not path.is_file():
            raise ArtifactError(f"missing prepared manifest: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        if set(frame) != required_columns:
            raise ArtifactError(f"{role} manifest schema fields do not match")
        if set(frame["original_split"]) != {expected_splits[role]}:
            raise ArtifactError(f"{role} manifest violates its official boundary")
        rows = []
        for raw in frame.to_dict("records"):
            raw["n_words"] = int(raw["n_words"])
            rows.append(ManifestRow(**raw))
        hashes[role] = manifest_hash(rows)
        counts[role] = len(rows)
    return {
        "schema_version": "dlmrel-paper-manifest-refs-v1",
        "dataset": dataset.id,
        "roles": list(roles),
        "manifest_hashes": hashes,
        "counts": counts,
        "development_opened": False,
    }


def _exclusion(row, role: str, reason: str) -> dict[str, Any]:
    return {"sentence_id": str(row.sentence_id), "instance_id": None, "role": role, "reason": reason}

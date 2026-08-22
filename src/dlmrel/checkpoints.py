"""Atomic, validated sentence-chunk checkpoints for deterministic GPU runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd

from .artifacts import ArtifactError, atomic_json, canonical_hash

CHECKPOINT_SCHEMA = "dlmrel-sentence-checkpoint-v1"
SENTENCES_PER_CHUNK = 300
T = TypeVar("T")


@dataclass(frozen=True)
class CheckpointIdentity:
    stage: str
    seed: int
    normalized_progress: float
    timestep: int
    heads: tuple[tuple[int, int], ...] | None = None

    def filename(self, start: int, end: int) -> str:
        stage = re.sub(r"[^A-Za-z0-9_-]+", "-", self.stage).strip("-")
        heads = (
            "all"
            if self.heads is None
            else "-".join(f"l{layer}h{head}" for layer, head in self.heads)
        )
        return (
            f"{stage}__seed-{self.seed}__p-{self.normalized_progress:.6f}__"
            f"t-{self.timestep}__heads-{heads}__sentences-{start:06d}-{end:06d}.parquet"
        )


class SentenceCheckpointStore:
    """Load or atomically compute consecutive, non-overlapping sentence chunks."""

    def __init__(self, run_dir: str | Path, *, chunk_size: int = SENTENCES_PER_CHUNK):
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        metadata = json.loads((self.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        manifests = json.loads((self.run_dir / "manifest_refs.json").read_text(encoding="utf-8"))
        self.scientific_config_hash = metadata.get("scientific_config_hash") or metadata.get(
            "config_hash"
        )
        self.manifests = manifests
        if metadata.get("manifest_hashes_hash") not in {None, canonical_hash(manifests)}:
            raise ArtifactError("run metadata and manifest hashes disagree")

    def run(
        self,
        examples: Sequence[T],
        identity: CheckpointIdentity,
        compute: Callable[[Sequence[T], int], pd.DataFrame],
        *,
        legacy_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Reuse one legacy whole-seed file or resume at the first absent chunk."""
        frames = list(
            self.iter_chunks(
                examples,
                identity,
                compute,
                legacy_path=legacy_path,
            )
        )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def iter_chunks(
        self,
        examples: Sequence[T],
        identity: CheckpointIdentity,
        compute: Callable[[Sequence[T], int], pd.DataFrame],
        *,
        legacy_path: str | Path | None = None,
    ):
        """Yield validated chunks without concatenating a large stage in memory."""
        sentence_ids = [str(example.sentence_id) for example in examples]
        if len(sentence_ids) != len(set(sentence_ids)):
            raise ArtifactError("checkpoint input contains duplicate sentence IDs")
        if legacy_path is not None:
            legacy = self._load_legacy(Path(legacy_path), identity, examples)
            if legacy is not None:
                yield legacy
                return

        for start in range(0, len(examples), self.chunk_size):
            end = min(start + self.chunk_size, len(examples))
            path = self.directory / identity.filename(start, end)
            expected = self._expected_metadata(examples, identity, start, end)
            frame = self._load_chunk(path, expected, sentence_ids[start:end])
            if frame is None:
                frame = compute(examples[start:end], start)
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError("checkpoint computation must return a pandas DataFrame")
                _validate_frame_sentence_ids(frame, sentence_ids[start:end])
                self._write_chunk(path, frame, expected)
            yield frame

    def iter_paired_chunks(
        self,
        examples: Sequence[T],
        first_identity: CheckpointIdentity,
        second_identity: CheckpointIdentity,
        compute: Callable[[Sequence[T], int], tuple[pd.DataFrame, pd.DataFrame]],
    ):
        """Atomically checkpoint two products created by the same GPU pass.

        This lets the relation trajectory retain its ordinary resumable rows
        while also storing compact entropy rows for the later entropy runner.
        If either half is missing or invalid, both are recomputed together so
        they can never describe different model forwards.
        """
        if first_identity == second_identity:
            raise ValueError("paired checkpoint identities must differ")
        sentence_ids = [str(example.sentence_id) for example in examples]
        if len(sentence_ids) != len(set(sentence_ids)):
            raise ArtifactError("checkpoint input contains duplicate sentence IDs")
        for start in range(0, len(examples), self.chunk_size):
            end = min(start + self.chunk_size, len(examples))
            first_path = self.directory / first_identity.filename(start, end)
            second_path = self.directory / second_identity.filename(start, end)
            first_expected = self._expected_metadata(
                examples, first_identity, start, end
            )
            second_expected = self._expected_metadata(
                examples, second_identity, start, end
            )
            expected_ids = sentence_ids[start:end]
            first = self._load_chunk(first_path, first_expected, expected_ids)
            second = self._load_chunk(second_path, second_expected, expected_ids)
            if first is None or second is None:
                result = compute(examples[start:end], start)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise TypeError("paired checkpoint computation must return two DataFrames")
                first, second = result
                if not isinstance(first, pd.DataFrame) or not isinstance(second, pd.DataFrame):
                    raise TypeError("paired checkpoint computation must return two DataFrames")
                _validate_frame_sentence_ids(first, expected_ids)
                _validate_frame_sentence_ids(second, expected_ids)
                self._write_chunk(first_path, first, first_expected)
                self._write_chunk(second_path, second, second_expected)
            yield first, second

    def _expected_metadata(
        self,
        examples: Sequence[T],
        identity: CheckpointIdentity,
        start: int,
        end: int,
    ) -> dict:
        sentence_ids = [str(example.sentence_id) for example in examples[start:end]]
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            **asdict(identity),
            "heads": [list(head) for head in identity.heads] if identity.heads is not None else None,
            "sentence_start": start,
            "sentence_end": end,
            "sentence_ids_hash": canonical_hash(sentence_ids),
            "scientific_config_hash": self.scientific_config_hash,
            "manifest_hashes": self.manifests,
        }

    def _load_chunk(
        self, path: Path, expected: dict, expected_ids: Sequence[str]
    ) -> pd.DataFrame | None:
        metadata_path = _metadata_path(path)
        _discard_temporary(path)
        _discard_temporary(metadata_path)
        if not path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None
            if metadata.get("parquet_sha256") != _file_sha256(path):
                return None
            frame = pd.read_parquet(path)
            if len(frame) != metadata.get("row_count"):
                return None
            _validate_frame_sentence_ids(frame, expected_ids)
            return frame
        except (ArtifactError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _write_chunk(self, path: Path, frame: pd.DataFrame, expected: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        atomic_json(
            _metadata_path(path),
            {
                **expected,
                "row_count": len(frame),
                "parquet_sha256": _file_sha256(path),
            },
        )

    def _load_legacy(
        self, path: Path, identity: CheckpointIdentity, examples: Sequence[T]
    ) -> pd.DataFrame | None:
        """Accept old atomic whole-seed files after run-level identity validation."""
        if not path.exists():
            return None
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            return None
        if "seed" in frame and set(frame["seed"].astype(int)) != {identity.seed}:
            return None
        if "timestep" in frame and set(frame["timestep"].astype(int)) != {identity.timestep}:
            return None
        if "normalized_progress" in frame and not frame["normalized_progress"].eq(
            identity.normalized_progress
        ).all():
            return None
        if identity.heads is not None and {"layer", "head"}.issubset(frame):
            actual = set(zip(frame["layer"].astype(int), frame["head"].astype(int), strict=True))
            if not actual.issubset(set(identity.heads)):
                return None
        try:
            _validate_frame_sentence_ids(
                frame, [str(example.sentence_id) for example in examples]
            )
        except ArtifactError:
            return None
        expected = {
            **self._expected_metadata(examples, identity, 0, len(examples)),
            "legacy_whole_seed": True,
            "row_count": len(frame),
            "parquet_sha256": _file_sha256(path),
        }
        metadata_path = _metadata_path(path)
        if metadata_path.exists():
            try:
                if json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
                    return None
            except (OSError, json.JSONDecodeError):
                return None
        else:
            atomic_json(metadata_path, expected)
        return frame


def atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _discard_temporary(path: Path) -> None:
    path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def _validate_frame_sentence_ids(frame: pd.DataFrame, expected_ids: Sequence[str]) -> None:
    """Reject checkpoint rows assigned to sentences outside their input chunk."""
    if frame.empty:
        return
    if "sentence_id" not in frame:
        raise ArtifactError("nonempty sentence checkpoint is missing sentence_id")
    actual = set(frame["sentence_id"].dropna().astype(str))
    if frame["sentence_id"].isna().any() or not actual.issubset(set(expected_ids)):
        raise ArtifactError("sentence checkpoint contains rows outside its sentence range")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

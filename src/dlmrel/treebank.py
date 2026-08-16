"""Pinned treebank acquisition and checksum verification."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from .config import DatasetConfig


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conllu_name(dataset: DatasetConfig, split: str) -> str:
    suffix = dataset.treebank.removeprefix("UD_").split("-", 1)[-1].lower()
    return f"{dataset.language}_{suffix}-ud-{split}.conllu"


def pinned_url(dataset: DatasetConfig, split: str) -> str:
    return f"{dataset.repository}/raw/{dataset.revision}/{conllu_name(dataset, split)}"


def verify_checksum(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    wanted = expected.removeprefix("sha256:").lower()
    if actual != wanted:
        raise ChecksumError(f"checksum mismatch for {path}: expected {wanted}, got {actual}")


def acquire_split(dataset: DatasetConfig, split: str, *, download: bool = True) -> Path:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"invalid UD split: {split}")
    path = Path(dataset.cache_dir) / dataset.id / dataset.release / conllu_name(dataset, split)
    expected = dataset.checksums.get(split)
    if not expected:
        raise ChecksumError(f"no pinned checksum configured for {dataset.id}/{split}")
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(pinned_url(dataset, split), temporary)
        verify_checksum(temporary, expected)
        temporary.replace(path)
    verify_checksum(path, expected)
    return path

"""Pinned, checksum-verified Stanford POS tagger provisioning."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

STANFORD_TAGGER_VERSION = "4.2.0"
STANFORD_TAGGER_URL = "https://nlp.stanford.edu/software/stanford-tagger-4.2.0.zip"
STANFORD_TAGGER_ARCHIVE_SHA256 = (
    "0e900017c052114d30e688a6218e229551c045f6abb6907ba8a52b2a119bcb23"
)
STANFORD_TAGGER_ROOT = "stanford-postagger-full-2020-11-17"
STANFORD_TAGGER_JAR = "stanford-postagger.jar"
STANFORD_TAGGER_MODEL = "models/english-left3words-distsim.tagger"
STANFORD_TAGGER_JAR_SHA256 = (
    "f6090106c57da13d2ac8a1b2798dd7f437e07a9909a00f917e884bf6fa52fc8d"
)
STANFORD_TAGGER_MODEL_SHA256 = (
    "ebb5f7454da95775ecdb3ee20d3c58488cd87aa9999585951645f949e962089f"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_cache() -> Path:
    configured = os.environ.get("DLMREL_STANFORD_POS_CACHE")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "dlmrel" / f"stanford-pos-{STANFORD_TAGGER_VERSION}"


def _installed_paths(cache: Path) -> tuple[Path, Path]:
    root = cache / STANFORD_TAGGER_ROOT
    return root / STANFORD_TAGGER_JAR, root / STANFORD_TAGGER_MODEL


def _valid_install(cache: Path) -> bool:
    jar, model = _installed_paths(cache)
    return (
        jar.is_file()
        and model.is_file()
        and _sha256(jar) == STANFORD_TAGGER_JAR_SHA256
        and _sha256(model) == STANFORD_TAGGER_MODEL_SHA256
    )


def _download_archive(destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(STANFORD_TAGGER_URL, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if _sha256(temporary) != STANFORD_TAGGER_ARCHIVE_SHA256:
            raise RuntimeError("downloaded Stanford POS archive failed its pinned SHA-256")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise RuntimeError("Stanford POS archive contains an unsafe path") from error
        bundle.extractall(destination)


def provision_stanford_pos(cache_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Install the exact audited archive once and configure the POS runner.

    No binary is committed to the repository.  Repeated calls verify and reuse
    the persistent cache; corrupt archives or extracted files fail closed.
    """
    cache = Path(cache_dir) if cache_dir is not None else _default_cache()
    cache.mkdir(parents=True, exist_ok=True)
    if not _valid_install(cache):
        archive = cache / f"stanford-tagger-{STANFORD_TAGGER_VERSION}.zip"
        if archive.is_file() and _sha256(archive) != STANFORD_TAGGER_ARCHIVE_SHA256:
            archive.unlink()
        if not archive.is_file():
            _download_archive(archive)
        with tempfile.TemporaryDirectory(prefix="stanford-pos-extract-", dir=cache) as temporary:
            extraction = Path(temporary) / "contents"
            _safe_extract(archive, extraction)
            extracted_root = extraction / STANFORD_TAGGER_ROOT
            if not extracted_root.is_dir():
                raise RuntimeError("Stanford POS archive has an unexpected directory layout")
            installed_root = cache / STANFORD_TAGGER_ROOT
            if installed_root.exists():
                shutil.rmtree(installed_root)
            os.replace(extracted_root, installed_root)
        if not _valid_install(cache):
            raise RuntimeError("extracted Stanford POS JAR/model failed pinned SHA-256 checks")
    jar, model = _installed_paths(cache)
    os.environ["STANFORD_POS_TAGGER_JAR"] = str(jar.resolve())
    os.environ["STANFORD_POS_TAGGER_MODEL"] = str(model.resolve())
    return jar.resolve(), model.resolve()


def stanford_pos_identity(jar: str | Path, model: str | Path) -> dict[str, str | bool]:
    """Describe whether configured files are the pinned audited release."""
    jar_path, model_path = Path(jar), Path(model)
    jar_hash, model_hash = _sha256(jar_path), _sha256(model_path)
    pinned = (
        jar_hash == STANFORD_TAGGER_JAR_SHA256
        and model_hash == STANFORD_TAGGER_MODEL_SHA256
    )
    return {
        "release": STANFORD_TAGGER_VERSION if pinned else "externally_configured",
        "archive_url": STANFORD_TAGGER_URL if pinned else "unknown",
        "archive_sha256": STANFORD_TAGGER_ARCHIVE_SHA256 if pinned else "unknown",
        "jar_sha256": jar_hash,
        "model_sha256": model_hash,
        "matches_pinned_release": pinned,
    }

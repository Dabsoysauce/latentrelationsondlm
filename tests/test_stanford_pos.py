from __future__ import annotations

import hashlib
import zipfile

from dlmrel import stanford_pos


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_pinned_stanford_package_is_extracted_verified_and_reused(tmp_path, monkeypatch):
    root = "test-stanford"
    jar_bytes = b"audited jar"
    model_bytes = b"audited model"
    archive = tmp_path / "stanford-tagger-test.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{root}/stanford-postagger.jar", jar_bytes)
        bundle.writestr(f"{root}/models/english-left3words-distsim.tagger", model_bytes)

    monkeypatch.setattr(stanford_pos, "STANFORD_TAGGER_VERSION", "test")
    monkeypatch.setattr(stanford_pos, "STANFORD_TAGGER_ROOT", root)
    monkeypatch.setattr(
        stanford_pos, "STANFORD_TAGGER_ARCHIVE_SHA256", stanford_pos._sha256(archive)
    )
    monkeypatch.setattr(stanford_pos, "STANFORD_TAGGER_JAR_SHA256", _digest(jar_bytes))
    monkeypatch.setattr(stanford_pos, "STANFORD_TAGGER_MODEL_SHA256", _digest(model_bytes))

    jar, model = stanford_pos.provision_stanford_pos(tmp_path)
    assert jar.read_bytes() == jar_bytes
    assert model.read_bytes() == model_bytes
    assert stanford_pos.stanford_pos_identity(jar, model)["matches_pinned_release"] is True

    archive.unlink()
    assert stanford_pos.provision_stanford_pos(tmp_path) == (jar, model)


def test_official_stanford_dependency_hashes_are_frozen():
    assert stanford_pos.STANFORD_TAGGER_URL.endswith("stanford-tagger-4.2.0.zip")
    assert len(stanford_pos.STANFORD_TAGGER_ARCHIVE_SHA256) == 64
    assert len(stanford_pos.STANFORD_TAGGER_JAR_SHA256) == 64
    assert len(stanford_pos.STANFORD_TAGGER_MODEL_SHA256) == 64

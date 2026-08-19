from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dlmrel.artifacts import ArtifactError
from dlmrel.config import DatasetConfig
from dlmrel.data import load_audit, load_manifest_examples, manifest_root, prepare_manifests
from dlmrel.splits import ManifestRow, manifest_hash
from dlmrel.treebank import ChecksumError, conllu_name, sha256_file


def _sentence(sent_id: str, subject: str, obj: str) -> str:
    text = f"The old {subject} saw a red {obj} ."
    return f"""# sent_id = {sent_id}
# text = {text}
1\tThe\tthe\tDET\tDT\t_\t3\tdet\t_\t_
2\told\told\tADJ\tJJ\t_\t3\tamod\t_\t_
3\t{subject}\t{subject}\tNOUN\tNN\t_\t4\tnsubj\t_\t_
4\tsaw\tsee\tVERB\tVBD\t_\t0\troot\t_\t_
5\ta\ta\tDET\tDT\t_\t7\tdet\t_\t_
6\tred\tred\tADJ\tJJ\t_\t7\tamod\t_\t_
7\t{obj}\t{obj}\tNOUN\tNN\t_\t4\tobj\t_\t_
8\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_

"""


@pytest.fixture
def prepared_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "ud"
    base = DatasetConfig(
        id="tiny",
        treebank="UD_English-Tiny",
        language="en",
        release="2.15",
        repository="https://example.invalid/tiny",
        revision="a" * 40,
        checksums={"train": "", "dev": "", "test": ""},
        cache_dir=str(cache),
        n_select=None,
        n_dev=None,
        n_test=None,
    )
    contents = {
        "train": _sentence("train-1", "chef", "meal")
        + _sentence("train-2", "teacher", "lesson"),
        "dev": _sentence("dev-1", "pilot", "route"),
        "test": _sentence("test-1", "artist", "mural"),
    }
    checksums = {}
    for split, content in contents.items():
        path = cache / base.id / base.release / conllu_name(base, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        checksums[split] = f"sha256:{sha256_file(path)}"
    dataset = replace(base, checksums=checksums)
    prepare_manifests(dataset, download=False)
    return dataset


def test_load_audit_recomputes_manifest_hash_and_identity(prepared_dataset):
    audit = load_audit(prepared_dataset)
    assert audit["counts"] == {"select": 2, "dev": 1, "test": 1}

    path = manifest_root(prepared_dataset) / "select.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, "normalized_text"] = "silently changed text"
    frame.to_csv(path, index=False)

    with pytest.raises(ArtifactError, match="row identity|hash differs"):
        load_audit(prepared_dataset)


def test_manifest_and_audit_cannot_be_coordinately_rewritten(prepared_dataset):
    root = manifest_root(prepared_dataset)
    path = root / "select.csv"
    frame = pd.read_csv(path, keep_default_na=False).iloc[::-1].reset_index(drop=True)
    frame.to_csv(path, index=False)
    rows = []
    for raw in frame.to_dict("records"):
        raw["n_words"] = int(raw["n_words"])
        rows.append(ManifestRow(**raw))
    audit_path = root / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["manifest_hashes"]["select"] = manifest_hash(rows)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ArtifactError, match="deterministic preparation"):
        load_audit(prepared_dataset)


def test_load_audit_rechecks_pinned_raw_file_checksum(prepared_dataset):
    raw = (
        Path(prepared_dataset.cache_dir)
        / prepared_dataset.id
        / prepared_dataset.release
        / conllu_name(prepared_dataset, "test")
    )
    raw.write_text(raw.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    with pytest.raises(ChecksumError, match="checksum mismatch"):
        load_audit(prepared_dataset)


def test_tokenizer_failure_is_recorded_without_replacement(prepared_dataset):
    class BrokenTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("synthetic tokenizer failure")

    examples, exclusions = load_manifest_examples(
        SimpleNamespace(dataset=prepared_dataset), BrokenTokenizer(), "select"
    )

    assert examples == []
    assert exclusions[["role", "reason"]].to_dict("records") == [
        {"role": "select", "reason": "tokenizer_or_alignment_error:RuntimeError"},
        {"role": "select", "reason": "tokenizer_or_alignment_error:RuntimeError"},
    ]

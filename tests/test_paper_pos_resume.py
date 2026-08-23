from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from dlmrel.experiments.paper_pos import (
    _frozen_probe_path,
    _load_frozen_probe,
    _write_frozen_probe,
)


def test_frozen_probe_round_trip_is_atomic_and_identity_checked(tmp_path: Path):
    key = (42, 0.25, "middle", "head_7")
    path = _frozen_probe_path(tmp_path, key)
    fitted = ("scaler", "classifier", [1, 2, 3], ["NOUN", "VERB"])

    _write_frozen_probe(path, key, fitted)

    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert _load_frozen_probe(path, key) == fitted[:2]
    with path.open("rb") as handle:
        assert pickle.load(handle)["format"] == "compact-v1"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load_frozen_probe(path, (43, 0.25, "middle", "head_7"))


def test_loading_legacy_probe_compacts_it_in_place(tmp_path: Path):
    key = (42, 0.5, "late", "residual")
    path = _frozen_probe_path(tmp_path, key)
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump({"key": key, "fitted": ("scaler", "classifier", [1, 2], ["NOUN"])}, handle)

    assert _load_frozen_probe(path, key) == ("scaler", "classifier")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload == {
        "key": key,
        "format": "compact-v1",
        "fitted": ("scaler", "classifier"),
    }

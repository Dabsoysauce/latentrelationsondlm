from __future__ import annotations

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
    fitted = {"classifier": [1, 2, 3], "training_rows": 17}

    _write_frozen_probe(path, key, fitted)

    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert _load_frozen_probe(path, key) == fitted
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load_frozen_probe(path, (43, 0.25, "middle", "head_7"))

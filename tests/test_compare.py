import json

import pandas as pd
import pytest
import yaml

from dlmrel.artifacts import ArtifactError
from dlmrel.evaluation import compare_models
from dlmrel.evaluation.compare_models import common_instance_comparison, compare_runs


def test_comparison_uses_common_instance_intersection():
    first = pd.DataFrame(
        {
            "model": ["a", "a"],
            "treebank": ["ewt", "ewt"],
            "relation": ["r", "r"],
            "instance_id": ["i1", "i2"],
            "correct": [1, 0],
        }
    )
    second = pd.DataFrame(
        {
            "model": ["b", "b"],
            "treebank": ["ewt", "ewt"],
            "relation": ["r", "r"],
            "instance_id": ["i2", "i3"],
            "correct": [1, 1],
        }
    )
    output = common_instance_comparison([first, second])
    assert set(output["n_common_instances"]) == {1}
    assert set(output["accuracy"]) == {0.0, 1.0}


def _comparison_run(path, model, *, seeds=None, progress=None):
    path.mkdir()
    config = {
        "schema_version": "dlmrel-config-v2",
        "track": "confirmatory_ewt",
        "model": {"id": model, "revision": f"{model}-revision"},
        "dataset": {"id": "ewt", "revision": "dataset-revision"},
        "experiment": {
            "id": "head-search",
            "type": "head_search",
            "steps": 64,
            "normalized_progress": progress or [0.0, 1.0],
            "seeds": seeds or [42, 43, 44],
            "scoring": {"receiver_span": "sum"},
        },
        "runtime": {"run_id": path.name},
    }
    (path / "config.resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (path / "manifest_refs.json").write_text(
        json.dumps({"select": "s", "dev": "d", "test": "t"}), encoding="utf-8"
    )
    pd.DataFrame([{"relation": "r", "accuracy": 1.0}]).to_csv(
        path / "metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "instance_id": "i1",
                "treebank": "ewt",
                "relation": "r",
                "seed": 42,
                "correct": 1,
            }
        ]
    ).to_parquet(path / "instances.parquet", index=False)


@pytest.mark.parametrize(
    "change",
    [
        {"seeds": [42, 43, 45]},
        {"progress": [0.0, 0.5, 1.0]},
    ],
)
def test_compare_runs_rejects_non_model_scientific_mismatch(tmp_path, monkeypatch, change):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _comparison_run(first, "dream")
    _comparison_run(second, "diffullama", **change)
    monkeypatch.setattr(compare_models, "validate_run", lambda _path: {"valid": True})

    with pytest.raises(ArtifactError, match="non-model scientific configurations"):
        compare_runs([str(first), str(second)], tmp_path / "comparison.csv")


def test_common_instance_comparison_rejects_duplicate_observations():
    duplicated = pd.DataFrame(
        {
            "model": ["a", "a"],
            "treebank": ["ewt", "ewt"],
            "relation": ["r", "r"],
            "instance_id": ["i1", "i1"],
            "seed": [42, 42],
            "correct": [1, 1],
        }
    )
    with pytest.raises(ArtifactError, match="duplicate observations"):
        common_instance_comparison([duplicated])


def test_common_instance_comparison_is_deterministically_sorted():
    frame = pd.DataFrame(
        {
            "model": ["b", "b"],
            "treebank": ["ewt", "ewt"],
            "relation": ["z", "a"],
            "instance_id": ["i2", "i1"],
            "seed": [42, 42],
            "correct": [1, 0],
        }
    )
    output = common_instance_comparison([frame])
    assert output["relation"].tolist() == ["a", "z"]

import json
from pathlib import Path

import pandas as pd

from dlmrel import cli
from dlmrel.artifacts import ArtifactError, merge_shards
from dlmrel.cli import main
from dlmrel.config import RELATION_NAMES
from dlmrel.relation_selection import load_relation_locks

ROOT = Path(__file__).parents[1]


def test_run_dry_run_does_not_load_model(capsys):
    code = main(
        [
            "run",
            "--model",
            str(ROOT / "configs/models/fake.yaml"),
            "--dataset",
            str(ROOT / "configs/datasets/ewt.yaml"),
            "--experiment",
            str(ROOT / "configs/experiments/head_search.yaml"),
            "--dry-run",
        ]
    )
    assert code == 0
    assert '"valid": true' in capsys.readouterr().out.lower()


def test_documented_command_surface_parses(capsys):
    code = main(
        [
            "smoke-test",
            "--model",
            str(ROOT / "configs/models/fake.yaml"),
            "--dry-run",
        ]
    )
    assert code == 0


def test_external_transfer_requires_non_ewt_and_lock(capsys):
    code = main(
        [
            "run",
            "--model",
            str(ROOT / "configs/models/fake.yaml"),
            "--dataset",
            str(ROOT / "configs/datasets/de_gsd.yaml"),
            "--experiment",
            str(ROOT / "configs/experiments/external_transfer.yaml"),
            "--dry-run",
        ]
    )
    assert code == 0


def test_normal_cli_run_can_continue_with_resume_flag(tmp_path, monkeypatch, capsys):
    original = cli.run_fake

    monkeypatch.setattr(
        cli,
        "load_audit",
        lambda _dataset: {
            "manifest_hashes": {
                "select": "select-fixture",
                "dev": "dev-fixture",
                "test": "test-fixture",
            }
        },
    )

    def interrupt(_cfg, _target):
        raise ArtifactError("simulated interruption")

    arguments = [
        "run",
        "--model",
        str(ROOT / "configs/models/fake.yaml"),
        "--dataset",
        str(ROOT / "configs/datasets/ewt.yaml"),
        "--experiment",
        str(ROOT / "configs/experiments/head_search.yaml"),
        "--results",
        str(tmp_path),
        "--run-id",
        "resume-test",
    ]
    monkeypatch.setattr(cli, "run_fake", interrupt)
    assert main(arguments) == 2
    assert "simulated interruption" in capsys.readouterr().err

    monkeypatch.setattr(cli, "run_fake", original)
    assert main([*arguments, "--resume"]) == 0
    assert '"valid": true' in capsys.readouterr().out.lower()

    run = (
        tmp_path
        / "confirmatory_ewt"
        / "fake"
        / "ewt"
        / "confirmatory_head_search"
        / "resume-test"
    )
    validation = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    bundle = json.loads(
        (run / "relation-selection/relation_selection_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    locks = load_relation_locks(run / "relation-selection")
    instances = pd.read_parquet(run / "instances.parquet")

    assert validation["valid"] is True
    assert summary["completion_status"] == "complete"
    assert set(summary["relations"]) == set(RELATION_NAMES)
    assert set(bundle["relations"]) == set(RELATION_NAMES)
    assert set(locks.locks) == set(RELATION_NAMES)
    assert len(merge_shards(run)) == len(instances)
    assert instances[["sentence_id", "instance_id", "seed", "relation"]].duplicated().sum() == 0
    for relation in RELATION_NAMES:
        checkpoint = json.loads(
            (run / "permutation-checkpoints" / f"{relation}.json").read_text(
                encoding="utf-8"
            )
        )
        lock = locks.resolve(relation)
        relation_rows = instances[instances["relation"] == relation]
        assert checkpoint["completion_status"] == "complete"
        assert checkpoint["completed_permutation_indices"] == list(range(10))
        assert set(
            relation_rows[["layer", "head"]].itertuples(index=False, name=None)
        ) == {(lock.layer, lock.head)}

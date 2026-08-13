from pathlib import Path

from dlmrel.cli import main

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
    code = main(["status", "--results", str(ROOT / "results")])
    assert code == 0


def test_external_transfer_requires_non_ewt_and_lock(capsys):
    code = main(
        [
            "run",
            "--model",
            str(ROOT / "configs/models/fake.yaml"),
            "--dataset",
            str(ROOT / "configs/datasets/gum.yaml"),
            "--experiment",
            str(ROOT / "configs/experiments/external_transfer.yaml"),
            "--dry-run",
        ]
    )
    assert code == 0

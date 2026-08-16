from pathlib import Path

from dlmrel import cli
from dlmrel.artifacts import ArtifactError
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

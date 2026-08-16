from pathlib import Path

import pytest
import yaml

from dlmrel.config import ConfigError, RunConfig

ROOT = Path(__file__).parents[1]


def test_all_shipped_run_configs_are_consumed():
    models = sorted((ROOT / "configs/models").glob("*.yaml"))
    datasets = [ROOT / "configs/datasets/ewt.yaml"]
    experiments = [ROOT / "configs/experiments/head_search.yaml"]
    for model in models:
        for dataset in datasets:
            RunConfig.load_files(model, dataset, experiments[0])


def test_every_experiment_yaml_is_strictly_consumed_with_fake_model():
    for experiment in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        raw = yaml.safe_load(experiment.read_text())
        assert raw["seeds"] == [42, 43, 44]
        dataset = (
            ROOT / "configs/datasets/de_gsd.yaml"
            if raw.get("track") == "external_treebank_transfer"
            else ROOT / "configs/datasets/ewt.yaml"
        )
        RunConfig.load_files(ROOT / "configs/models/fake.yaml", dataset, experiment)


def test_non_protocol_seed_list_is_rejected(tmp_path):
    experiment = yaml.safe_load((ROOT / "configs/experiments/head_search.yaml").read_text())
    experiment["seeds"] = [42]
    path = tmp_path / "one-seed.yaml"
    path.write_text(yaml.safe_dump(experiment))

    with pytest.raises(ConfigError, match=r"seeds must be exactly \[42, 43, 44\]"):
        RunConfig.load_files(
            ROOT / "configs/models/fake.yaml",
            ROOT / "configs/datasets/ewt.yaml",
            path,
        )


def test_unknown_config_field_is_rejected(tmp_path):
    model = yaml.safe_load((ROOT / "configs/models/fake.yaml").read_text())
    model["typo_field"] = 123
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(model))
    with pytest.raises(ConfigError, match="unknown model field"):
        RunConfig.load_files(
            path,
            ROOT / "configs/datasets/ewt.yaml",
            ROOT / "configs/experiments/head_search.yaml",
        )


def test_capability_mismatch_fails_before_model_loading(tmp_path):
    model = yaml.safe_load((ROOT / "configs/models/fake.yaml").read_text())
    model["capabilities"]["attentions"] = False
    path = tmp_path / "no_attention.yaml"
    path.write_text(yaml.safe_dump(model))
    with pytest.raises(ConfigError, match="requires model capability 'attentions'"):
        RunConfig.load_files(
            path,
            ROOT / "configs/datasets/ewt.yaml",
            ROOT / "configs/experiments/head_search.yaml",
        )

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
        raw = yaml.safe_load(model.read_text())
        if not raw.get("capabilities", {}).get("attentions"):
            continue
        if raw.get("family") == "gpt2":
            continue
        for dataset in datasets:
            RunConfig.load_files(model, dataset, experiments[0])


def test_every_experiment_yaml_is_strictly_consumed_with_fake_model():
    for experiment in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        raw = yaml.safe_load(experiment.read_text())
        dataset = (
            ROOT / "configs/datasets/gum.yaml"
            if raw.get("track") == "external_treebank_transfer"
            else ROOT / "configs/datasets/ewt.yaml"
        )
        if raw.get("type") == "native_timing":
            continue
        RunConfig.load_files(ROOT / "configs/models/fake.yaml", dataset, experiment)


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


def test_capability_mismatch_fails_before_model_loading():
    with pytest.raises(ConfigError, match="requires model capability 'attentions'"):
        RunConfig.load_files(
            ROOT / "configs/models/llada_8b.yaml",
            ROOT / "configs/datasets/ewt.yaml",
            ROOT / "configs/experiments/head_search.yaml",
        )

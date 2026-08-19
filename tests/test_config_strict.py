from pathlib import Path

import pytest
import yaml

from dlmrel.artifacts import canonical_hash
from dlmrel.config import ConfigError, RunConfig

ROOT = Path(__file__).parents[1]


def test_research_model_and_dataset_yaml_scientific_identities_are_frozen():
    expected = {
        "configs/models/diffullama_7b.yaml": (
            "5997b275f425a9489a9f58e029bd0b192f4dd21314bda68fdbf72216dab3acdd"
        ),
        "configs/models/dream_7b.yaml": "00049bf5913d7c33a8b026b2993a42075218263f15c857569f877a49ea79701e",
        "configs/datasets/de_gsd.yaml": "f10a3cfe6b36b464b5655c7c27c6d6a84d2febbe82d80eab3f5e02537bc1f711",
        "configs/datasets/ewt.yaml": "826daf2e7604f37e6d8a12a8b9f00b384a3a9901817628fcb07e6dae550e0a88",
        "configs/datasets/ja_gsd.yaml": "9d7fbea3c4071838e9e5de03bca09f98c4491eee1c08328ef98716204e89c63c",
    }
    for filename, digest in expected.items():
        assert canonical_hash(yaml.safe_load((ROOT / filename).read_text())) == digest


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


@pytest.mark.parametrize(
    "filename,field,error",
    [
        ("configs/models/fake.yaml", "revision", "missing required model field: revision"),
        ("configs/datasets/ewt.yaml", "checksums", "missing required dataset field: checksums"),
        ("configs/experiments/head_search.yaml", "steps", "missing required experiment field: steps"),
    ],
)
def test_missing_config_fields_fail_closed(tmp_path, filename, field, error):
    source = ROOT / filename
    raw = yaml.safe_load(source.read_text())
    raw.pop(field)
    path = tmp_path / source.name
    path.write_text(yaml.safe_dump(raw))
    arguments = {
        "model": ROOT / "configs/models/fake.yaml",
        "dataset": ROOT / "configs/datasets/ewt.yaml",
        "experiment": ROOT / "configs/experiments/head_search.yaml",
    }
    arguments[source.parent.name.removesuffix("s")] = path
    with pytest.raises(ConfigError, match=error):
        RunConfig.load_files(**arguments)


def test_missing_nested_scoring_field_fails_closed(tmp_path):
    experiment = yaml.safe_load((ROOT / "configs/experiments/head_search.yaml").read_text())
    experiment["scoring"].pop("receiver_span")
    path = tmp_path / "missing-scoring.yaml"
    path.write_text(yaml.safe_dump(experiment))
    with pytest.raises(ConfigError, match="missing required experiment.scoring field"):
        RunConfig.load_files(
            ROOT / "configs/models/fake.yaml",
            ROOT / "configs/datasets/ewt.yaml",
            path,
        )


@pytest.mark.parametrize("change", ["missing", "malformed"])
def test_dataset_requires_all_three_full_sha256_checksums(tmp_path, change):
    dataset = yaml.safe_load((ROOT / "configs/datasets/ewt.yaml").read_text())
    if change == "missing":
        dataset["checksums"].pop("test")
    else:
        dataset["checksums"]["test"] = "sha256:1234"
    path = tmp_path / "bad-checksum.yaml"
    path.write_text(yaml.safe_dump(dataset))
    with pytest.raises(ConfigError, match="dataset checksums"):
        RunConfig.load_files(
            ROOT / "configs/models/fake.yaml",
            path,
            ROOT / "configs/experiments/head_search.yaml",
        )


def test_mutable_tokenizer_revision_is_rejected(tmp_path):
    model = yaml.safe_load((ROOT / "configs/models/fake.yaml").read_text())
    model["tokenizer_revision"] = "main"
    path = tmp_path / "mutable-tokenizer.yaml"
    path.write_text(yaml.safe_dump(model))
    with pytest.raises(ConfigError, match="tokenizer revision must be immutable"):
        RunConfig.load_files(
            path,
            ROOT / "configs/datasets/ewt.yaml",
            ROOT / "configs/experiments/head_search.yaml",
        )


def test_frozen_steps_progress_and_scoring_are_rejected_if_changed(tmp_path):
    original = yaml.safe_load((ROOT / "configs/experiments/head_search.yaml").read_text())
    changes = [
        ("steps", 63),
        ("normalized_progress", [0.0, 1.0]),
        ("scoring", {**original["scoring"], "top_k": 4}),
    ]
    for field, value in changes:
        experiment = {**original, field: value}
        path = tmp_path / f"changed-{field}.yaml"
        path.write_text(yaml.safe_dump(experiment))
        with pytest.raises(ConfigError):
            RunConfig.load_files(
                ROOT / "configs/models/fake.yaml",
                ROOT / "configs/datasets/ewt.yaml",
                path,
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

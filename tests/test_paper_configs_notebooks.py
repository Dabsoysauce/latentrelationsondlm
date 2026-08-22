import ast
import json
import re
from pathlib import Path

import yaml

from dlmrel.cli import main
from dlmrel.config import PAPER_EXPERIMENT_TYPES, RunConfig, is_paper_experiment

ROOT = Path(__file__).parents[1]


def test_all_ten_canonical_configs_are_strict_and_have_exact_seeds():
    for experiment_id in PAPER_EXPERIMENT_TYPES:
        path = ROOT / "configs" / "experiments" / f"{experiment_id}.yaml"
        assert path.is_file()
        raw = yaml.safe_load(path.read_text())
        assert raw["id"] == experiment_id
        assert raw["type"] == experiment_id
        assert raw["seeds"] == [42, 43, 44]
        dataset = "de_gsd.yaml" if experiment_id == "multilingual_relation_head_transfer" else "ewt.yaml"
        cfg = RunConfig.load_files(
            ROOT / "configs/models/fake.yaml",
            ROOT / "configs/datasets" / dataset,
            path,
        )
        assert is_paper_experiment(cfg.experiment)
        serialized = path.read_text().lower()
        assert "permutation" not in serialized
        assert "holm" not in serialized


def test_corrected_time_configs_cover_every_step_and_relative_depths_are_frozen():
    time_ids = {
        "relation_head_receiver_prediction_over_diffusion_time",
        "attention_entropy",
        "final_token_prediction_by_layer",
        "prediction_before_unmasking_timing_analysis",
        "attention_heatmaps_and_trajectories",
        "multilingual_relation_head_transfer",
    }
    for experiment_id in time_ids:
        raw = yaml.safe_load(
            (ROOT / "configs/experiments" / f"{experiment_id}.yaml").read_text()
        )
        assert [round(value * 63) for value in raw["normalized_progress"]] == list(range(64))
    for experiment_id in (
        "attention_entropy",
        "pos_token_class_linear_probes",
        "final_token_prediction_by_layer",
    ):
        settings = yaml.safe_load(
            (ROOT / "configs/experiments" / f"{experiment_id}.yaml").read_text()
        )["settings"]
        assert settings["relative_depths"] == {"early": 0.2, "middle": 0.5, "late": 0.9}


def test_pos_protocol_is_multi_depth_multi_mask_fixed_and_has_no_tuning_role():
    raw = yaml.safe_load(
        (ROOT / "configs/experiments/pos_token_class_linear_probes.yaml").read_text()
    )
    assert raw["normalized_progress"] == [0.0, 0.25, 0.5, 0.75]
    assert raw["settings"]["mask_ratios"] == [1.0, 0.75, 0.5, 0.25]
    assert raw["settings"]["fixed_regularization_c"] == 1.0
    assert raw["settings"]["label_inventory"] == [
        "NOUN", "VERB", "ADJ", "ADV", "PREP", "DET", "PRON", "CONJ"
    ]
    assert raw["settings"]["automatic_tagger_release"] == "4.2.0"
    assert raw["settings"]["historical_tagger_release_recovered"] is False


def test_corrected_runner_modules_have_no_development_load_or_inference_calls():
    paper_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src/dlmrel/experiments").glob("paper_*.py"))
    )
    assert not re.search(r"load_manifest_examples\([^\n]+[\"']dev[\"']", paper_sources)
    assert "selection_aware_permutation(" not in paper_sources
    assert "holm_correction(" not in paper_sources
    assert "n_permutations" not in paper_sources


def test_prompt_manifest_preserves_twelve_reasoning_and_twelve_creative():
    raw = json.loads((ROOT / "configs/prompts/paper_reasoning_creative.json").read_text())
    tasks = [row["task"] for row in raw["prompts"]]
    assert tasks.count("reasoning") == 12
    assert tasks.count("creative") == 12
    assert raw["seed"] == 42


def test_both_colab_notebooks_are_valid_thin_restart_safe_launchers():
    names = ["Dream_Paper_Experiments.ipynb", "DiffuLLaMA_Paper_Experiments.ipynb"]
    rendered = []
    pins = []
    for name in names:
        notebook = json.loads((ROOT / "notebooks" / name).read_text())
        assert notebook["nbformat"] == 4
        sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        rendered.append(sources)
        pin = re.search(r'^GIT_COMMIT = "([0-9a-f]{40})"$', sources, re.MULTILINE)
        assert pin is not None
        pins.append(pin.group(1))
        for experiment_id in PAPER_EXPERIMENT_TYPES:
            assert experiment_id in sources
        for required in (
            "nvidia-smi",
            "GH_TOKEN",
            "HF_TOKEN",
            "GIT_COMMIT",
            "checkout\", \"--detach",
            "pytest",
            "ruff",
            "configs/datasets/ewt.yaml",
            "configs/datasets/de_gsd.yaml",
            "configs/datasets/ja_gsd.yaml",
            "--resume",
            "validate-selection-locks",
            "selection-locks",
            "--timestep-batch-size",
            "--export-attention-cache",
            "--attention-cache",
            "DLMREL_STANFORD_POS_CACHE",
            "TIME_RUN",
            "RUN_EXPENSIVE = False",
            "summary.json",
            "summarize",
        ):
            assert required in sources
        assert "print(gh_token" not in sources.lower()
        assert "print(hf_token" not in sources.lower()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    assert "dlmrel-paper-results\" / \"dream" in rendered[0]
    assert "dlmrel-paper-results\" / \"diffullama" in rendered[1]
    assert pins == [
        "8a56e00b1dbec4081caf1f288fec02d8da2dd600",
        "8a56e00b1dbec4081caf1f288fec02d8da2dd600",
    ]


def test_fake_cli_runs_and_validates_all_ten_canonical_experiments(tmp_path, capsys):
    results = tmp_path / "p"

    def launch(experiment, run_id, *, dataset="ewt", lock=None):
        arguments = [
            "run",
            "--model",
            str(ROOT / "configs/models/fake.yaml"),
            "--dataset",
            str(ROOT / f"configs/datasets/{dataset}.yaml"),
            "--experiment",
            str(ROOT / f"configs/experiments/{experiment}.yaml"),
            "--results",
            str(results),
            "--run-id",
            run_id,
        ]
        if lock is not None:
            arguments.extend(["--selection-lock", str(lock)])
        assert main(arguments) == 0, capsys.readouterr().err
        matches = list(results.glob(f"*/fake/*/{experiment}/{run_id}/validation.json"))
        assert len(matches) == 1
        assert json.loads(matches[0].read_text())["valid"] is True

    launch("relation_head_receiver_prediction", "s")
    lock = next(
        results.glob(
            "*/fake/ewt/relation_head_receiver_prediction/s/selection-locks"
        )
    )
    lock_consumers = {
        "relation_head_receiver_prediction_over_diffusion_time",
        "direct_logit_attribution",
        "matched_relation_head_ablation",
        "attention_heatmaps_and_trajectories",
    }
    remaining = sorted(
        experiment
        for experiment in PAPER_EXPERIMENT_TYPES
        if experiment != "relation_head_receiver_prediction"
    )
    for index, experiment in enumerate(remaining):
        if experiment == "multilingual_relation_head_transfer":
            for dataset in ("de_gsd", "ja_gsd"):
                launch(experiment, f"t{index}{dataset[0]}", dataset=dataset, lock=lock)
        else:
            launch(
                experiment,
                f"e{index}",
                lock=lock if experiment in lock_consumers else None,
            )

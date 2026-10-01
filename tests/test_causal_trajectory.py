"""CPU mechanism tests, not scientific evidence about pretrained DLMs."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from conllu import parse

from dlmrel.config import DatasetConfig
from dlmrel.evaluation.causal import matched_control_contrasts, paired_summary
from dlmrel.experiments.causal_plots import plot_results
from dlmrel.experiments.causal_run import assert_equivalent, run_examples
from dlmrel.experiments.causal_trajectory import (
    CausalSettings,
    exogenous_reveal_plan,
    reconstruct,
    select_controls,
)
from dlmrel.models.interventions import HeadScale, scale_projection_heads, temporal_gate
from dlmrel.models.native import aligned_logits, random_reveal_trajectories, random_reveal_trajectory
from dlmrel.relations import build_example


class Tokenizer:
    bos_token_id = 1
    mask_token_id = 0
    words = ["[MASK]", "[BOS]", "The", "small", "dogs", "chase", "c", "ats", "."]

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        import re

        for match in re.finditer(r"\S+", text):
            parts = ["c", "ats"] if match.group() == "cats" else [match.group()]
            start = match.start()
            for part in parts:
                ids.append(self.words.index(part) if part in self.words else 0)
                offsets.append((start, start + len(part)))
                start += len(part)
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def decode(self, ids, **kwargs):
        return " ".join(self.words[int(i)] for i in ids)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.qkv = torch.nn.Linear(8, 24, bias=False)
        self.o_proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        q, k, v = [p.reshape(len(x), x.shape[1], 4, 2).transpose(1, 2) for p in self.qkv(x).chunk(3, dim=-1)]
        weighted = (q @ k.transpose(-1, -2) / 2**0.5).softmax(-1) @ v
        return self.o_proj(weighted.transpose(1, 2).reshape_as(x))


class Toy(torch.nn.Module):
    device = "cpu"
    prediction_offset = 0

    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.embed = torch.nn.Embedding(9, 8)
        self.denoise_model = torch.nn.Module()
        self.denoise_model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for block in self.denoise_model.layers:
            block.self_attn = Attention()
        self.lm = torch.nn.Linear(8, 9, bias=False)

    def forward_logits(self, ids):
        x = self.embed(ids)
        for block in self.denoise_model.layers:
            x = x + block.self_attn(x)
        return self.lm(x)


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    sentence = parse("""# sent_id = synthetic-1
# text = The small dogs chase cats .
1\tThe\t_\tDET\t_\t_\t3\tdet\t_\t_
2\tsmall\t_\tADJ\t_\t_\t3\tamod\t_\t_
3\tdogs\t_\tNOUN\t_\t_\t4\tnsubj\t_\t_
4\tchase\t_\tVERB\t_\t_\t0\troot\t_\t_
5\tcats\t_\tNOUN\t_\t_\t4\tobj\t_\t_
6\t.\t_\tPUNCT\t_\t_\t4\tpunct\t_\t_

""")[0]
    tokenizer = Tokenizer()
    example = build_example(sentence, tokenizer, DatasetConfig())
    locks = SimpleNamespace(
        locks={
            "object_to_verb": SimpleNamespace(layer=0, head=0),
            "subject_to_verb": SimpleNamespace(layer=1, head=1),
        }
    )
    locks.heads = {(v.layer, v.head) for v in locks.locks.values()}
    locks.resolve = lambda relation: locks.locks[relation]
    return Toy().eval(), tokenizer, example, locks


@pytest.mark.parametrize("alpha", [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
def test_exact_slice_and_unselected_heads(fixture, alpha):
    model, _, _, _ = fixture
    projection = model.denoise_model.layers[1].self_attn.o_proj
    x = torch.arange(48.0).reshape(2, 3, 8)
    expected = x.clone()
    expected[0, :, 4:6] *= alpha
    original_weights = projection.weight.clone()
    with scale_projection_heads(
        model, [[HeadScale(1, 2, alpha)], [HeadScale(1, 2, alpha)]], torch.tensor([True, False])
    ):
        actual = projection(x)
    torch.testing.assert_close(actual, torch.nn.functional.linear(expected, projection.weight))
    torch.testing.assert_close(projection.weight, original_weights, rtol=0, atol=0)
    torch.testing.assert_close(projection(x), torch.nn.functional.linear(x, projection.weight))
    assert not projection._forward_pre_hooks


def test_multilayer_multhead_patch_and_cleanup(fixture):
    model, _, _, _ = fixture
    ids = torch.tensor([[1, 0, 0]])
    original = model.forward_logits(ids)
    heads = [[HeadScale(0, 0), HeadScale(0, 2), HeadScale(1, 3)]]
    capture = {}
    with scale_projection_heads(model, heads, torch.tensor([False]), capture=capture):
        torch.testing.assert_close(model.forward_logits(ids), original)
    with scale_projection_heads(model, heads, torch.tensor([True]), patches=capture):
        torch.testing.assert_close(model.forward_logits(ids), original)
    with pytest.raises(RuntimeError, match="deliberate"):
        with scale_projection_heads(model, heads, torch.tensor([True])):
            model.forward_logits(ids)
            raise RuntimeError("deliberate")
    torch.testing.assert_close(model.forward_logits(ids), original)
    assert all(not b.self_attn.o_proj._forward_pre_hooks for b in model.denoise_model.layers)


@pytest.mark.parametrize(
    "heads",
    [
        [HeadScale(-1, 0)],
        [HeadScale(0, 4)],
        [HeadScale(0, 1), HeadScale(0, 1)],
        [HeadScale(0, 1, float("nan"))],
    ],
)
def test_invalid_requests_fail_before_hooks(fixture, heads):
    model, _, _, _ = fixture
    with pytest.raises(ValueError):
        with scale_projection_heads(model, [heads], torch.tensor([True])):
            pass
    assert all(not b.self_attn.o_proj._forward_pre_hooks for b in model.denoise_model.layers)


def test_temporal_partial_spans_and_half_open_ranges():
    mask = torch.tensor([[False, True, True, True], [False, True, False, True], [False] * 4])
    spans, gov = [[1, 2]] * 3, [[3]] * 3

    def gate(window, step=1, **kwargs):
        return temporal_gate(mask, spans, gov, step=step, steps=8, window=window, **kwargs).tolist()

    assert gate("pre") == [True, False, False]
    assert gate("post") == [False, False, True]
    assert gate("entire", start=2, stop=5) == [False] * 3
    assert gate("entire", step=5, start=2, stop=5) == [False] * 3
    assert gate("early_pre", first_reveal=[3] * 3) == [True, False, False]
    assert gate("late_pre", first_reveal=[3] * 3) == [False] * 3
    assert gate("late_pre", step=2, first_reveal=[3] * 3) == [True, False, False]


def test_alignment_reveal_schedule_and_prediction_offset(fixture):
    _, tokenizer, example, _ = fixture
    obj = next(r for r in example.relations if r.relation == "object_to_verb")
    assert obj.attender_span == [5, 6] and obj.receiver_span == [4]
    assert tokenizer.encode(example.text)[4:6] == [6, 7]
    mask = torch.tensor([[False, True, True]])
    plan = exogenous_reveal_plan(mask, 42)
    assert torch.equal(plan.sum(0), mask.long())
    assert torch.equal(plan, exogenous_reveal_plan(mask, 42))
    logits = torch.arange(27.0).reshape(1, 3, 9)
    ids = torch.tensor([[1, 0, 0]])
    torch.testing.assert_close(aligned_logits(logits, ids, -1)[:, 1:], logits[:, :-1])


def test_native_defaults_unchanged(fixture):
    model, tokenizer, _, _ = fixture
    singleton = random_reveal_trajectory(model, tokenizer, "The", seed=42, generation_length=8)
    batch = random_reveal_trajectories(model, tokenizer, ["The", "The"], seed=42, generation_length=8)
    for result in batch:
        assert torch.equal(singleton.final_ids, result.final_ids)
        assert all(
            torch.equal(a, b) for a, b in zip(singleton.pre_forward_ids, result.pre_forward_ids, strict=True)
        )


def test_alpha_one_batch_and_temporal_behavior(fixture):
    model, tokenizer, example, _ = fixture
    anchor = next(r for r in example.relations if r.relation == "object_to_verb")
    gold = torch.tensor([[1, *tokenizer.encode(example.text)]])
    settings = CausalSettings()
    baseline = reconstruct(model, tokenizer, gold, anchor, 42, [None], settings)[0]
    conditions = [(((0, 0),), 1.0, "entire"), (((0, 0),), 0.0, "pre"), (((0, 0),), 0.0, "post")]
    results = reconstruct(model, tokenizer, gold, anchor, 42, conditions, settings)
    assert_equivalent(baseline, results[0])
    for condition, result in zip(conditions, results, strict=True):
        separate = reconstruct(model, tokenizer, gold, anchor, 42, [condition], settings)[0]
        assert_equivalent(separate, result)
        assert result["reveal_steps"] == baseline["reveal_steps"]
        for step in result["active_steps"]:
            selected = [
                result["trace"][step]["masked"][p] for p in anchor.attender_span + anchor.receiver_span
            ]
            if condition[2] == "pre":
                assert all(selected)
            if condition[2] == "post":
                assert not any(selected)
    for pos in anchor.attender_span + anchor.receiver_span:
        assert results[2]["final_ids"][pos] == baseline["final_ids"][pos]
    assert not np.array_equal(
        np.array([r["gold_logits"] for r in baseline["trace"]]),
        np.array([r["gold_logits"] for r in results[1]["trace"]]),
    )


def test_control_selection_metadata_fail_closed(fixture):
    model, _, _, locks = fixture
    settings = CausalSettings()
    selected, audit = select_controls(model, locks, "object_to_verb", None, settings)
    assert selected["same_layer"][0][0] == 0
    assert selected["random"][0] not in locks.heads
    assert "matched_attention" not in selected
    assert any(r["status"] == "unavailable" for r in audit)
    scores = pd.DataFrame(
        [
            {
                "relation": "object_to_verb",
                "layer": layer,
                "head": h,
                "accuracy": 0.9 if (layer, h) == (0, 0) else 0.1,
                "attention_entropy": float(h),
                "attention_magnitude": float(layer),
            }
            for layer in range(2)
            for h in range(4)
        ]
    )
    selected, _ = select_controls(model, locks, "object_to_verb", scores, settings)
    assert "matched_attention" in selected


def test_end_to_end_synthetic_smoke(fixture, tmp_path):
    model, tokenizer, example, locks = fixture
    settings = replace(
        CausalSettings(),
        examples=1,
        seeds=(42, 43),
        windows=("pre", "post"),
        controls=("random", "same_layer", "other_relation"),
        batch_size=3,
        bootstrap_samples=100,
    )
    rows, audit, sanity = run_examples(
        model, tokenizer, [example], locks, None, settings, tmp_path / "examples", "synthetic"
    )
    assert len(rows) == 2 * 2 * 2 * 4 * len(example.relations)
    assert all(r["passed"] for r in sanity)
    assert rows.baseline_relation_preserved.isna().all()
    assert rows.baseline_non_target_accuracy.notna().all()
    summary, seeds = paired_summary(rows, n_boot=100)
    assert summary.n_sentences.eq(1).all()  # seeds/relations are NOT independent N
    identity = summary[summary.alpha == 1]
    assert identity.difference.eq(0).all()
    assert seeds.mcnemar_p.dropna().between(0, 1).all()
    contrasts = matched_control_contrasts(rows)
    assert not contrasts.empty
    paired_summary(contrasts, n_boot=100)
    rows.to_json(tmp_path / "paired.jsonl", orient="records", lines=True)
    summary.to_csv(tmp_path / "aggregate.csv", index=False)
    plot_results(tmp_path)
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 5
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5


def test_settings_reject_typos_and_bad_windows():
    with pytest.raises(ValueError):
        CausalSettings.from_dict({"alpah": [0]})
    with pytest.raises(ValueError):
        CausalSettings.from_dict({"stop": 0})


def test_cli_parses_fixed_head_overrides_and_smoke():
    from dlmrel.cli import build_parser

    args = build_parser().parse_args(
        [
            "causal-run",
            "--model",
            "configs/models/dream_7b.yaml",
            "--selection-lock",
            "existing-locks",
            "--output",
            "results/causal/unit",
            "--smoke-test",
            "--layer",
            "2",
            "--head",
            "3",
            "--alpha",
            "0",
            "--alpha",
            "1",
            "--window",
            "pre",
            "--seed",
            "42",
        ]
    )
    assert args.alpha == [0, 1] and args.layer == [2] and args.head == [3]
    assert args.smoke_test


def test_absolute_range_is_applied_in_trajectory(fixture):
    model, tokenizer, example, _ = fixture
    anchor = next(r for r in example.relations if r.relation == "object_to_verb")
    gold = torch.tensor([[1, *tokenizer.encode(example.text)]])
    result = reconstruct(
        model, tokenizer, gold, anchor, 42, [(((0, 0),), 0.0, "entire", 7, 11)], CausalSettings()
    )[0]
    assert result["active_steps"] == [7, 8, 9, 10]


def test_multiple_pairs_preserve_clustered_n_and_mcnemar_units(fixture, tmp_path):
    import copy

    model, tokenizer, example, locks = fixture
    extra = copy.deepcopy(next(r for r in example.relations if r.relation == "object_to_verb"))
    extra.instance_id += "-second"
    example.relations.append(extra)
    settings = replace(
        CausalSettings(), seeds=(42,), controls=(), windows=("pre",), alphas=(1.0,), bootstrap_samples=100
    )
    rows, _, sanity = run_examples(model, tokenizer, [example], locks, None, settings, tmp_path, "synthetic")
    assert len(sanity) == 2 and rows.anchor_instance_id.nunique() == 2
    summary, seeds = paired_summary(rows, n_boot=100)
    assert summary.n_sentences.eq(1).all()
    assert seeds.mcnemar_n.dropna().eq(1).all()
    assert len(list(tmp_path.glob("*-baseline.json"))) == 2


def test_cli_completes_artifacts(fixture, tmp_path, monkeypatch):
    import json
    from dataclasses import make_dataclass

    from dlmrel.cli import main
    from dlmrel.config import RunConfig
    from dlmrel.experiments import causal_run

    model, tokenizer, example, locks = fixture
    cfg = RunConfig.load_files(
        "configs/models/dream_7b.yaml",
        "configs/datasets/ewt.yaml",
        "configs/experiments/matched_relation_head_ablation.yaml",
    )
    cfg = replace(cfg, model=replace(cfg.model, device="cpu"))
    head_type = make_dataclass("Head", ["layer", "head"])
    locks.locks = {k: head_type(v.layer, v.head) for k, v in locks.locks.items()}
    locks.source, locks.source_kind = tmp_path / "locks", "paper_selection_only"
    monkeypatch.setattr(causal_run.RunConfig, "load_files", lambda *a, **kw: cfg)
    monkeypatch.setattr(causal_run, "load_fixed_targets", lambda *a: locks)
    monkeypatch.setattr(causal_run, "load_paper_manifest_refs", lambda *a: {"test": "unit"})
    monkeypatch.setattr(causal_run, "load_adapter", lambda *a: (model, tokenizer, {"synthetic": True}))
    monkeypatch.setattr(causal_run, "load_manifest_examples", lambda *a: ([example], pd.DataFrame()))
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("alphas: [1.0]\nwindows: [pre]\ncontrols: []\nbootstrap_samples: 100\n")
    output = tmp_path / "run"
    args = [
        "causal-run",
        "--model",
        "unused",
        "--selection-lock",
        "unused",
        "--causal-config",
        str(settings_path),
        "--output",
        str(output),
        "--smoke-test",
    ]
    assert main(args) == 0
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    assert (output / "paired.jsonl").is_file()
    assert (output / "aggregate.csv").is_file()
    assert main(args) == 2  # refusal to overwrite scientific evidence


def test_exact_mcnemar_known_discordance():
    from dlmrel.evaluation.causal import METRICS

    records = []
    for index in range(8):
        record = dict(
            sentence_id=str(index),
            seed=42,
            primary_anchor=True,
            is_anchor=True,
            head_relation="object_to_verb",
            relation="object_to_verb",
            control="target",
            alpha=0,
            window="pre",
            start=0,
            stop=64,
            specificity_degradation=0.0,
        )
        for metric in METRICS:
            record[f"baseline_{metric}"] = 1.0
            record[f"intervention_{metric}"] = float(index >= 6)
        records.append(record)
    summary, seeds = paired_summary(pd.DataFrame(records), n_boot=100)
    binary = seeds[seeds.metric == "governor_exact"].iloc[0]
    assert binary.mcnemar_n == 8 and binary.harmed == 6 and binary.helped == 0
    assert binary.mcnemar_p == pytest.approx(0.03125)
    assert summary.difference.iloc[0] == -0.75

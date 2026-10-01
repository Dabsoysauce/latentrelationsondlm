"""CLI orchestration for causal trajectories, with durable per-example artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..artifacts import ArtifactError, atomic_json, canonical_hash, dataframe_records
from ..config import RunConfig
from ..data import load_manifest_examples, load_paper_manifest_refs
from ..diffusion import tokenize
from ..evaluation.causal import matched_control_contrasts, paired_summary
from ..paper_protocol import write_resolved_selection_locks
from ..pipeline import load_adapter
from ..relation_selection import load_relation_locks, write_resolved_lock_manifest
from .causal_trajectory import CausalSettings, paired_rows, reconstruct, select_controls
from .paper_causal import _selection_scores_path
from .paper_relation import load_paper_locks


def load_fixed_targets(path, cfg):
    """Read either existing lock format; never derive or rank target heads."""
    source = Path(path)
    directory = source if source.is_dir() else source.parent
    if (directory / "selection_bundle.json").is_file():
        locks = load_paper_locks(path, cfg)
        if any(lock.tokenizer_revision != cfg.model.tokenizer_revision for lock in locks.locks.values()):
            raise ValueError("selection locks belong to a different tokenizer revision")
    else:
        locks = load_relation_locks(path, cfg)
    return locks


def assert_equivalent(baseline, identity):
    if baseline["final_ids"] != identity["final_ids"]:
        raise RuntimeError("alpha=1/restoration/batch sanity failed: final IDs differ")
    a = np.array([r["gold_logits"] for r in baseline["trace"]])
    b = np.array([r["gold_logits"] for r in identity["trace"]])
    if not np.allclose(a, b, atol=1e-5, rtol=1e-5):
        raise RuntimeError("alpha=1/restoration/batch sanity failed: gold logits differ")
    return float(np.max(np.abs(a - b)))


def run_examples(model, tokenizer, examples, locks, scores, settings, output, model_id):
    """Shared real/tiny-test runner. No outcome-based example or head selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, audit, sanity = [], [], []
    for relation in settings.relations:
        controls, report = select_controls(model, locks, relation, scores, settings)
        audit.extend(report)
        eligible = [e for e in examples if any(r.relation == relation for r in e.relations)][
            : settings.examples
        ]
        if not eligible:
            raise ValueError(f"no eligible examples for {relation}")
        workloads = [
            (example, anchor)
            for example in eligible
            for anchor in [r for r in example.relations if r.relation == relation]
        ]
        for example, anchor in workloads:
            gold, _ = tokenize(tokenizer, example.text, model.device, add_bos=True)
            if gold.shape[1] != example.seq_len:
                raise ValueError("gold tokenization disagrees with existing span alignment")
            for span in example.word_to_tokens.values():
                if not span or min(span) < 1 or max(span) >= gold.shape[1]:
                    raise ValueError("invalid BOS-shifted word span")
            for seed in settings.seeds:
                identity = canonical_hash(
                    {
                        "sentence": example.sentence_id,
                        "anchor": anchor.instance_id,
                        "relation": relation,
                        "seed": seed,
                    }
                )[:20]
                baseline = reconstruct(model, tokenizer, gold, anchor, seed, [None], settings)[0]
                check = reconstruct(
                    model,
                    tokenizer,
                    gold,
                    anchor,
                    seed,
                    [None, (controls["target"], 1.0, "entire")],
                    settings,
                )
                errors = [assert_equivalent(baseline, record) for record in check]
                atomic_json(output / f"{identity}-baseline.json", baseline)
                conditions = [
                    (kind, heads, alpha, window, begin, end)
                    for kind, heads in controls.items()
                    for alpha in settings.alphas
                    for window in settings.windows
                    for begin, end in (settings.step_ranges or ((settings.start, settings.stop),))
                ]
                shard = []
                for start in range(0, len(conditions), settings.batch_size):
                    batch = conditions[start : start + settings.batch_size]
                    generated = reconstruct(
                        model,
                        tokenizer,
                        gold,
                        anchor,
                        seed,
                        [(heads, alpha, window, begin, end) for _, heads, alpha, window, begin, end in batch],
                        settings,
                    )
                    for offset, ((kind, heads, alpha, window, begin, end), result) in enumerate(
                        zip(batch, generated, strict=True)
                    ):
                        if alpha == 1:
                            errors.append(assert_equivalent(baseline, result))
                        trace_name = f"{identity}-condition-{start + offset:04d}.json"
                        atomic_json(output / trace_name, result)
                        paired = paired_rows(
                            example,
                            anchor,
                            gold[0].tolist(),
                            baseline,
                            result,
                            seed=seed,
                            heads=heads,
                            alpha=alpha,
                            window=window,
                            control=kind,
                            model_id=model_id,
                            settings=replace(settings, start=begin, stop=end),
                        )
                        for row in paired:
                            row.update(
                                baseline_trace=f"{identity}-baseline.json", intervention_trace=trace_name
                            )
                        shard.extend(paired)
                restored = reconstruct(model, tokenizer, gold, anchor, seed, [None], settings)[0]
                errors.append(assert_equivalent(baseline, restored))
                sanity.append(
                    {
                        "sentence_id": example.sentence_id,
                        "seed": seed,
                        "relation": relation,
                        "max_identity_logit_error": max(errors),
                        "passed": True,
                    }
                )
                atomic_json(output / f"{identity}-paired.json", shard)
                rows.extend(shard)
                print(f"causal: {relation} {example.sentence_id} seed={seed} complete", flush=True)
    return pd.DataFrame(rows), audit, sanity


def command(args):
    cfg = RunConfig.load_files(
        args.model, args.dataset, "configs/experiments/matched_relation_head_ablation.yaml"
    )
    raw = yaml.safe_load(Path(args.causal_config).read_text(encoding="utf-8")) or {}
    for key in ("examples", "batch_size", "start", "stop"):
        value = getattr(args, key, None)
        if value is not None:
            raw[key] = value
            if key in {"start", "stop"}:
                raw["step_ranges"] = ()
    for argument, key in (
        ("relation", "relations"),
        ("seed", "seeds"),
        ("alpha", "alphas"),
        ("window", "windows"),
        ("control", "controls"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            raw[key] = value
    if args.layer is not None or args.head is not None:
        if args.layer is None or args.head is None or len(args.layer) != len(args.head):
            raise ValueError("supply one --layer per --head")
        raw["heads"] = list(zip(args.layer, args.head, strict=True))
    settings = CausalSettings.from_dict(raw)
    if args.smoke_test:
        settings = replace(settings, examples=min(settings.examples, 2), seeds=(settings.seeds[0],))
    locks = load_fixed_targets(args.selection_lock, cfg)
    refs = load_paper_manifest_refs(cfg.dataset, ("test",))
    try:
        score_path = _selection_scores_path(locks)
        scores = pd.read_csv(score_path)
        score_hash = canonical_hash(dataframe_records(scores))
    except ArtifactError:
        legacy_scores = locks.source.parent / "select_all_head_scores.csv"
        scores = pd.read_csv(legacy_scores) if legacy_scores.is_file() else None
        score_hash = canonical_hash(dataframe_records(scores)) if scores is not None else None
    provenance = {
        "schema": "dlmrel-causal-v1",
        "base_config": cfg.to_dict(),
        "causal_settings": asdict(settings),
        "manifest_refs": refs,
        "lock_hash": canonical_hash({k: asdict(v) for k, v in locks.locks.items()}),
        "selection_scores_hash": score_hash,
        "smoke_test": args.smoke_test,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "reveal_policy": "native_hazard_independent_paired_schedule",
        "task": "all_non_BOS_masked_free_running_reconstruction",
        "eligibility": "existing_manifest_examples; all_eligible_pairs",
        "target_source_kind": locks.source_kind,
        "fixed_targets": {k: {"layer": v.layer, "head": v.head} for k, v in locks.locks.items()},
    }
    if args.dry_run:
        print(json.dumps(provenance, indent=2))
        return
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    provenance["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    atomic_json(output / "config.resolved.json", provenance)
    atomic_json(output / "status.json", {"status": "running"})
    if cfg.model.device.startswith("cuda") and not torch.cuda.is_available():
        reason = "Configured CUDA model requires a GPU host; no checkpoint weights were downloaded."
        atomic_json(output / "status.json", {"status": "blocked", "reason": reason})
        raise ValueError(reason)
    model, tokenizer, metadata = load_adapter(cfg)
    if tokenizer is None:
        raise ValueError("generic fake adapter lacks projection heads; use the automated tiny-model tests")
    if hasattr(model, "eval"):
        model.eval()
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    exclusions.to_csv(output / "exclusions.csv", index=False)
    rows, audit, sanity = run_examples(
        model, tokenizer, examples, locks, scores, settings, output / "examples", cfg.model.id
    )
    rows.to_json(output / "paired.jsonl", orient="records", lines=True, force_ascii=False)
    summary, seeds = paired_summary(rows, n_boot=settings.bootstrap_samples)
    summary.to_csv(output / "aggregate.csv", index=False)
    seeds.to_csv(output / "per_seed.csv", index=False)
    disjoint = rows[(rows.relation != rows.head_relation) & ~rows.shares_anchor_tokens]
    if not disjoint.empty:
        disjoint_summary, _ = paired_summary(disjoint, n_boot=settings.bootstrap_samples)
        disjoint_summary.to_csv(output / "disjoint_other_relations.csv", index=False)
    contrasts = matched_control_contrasts(rows)
    if not contrasts.empty:
        contrast_summary, _ = paired_summary(contrasts, n_boot=settings.bootstrap_samples)
        contrast_summary.to_csv(output / "matched_control_contrasts.csv", index=False)
    atomic_json(output / "control_availability.json", audit)
    atomic_json(output / "sanity.json", sanity)
    atomic_json(output / "model.json", metadata)
    if locks.source_kind == "paper_selection_only":
        write_resolved_selection_locks(output, locks)
    else:
        write_resolved_lock_manifest(output, locks)
    from .causal_plots import plot_results

    plot_results(output)
    missing = [row for row in audit if row["status"] == "unavailable"]
    atomic_json(
        output / "status.json",
        {
            "status": "completed",
            "rows": len(rows),
            "missing_controls": missing,
            "confirmatory_claim_ready": False,
            "interpretation": "lexical recovery only; inspect matched contrasts and global degradation",
        },
    )
    print(json.dumps({"output": str(output), "rows": len(rows), "missing_controls": missing}, indent=2))


def add_parser(commands):
    parser = commands.add_parser("causal-run", help="paired free-running head intervention experiment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="configs/datasets/ewt.yaml")
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--causal-config", default="configs/causal/pilot.yaml")
    parser.add_argument("--output", required=True)
    for name in ("examples", "batch-size", "start", "stop"):
        parser.add_argument(f"--{name}", type=int)
    for name, kind in (
        ("relation", str),
        ("seed", int),
        ("alpha", float),
        ("window", str),
        ("control", str),
        ("layer", int),
        ("head", int),
    ):
        parser.add_argument(f"--{name}", type=kind, action="append")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(func=command)

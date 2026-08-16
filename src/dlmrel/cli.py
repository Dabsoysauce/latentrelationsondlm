"""Five commands: prepare, smoke-test, run, validate, and compare."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .artifacts import ArtifactError, atomic_json, initialize_run, run_directory, validate_run
from .config import ConfigError, DatasetConfig, RunConfig, RuntimeConfig, _strict_dataclass
from .data import load_audit, manifest_root, prepare_manifests
from .evaluation.compare_models import compare_runs
from .fake_run import run_fake
from .pipeline import load_adapter, model_smoke_report, run_real


def read_yaml(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return value


def dataset_config(path: str | Path) -> DatasetConfig:
    dataset = _strict_dataclass(DatasetConfig, read_yaml(path), "dataset")
    dataset.validate()
    return dataset


def resolve_run(args) -> RunConfig:
    runtime = RuntimeConfig(
        results_root=args.results,
        run_id=args.run_id,
        resume=args.resume,
        dry_run=args.dry_run,
        selection_lock=args.selection_lock,
    )
    return RunConfig.load_files(args.model, args.dataset, args.experiment, runtime=runtime)


def cmd_prepare(args) -> None:
    report = prepare_manifests(dataset_config(args.dataset), download=not args.no_download)
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_smoke_test(args) -> None:
    runtime = RuntimeConfig(dry_run=args.dry_run)
    cfg = RunConfig.load_files(
        args.model,
        args.dataset,
        args.experiment or "configs/experiments/head_search.yaml",
        runtime=runtime,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "model": cfg.model.id,
                    "revision": cfg.model.revision,
                    "capabilities": asdict(cfg.model.capabilities),
                },
                indent=2,
            )
        )
        return
    if cfg.model.family == "fake":
        output = load_adapter(cfg)[0].forward(torch.tensor([[1, 2, 3, 4]]), timestep=1)
        report = {
            "status": "passed",
            "logits": list(output.logits.shape),
            "layers": len(output.attentions or ()),
        }
    else:
        model, tokenizer, metadata = load_adapter(cfg)
        report = model_smoke_report(model, tokenizer, cfg, metadata)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


def cmd_run(args) -> None:
    cfg = resolve_run(args)
    run_id = cfg.runtime.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = run_directory(
        cfg.runtime.results_root,
        cfg.track,
        cfg.model.id,
        cfg.dataset.id,
        cfg.experiment.id,
        run_id,
    )
    if cfg.runtime.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "model": cfg.model.id,
                    "dataset": cfg.dataset.id,
                    "experiment": cfg.experiment.id,
                    "track": cfg.track,
                    "output": str(target),
                    "work": work_estimate(cfg),
                },
                indent=2,
            )
        )
        return

    audit = load_audit(cfg.dataset)
    command = " ".join(shlex.quote(piece) for piece in ["dlmrel", *sys.argv[1:]])
    initialize_run(target, cfg.to_dict(), command, audit["manifest_hashes"], resume=cfg.runtime.resume)
    if cfg.model.family == "fake":
        run_fake(cfg, target)
    else:
        run_real(cfg, target, audit["manifest_hashes"])
    validation = validate_run(target)
    if not validation["valid"]:
        raise ArtifactError("run failed validation: " + "; ".join(validation["errors"]))
    print(json.dumps({"run_dir": str(target), "validation": validation}, indent=2))


def work_estimate(cfg: RunConfig) -> dict:
    audit = manifest_root(cfg.dataset) / "audit.json"
    counts = json.loads(audit.read_text(encoding="utf-8")).get("counts", {}) if audit.exists() else {}
    sentences = sum(counts.values()) if counts else "unknown_until_prepare"
    steps = len(cfg.experiment.normalized_progress)
    return {
        "sentences": sentences,
        "trajectory_points": steps,
        "seeds": len(cfg.experiment.seeds),
        "estimated_forward_passes": (
            sentences if isinstance(sentences, str) else sentences * steps * len(cfg.experiment.seeds)
        ),
    }


def cmd_validate(args) -> None:
    validation = validate_run(args.run_dir)
    print(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


def cmd_compare(args) -> None:
    output, common = compare_runs(args.runs, args.output)
    print(f"wrote {output} and {common}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlmrel", description="Rigorous DLM relation analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="verify one treebank and write official manifests")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--no-download", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    smoke = commands.add_parser("smoke-test", help="verify one model adapter")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--dataset", default="configs/datasets/ewt.yaml")
    smoke.add_argument("--experiment")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--output")
    smoke.set_defaults(func=cmd_smoke_test)

    run = commands.add_parser("run", help="run or resume one experiment")
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--experiment", required=True)
    run.add_argument("--results", default="results")
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--selection-lock")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    validate = commands.add_parser("validate")
    validate.add_argument("--run-dir", required=True)
    validate.set_defaults(func=cmd_validate)

    compare = commands.add_parser("compare")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ArtifactError, ConfigError, FileNotFoundError, ValueError) as error:
        print(f"dlmrel: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

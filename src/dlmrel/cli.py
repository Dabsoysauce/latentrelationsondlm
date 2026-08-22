"""Command-line entry points for the rigorous DLM relation pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .artifacts import (
    ArtifactError,
    atomic_json,
    canonical_hash,
    initialize_run,
    run_directory,
    validate_run,
)
from .config import (
    ConfigError,
    DatasetConfig,
    RunConfig,
    RuntimeConfig,
    _strict_dataclass,
    is_paper_experiment,
)
from .data import load_audit, load_paper_manifest_refs, manifest_root, prepare_manifests
from .evaluation.compare_models import compare_runs
from .fake_run import run_fake
from .head_search_recovery import (
    complete_cpu_finalization,
    load_saved_head_search_config,
    recovery_status,
    score_missing_test_grid,
)
from .pipeline import load_adapter, model_smoke_report, run_real
from .relation_selection import derive_relation_selection_bundle


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

    if is_paper_experiment(cfg.experiment):
        roles = paper_manifest_roles(cfg)
        try:
            audit = load_paper_manifest_refs(cfg.dataset, roles)
        except ArtifactError:
            if cfg.model.family != "fake":
                raise
            audit = {
                "manifest_hashes": {
                    role: canonical_hash(
                        {"fake_cpu_only": True, "dataset": cfg.dataset.id, "role": role}
                    )
                    for role in roles
                }
            }
    else:
        audit = load_audit(cfg.dataset)
    command = " ".join(shlex.quote(piece) for piece in ["dlmrel", *sys.argv[1:]])
    initialize_run(target, cfg.to_dict(), command, audit["manifest_hashes"], resume=cfg.runtime.resume)
    if cfg.model.family == "fake" and not is_paper_experiment(cfg.experiment):
        run_fake(cfg, target)
    else:
        run_real(cfg, target, audit["manifest_hashes"])
    validation = validate_run(target)
    if not validation["valid"]:
        raise ArtifactError("run failed validation: " + "; ".join(validation["errors"]))
    print(json.dumps({"run_dir": str(target), "validation": validation}, indent=2))


def work_estimate(cfg: RunConfig) -> dict:
    if is_paper_experiment(cfg.experiment):
        counts = {}
        for role in paper_manifest_roles(cfg):
            path = manifest_root(cfg.dataset) / f"{role}.csv"
            if path.is_file():
                with path.open(encoding="utf-8") as stream:
                    counts[role] = max(sum(1 for _line in stream) - 1, 0)
    else:
        audit = manifest_root(cfg.dataset) / "audit.json"
        counts = (
            json.loads(audit.read_text(encoding="utf-8")).get("counts", {})
            if audit.exists()
            else {}
        )
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


def paper_manifest_roles(cfg: RunConfig) -> tuple[str, ...]:
    """Return the only UD roles a corrected runner may place in its identity."""
    if cfg.experiment.type in {
        "relation_head_receiver_prediction",
        "pos_token_class_linear_probes",
    }:
        return ("select", "test")
    return ("test",)


def cmd_validate(args) -> None:
    validation = validate_run(args.run_dir)
    print(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


def cmd_compare(args) -> None:
    output, common = compare_runs(args.runs, args.output)
    print(f"wrote {output} and {common}")


def cmd_validate_selection_locks(args) -> None:
    cfg = RunConfig.load_files(args.model, args.dataset, args.experiment)
    if not is_paper_experiment(cfg.experiment):
        raise ConfigError("selection-lock validation requires a corrected paper config")
    from .experiments.paper_relation import load_paper_locks

    locks = load_paper_locks(args.selection_lock, cfg)
    print(
        json.dumps(
            {
                "valid": True,
                "source": str(locks.source),
                "model": cfg.model.id,
                "model_revision": cfg.model.revision,
                "relations": {
                    relation: {"layer": lock.layer, "head": lock.head}
                    for relation, lock in locks.locks.items()
                },
            },
            indent=2,
        )
    )


def cmd_summarize(args) -> None:
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    payload = {"summary": summary}
    metrics = run_dir / "metrics.csv"
    if metrics.is_file():
        import pandas as pd

        payload["metrics_preview"] = pd.read_csv(metrics, nrows=args.rows).to_dict("records")
    print(json.dumps(payload, indent=2, default=str))


def cmd_derive_relation_locks(args) -> None:
    build = derive_relation_selection_bundle(args.source_run, args.output)
    statuses = {
        relation: record["status"] for relation, record in build.bundle["relations"].items()
    }
    print(
        json.dumps(
            {
                "output": str(build.output_dir),
                "source_run": str(Path(args.source_run).resolve()),
                "relation_statuses": statuses,
                "model_inference_performed": False,
                "test_artifacts_read": False,
            },
            indent=2,
        )
    )


def cmd_recover_head_search_test_grid(args) -> None:
    """Run only the missing Dream all-head test grid on an existing run."""
    run = Path(args.run_dir)
    cfg = load_saved_head_search_config(run)
    status = recovery_status(run, cfg)
    if not status["missing_test_grid_required"]:
        print(json.dumps({**status, "model_loaded": False}, indent=2, sort_keys=True))
        return
    summary_path = run / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("completion_status") == "complete":
            raise ArtifactError("a completed run cannot be given new all-head evidence")

    audit = load_audit(cfg.dataset)
    resumed = replace(cfg, runtime=replace(cfg.runtime, resume=True, dry_run=False))
    command = " ".join(shlex.quote(piece) for piece in ["dlmrel", *sys.argv[1:]])
    initialize_run(
        run,
        resumed.to_dict(),
        command,
        audit["manifest_hashes"],
        resume=True,
    )
    model, tokenizer, model_metadata = load_adapter(resumed)
    report, _locked, _exclusions = score_missing_test_grid(
        model,
        tokenizer,
        resumed,
        run,
        model_metadata=model_metadata,
        collect_locked_rows=False,
        reuse_existing_locked=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_finalize_head_search(args) -> None:
    """Finalize a saved head search from disk without loading a model/tokenizer."""
    run = Path(args.run_dir)
    cfg = load_saved_head_search_config(run)
    summary_path = run / "summary.json"
    already_complete = False
    if summary_path.is_file():
        try:
            already_complete = (
                json.loads(summary_path.read_text(encoding="utf-8")).get("completion_status")
                == "complete"
            )
        except json.JSONDecodeError as error:
            raise ArtifactError("existing head-search summary is unreadable") from error
    if not already_complete:
        manifests = json.loads((run / "manifest_refs.json").read_text(encoding="utf-8"))
        resumed = replace(cfg, runtime=replace(cfg.runtime, resume=True, dry_run=False))
        command = " ".join(shlex.quote(piece) for piece in ["dlmrel", *sys.argv[1:]])
        initialize_run(run, resumed.to_dict(), command, manifests, resume=True)
        cfg = resumed
    result = complete_cpu_finalization(
        cfg,
        run,
        n_permutations=10 if cfg.model.family == "fake" else 1000,
        checkpoint_interval=5 if cfg.model.family == "fake" else 50,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlmrel", description="Restored old-paper experiments for diffusion language models"
    )
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

    validate_locks = commands.add_parser(
        "validate-selection-locks",
        help="validate all six model-specific, selection-only paper locks",
    )
    validate_locks.add_argument("--model", required=True)
    validate_locks.add_argument("--dataset", default="configs/datasets/ewt.yaml")
    validate_locks.add_argument(
        "--experiment",
        default="configs/experiments/relation_head_receiver_prediction.yaml",
    )
    validate_locks.add_argument("--selection-lock", required=True)
    validate_locks.set_defaults(func=cmd_validate_selection_locks)

    summarize = commands.add_parser(
        "summarize", help="print summary.json and a small metrics preview without opening Parquet"
    )
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--rows", type=int, default=20)
    summarize.set_defaults(func=cmd_summarize)

    derive = commands.add_parser(
        "derive-relation-locks",
        help="derive six relation locks from a completed head-search run without inference",
    )
    derive.add_argument("--source-run", required=True)
    derive.add_argument("--output", required=True)
    derive.set_defaults(func=cmd_derive_relation_locks)

    recover_grid = commands.add_parser(
        "recover-head-search-test-grid",
        help="LEGACY ONLY: recover old permutation-era all-head test evidence",
    )
    recover_grid.add_argument("--run-dir", required=True)
    recover_grid.set_defaults(func=cmd_recover_head_search_test_grid)

    finalize = commands.add_parser(
        "finalize-head-search",
        help="LEGACY ONLY: finish old dev/permutation-era results",
    )
    finalize.add_argument("--run-dir", required=True)
    finalize.set_defaults(func=cmd_finalize_head_search)
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

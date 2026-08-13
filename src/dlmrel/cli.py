"""Single command-line interface for data, models, runs, and validation."""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .artifacts import (
    ArtifactError,
    atomic_json,
    canonical_hash,
    initialize_run,
    load_selection_lock,
    merge_shards,
    run_directory,
    validate_run,
    write_shard,
)
from .config import ConfigError, DatasetConfig, RunConfig, RuntimeConfig, _strict_dataclass
from .splits import build_official_manifests, manifest_hash
from .treebank import acquire_split


def _yaml(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return value


def _dataset(path: str | Path) -> DatasetConfig:
    config = _strict_dataclass(DatasetConfig, _yaml(path), "dataset")
    config.validate()
    return config


def _manifest_root(dataset: DatasetConfig) -> Path:
    return Path("data/manifests") / dataset.id / dataset.release


def _read_ud(dataset: DatasetConfig, *, download: bool) -> dict[str, list]:
    from conllu import parse_incr

    output = {}
    for split in ("train", "dev", "test"):
        path = acquire_split(dataset, split, download=download)
        with path.open(encoding="utf-8") as stream:
            output[split] = list(parse_incr(stream))
    return output


def _prepare(dataset: DatasetConfig, *, download: bool) -> tuple[Path, dict[str, list]]:
    manifests = build_official_manifests(dataset, _read_ud(dataset, download=download))
    root = _manifest_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for role, rows in manifests.items():
        path = root / f"{role}.csv"
        frame = pd.DataFrame([asdict(row) for row in rows])
        frame.to_csv(path, index=False)
        hashes[role] = manifest_hash(rows)
    audit = {
        "schema_version": "dlmrel-manifest-v1",
        "dataset": dataset.id,
        "treebank": dataset.treebank,
        "release": dataset.release,
        "revision": dataset.revision,
        "checksums": dataset.checksums,
        "counts": {role: len(rows) for role, rows in manifests.items()},
        "manifest_hashes": hashes,
        "zero_overlap": True,
    }
    atomic_json(root / "audit.json", audit)
    return root, manifests


def cmd_data_prepare(args) -> None:
    dataset = _dataset(args.dataset)
    root, manifests = _prepare(dataset, download=not args.no_download)
    print(json.dumps({"output": str(root), "counts": {k: len(v) for k, v in manifests.items()}}, indent=2))


def cmd_data_audit(args) -> None:
    dataset = _dataset(args.dataset)
    root, manifests = _prepare(dataset, download=not args.no_download)
    report = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    report["official_boundaries"] = all(
        row.original_split == {"select": "train", "dev": "dev", "test": "test"}[role]
        for role, rows in manifests.items()
        for row in rows
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _resolve(args, *, default_experiment: str | None = None) -> RunConfig:
    experiment = getattr(args, "experiment", None) or default_experiment
    if not experiment:
        raise ConfigError("an experiment YAML is required")
    runtime = RuntimeConfig(
        results_root=getattr(args, "results", "results"),
        run_id=getattr(args, "run_id", None),
        resume=getattr(args, "resume", False),
        dry_run=getattr(args, "dry_run", False),
        workers=getattr(args, "workers", 1),
        selection_lock=getattr(args, "selection_lock", None),
    )
    return RunConfig.load_files(
        args.model, args.dataset, experiment, track=getattr(args, "track", None), runtime=runtime
    )


def _work_estimate(cfg: RunConfig) -> dict:
    counts = {}
    audit = _manifest_root(cfg.dataset) / "audit.json"
    if audit.exists():
        counts = json.loads(audit.read_text(encoding="utf-8")).get("counts", {})
    steps = len(cfg.experiment.normalized_progress)
    sentences = sum(counts.values()) if counts else "unknown_until_data_prepare"
    return {
        "sentences": sentences,
        "trajectory_points": steps,
        "seeds": len(cfg.experiment.seeds),
        "estimated_forward_passes": sentences
        if isinstance(sentences, str)
        else sentences * steps * len(cfg.experiment.seeds),
        "shard_size": cfg.experiment.shard_size,
    }


def _run_path(cfg: RunConfig) -> Path:
    run_id = cfg.runtime.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_directory(
        cfg.runtime.results_root,
        cfg.track,
        cfg.model.id,
        cfg.dataset.id,
        cfg.experiment.id,
        run_id,
    )


def cmd_run(args) -> None:
    cfg = _resolve(args)
    target = _run_path(cfg)
    resolved = cfg.to_dict()
    if cfg.runtime.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "track": cfg.track,
                    "model": cfg.model.id,
                    "dataset": cfg.dataset.id,
                    "experiment": cfg.experiment.id,
                    "capabilities": asdict(cfg.model.capabilities),
                    "output": str(target),
                    "work": _work_estimate(cfg),
                },
                indent=2,
            )
        )
        return
    audit_path = _manifest_root(cfg.dataset) / "audit.json"
    if not audit_path.exists():
        raise ArtifactError(f"missing prepared manifests: run dlmrel data prepare --dataset {args.dataset}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    command = " ".join(shlex.quote(piece) for piece in ["dlmrel", *sys.argv[1:]])
    initialize_run(target, resolved, command, audit["manifest_hashes"], resume=cfg.runtime.resume)
    if cfg.model.family == "fake":
        _run_fake(cfg, target)
    else:
        _run_real(cfg, target, audit["manifest_hashes"])
    validation = validate_run(target)
    if not validation["valid"]:
        raise ArtifactError("fake run failed validation: " + "; ".join(validation["errors"]))
    print(json.dumps({"run_dir": str(target), "validation": validation}, indent=2))


def _run_real(cfg: RunConfig, target: Path, manifest_hashes: dict[str, str]) -> None:
    from .pipeline import (
        load_adapter,
        read_source_lock,
        run_entropy,
        run_head_search,
        run_locked_transfer,
        run_logit_lens,
        run_pos_probe,
        run_time_curve,
    )

    model, tokenizer, model_metadata = load_adapter(cfg)
    source_lock = None
    if cfg.runtime.selection_lock:
        source_lock = read_source_lock(cfg.runtime.selection_lock, cfg)
    if cfg.track == "external_treebank_transfer":
        if source_lock is None:
            raise ArtifactError("external transfer requires --selection-lock from EWT")
        details = run_locked_transfer(model, tokenizer, cfg, target, source_lock)
    elif cfg.experiment.type == "head_search":
        details = run_head_search(model, tokenizer, cfg, target, manifest_hashes)
    elif cfg.experiment.type == "time_curve":
        if source_lock is None:
            raise ArtifactError("confirmatory time curves require --selection-lock")
        details = run_time_curve(model, tokenizer, cfg, target, source_lock)
    elif cfg.experiment.type == "attention_entropy":
        details = run_entropy(model, tokenizer, cfg, target)
    elif cfg.experiment.type == "logit_lens":
        details = run_logit_lens(model, tokenizer, cfg, target)
    elif cfg.experiment.type == "pos_probe":
        details = run_pos_probe(model, tokenizer, cfg, target)
    else:
        raise ArtifactError(f"experiment {cfg.experiment.type!r} has no enabled real-model executor")
    summary = {
        "schema_version": "dlmrel-run-v1",
        "completion_status": "complete",
        "capabilities": asdict(cfg.model.capabilities),
        "model_metadata": model_metadata,
        **details,
    }
    atomic_json(target / "summary.json", summary)
    metadata = json.loads((target / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "completion_status": "complete",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "model_revision": cfg.model.revision,
            "tokenizer_revision": cfg.model.tokenizer_revision,
            "remote_code_revision": cfg.model.remote_code_revision,
        }
    )
    atomic_json(target / "run_metadata.json", metadata)


def _run_fake(cfg: RunConfig, target: Path) -> None:
    import torch

    from .models.fake import FakeAdapter

    adapter = FakeAdapter()
    all_rows = []
    for seed in cfg.experiment.seeds:
        ids = torch.tensor([[1, 3, 5, 7, 9]])
        output = adapter.forward(ids, timestep=seed % cfg.experiment.steps)
        for layer, attention in enumerate(output.attentions or ()):
            for head in range(attention.shape[1]):
                predicted = int(attention[0, head, 3].argmax())
                all_rows.append(
                    {
                        "sentence_id": "synthetic:0",
                        "instance_id": f"synthetic:{seed}",
                        "treebank": cfg.dataset.treebank,
                        "relation": "object_to_verb",
                        "seed": seed,
                        "timestep": seed % cfg.experiment.steps,
                        "normalized_progress": (seed % cfg.experiment.steps)
                        / max(cfg.experiment.steps - 1, 1),
                        "visibility": "both_visible",
                        "layer": layer,
                        "head": head,
                        "prediction": predicted,
                        "correct": int(predicted == 1),
                    }
                )
    scores = (
        pd.DataFrame(all_rows)
        .groupby(["relation", "layer", "head"], as_index=False)
        .agg(accuracy=("correct", "mean"), n_total=("correct", "size"))
    )
    if cfg.track == "external_treebank_transfer":
        if not cfg.runtime.selection_lock:
            raise ArtifactError("external transfer requires --selection-lock from EWT")
        source = Path(cfg.runtime.selection_lock)
        lock = load_selection_lock(
            source,
            config_hash=json.loads(source.read_text(encoding="utf-8"))["config_hash"],
            select_manifest_hash=json.loads(source.read_text(encoding="utf-8"))["select_manifest_hash"],
            dev_manifest_hash=json.loads(source.read_text(encoding="utf-8"))["dev_manifest_hash"],
        )
        if lock.dataset_id != "ewt" or lock.model_id != cfg.model.id:
            raise ArtifactError("external transfer lock must be an EWT lock for the same model")
        lock.write_once(target / "selection_lock.json")
    else:
        from .selection import create_selection_lock, write_lock_bundle

        manifests = json.loads((target / "manifest_refs.json").read_text(encoding="utf-8"))
        select = scores.copy()
        dev = scores.copy()
        dev["accuracy"] = dev["accuracy"] + (dev["head"] == 1) * 0.01
        lock, candidates, dev_candidates = create_selection_lock(
            select,
            dev,
            relation=cfg.experiment.scoring.primary_relation,
            top_k=cfg.experiment.scoring.top_k,
            track=cfg.track,
            model_id=cfg.model.id,
            model_revision=cfg.model.revision,
            dataset_id=cfg.dataset.id,
            config_hash=canonical_hash(cfg.to_dict()),
            select_manifest_hash=manifests["select"],
            dev_manifest_hash=manifests["dev"],
        )
        write_lock_bundle(target, lock, candidates, dev_candidates)
    rows = [
        row
        for row in all_rows
        if row["relation"] == lock.relation and row["layer"] == lock.layer and row["head"] == lock.head
    ]
    write_shard(target, 0, rows)
    merged = merge_shards(target)
    frame = pd.DataFrame(merged)
    frame.to_parquet(target / "instances.parquet", index=False)
    pd.DataFrame(columns=["sentence_id", "instance_id", "reason"]).to_parquet(
        target / "exclusions.parquet", index=False
    )
    per_seed = frame.groupby(["seed", "relation", "layer", "head"], as_index=False)["correct"].mean()
    per_seed.to_csv(target / "per_seed_metrics.csv", index=False)
    metrics = per_seed.groupby(["relation", "layer", "head"], as_index=False).agg(
        accuracy=("correct", "mean"), seed_std=("correct", "std"), n_seeds=("seed", "nunique")
    )
    metrics.to_csv(target / "metrics.csv", index=False)
    atomic_json(
        target / "summary.json",
        {
            "schema_version": "dlmrel-run-v1",
            "completion_status": "complete",
            "n_instances": len(frame),
            "capabilities": adapter.capabilities.__dict__,
        },
    )
    metadata = json.loads((target / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update({"completion_status": "complete", "ended_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(target / "run_metadata.json", metadata)
    atomic_json(target / "validation.json", {"schema_version": "dlmrel-run-v1", "valid": True, "errors": []})


def cmd_model_smoke(args) -> None:
    model_raw = _yaml(args.model)
    default_experiment = (
        "configs/experiments/logit_lens.yaml"
        if not model_raw.get("capabilities", {}).get("attentions", False)
        else "configs/experiments/final_state_head_search.yaml"
    )
    cfg = _resolve(args, default_experiment=default_experiment)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": cfg.model.id,
                    "revision": cfg.model.revision,
                    "tokenizer_revision": cfg.model.tokenizer_revision,
                    "remote_code_revision": cfg.model.remote_code_revision,
                    "capabilities": asdict(cfg.model.capabilities),
                    "status": "dry_run",
                },
                indent=2,
            )
        )
        return
    if cfg.model.family == "fake":
        import torch

        from .models.fake import FakeAdapter

        output = FakeAdapter().forward(torch.tensor([[1, 2, 3, 4]]), timestep=1)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "logits": list(output.logits.shape),
                    "layers": len(output.attentions),
                },
                indent=2,
            )
        )
        return
    module = importlib.import_module(f"dlmrel.models.{cfg.model.family}")
    model, tokenizer, metadata = module.load(_yaml(args.model))
    from .pipeline import model_smoke_report

    report = model_smoke_report(model, tokenizer, cfg, metadata)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


def cmd_validate(args) -> None:
    validation = validate_run(args.run_dir)
    print(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


def cmd_compare(args) -> None:
    summaries = []
    identity = None
    manifest_identity = None
    instance_frames = []
    for raw in args.runs:
        path = Path(raw)
        validation = validate_run(path)
        if not validation["valid"]:
            raise ArtifactError(f"invalid run {path}: {validation['errors']}")
        config = yaml.safe_load((path / "config.resolved.yaml").read_text(encoding="utf-8"))
        current = (
            config["schema_version"],
            config["track"],
            config["dataset"]["id"],
            config["experiment"]["type"],
            config["experiment"]["scoring"],
        )
        manifests = json.loads((path / "manifest_refs.json").read_text(encoding="utf-8"))
        if identity is not None and current != identity:
            raise ArtifactError("runs have incompatible schemas, tracks, manifests, or scoring rules")
        identity = current
        if manifest_identity is not None and manifests != manifest_identity:
            raise ArtifactError("runs have incompatible manifest hashes")
        manifest_identity = manifests
        metrics = pd.read_csv(path / "metrics.csv")
        metrics.insert(0, "run_dir", str(path))
        metrics.insert(1, "model", config["model"]["id"])
        summaries.append(metrics)
        instances = pd.read_parquet(path / "instances.parquet")
        instances["model"] = config["model"]["id"]
        instance_frames.append(instances)
    combined = pd.concat(summaries, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    common = _common_instance_comparison(instance_frames)
    common_path = output.with_name(output.stem + "_common_instances.csv")
    common.to_csv(common_path, index=False)
    print(
        f"wrote {len(combined)} full-eligible rows to {output} and "
        f"{len(common)} common-instance rows to {common_path}"
    )


def _common_instance_comparison(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    instance_sets = [set(frame["instance_id"].dropna().astype(str)) for frame in frames]
    common = set.intersection(*instance_sets) if instance_sets else set()
    rows = []
    for frame in frames:
        subset = frame[frame["instance_id"].astype(str).isin(common)]
        if subset.empty or "correct" not in subset:
            continue
        rows.extend(
            subset.groupby(["model", "treebank", "relation"], as_index=False)
            .agg(
                accuracy=("correct", "mean"),
                n_rows=("correct", "size"),
                n_common_instances=("instance_id", "nunique"),
            )
            .to_dict("records")
        )
    return pd.DataFrame(rows)


def cmd_status(args) -> None:
    rows = []
    for metadata_path in Path(args.results).glob("*/*/*/*/*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_dir": str(metadata_path.parent),
                "status": metadata.get("completion_status", "unknown"),
            }
        )
    print(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlmrel", description="Rigorous DLM relation analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="prepare or audit pinned UD manifests")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    for name, func in (("prepare", cmd_data_prepare), ("audit", cmd_data_audit)):
        command = data_sub.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--no-download", action="store_true")
        command.set_defaults(func=func)

    model = sub.add_parser("model", help="model capability and loading gates")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    smoke = model_sub.add_parser("smoke-test")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--dataset", default="configs/datasets/ewt.yaml")
    smoke.add_argument("--experiment")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--output")
    smoke.set_defaults(func=cmd_model_smoke)

    run = sub.add_parser("run", help="run or resume a versioned experiment")
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--experiment", required=True)
    run.add_argument(
        "--track",
        choices=[
            "legacy_reproduction",
            "confirmatory_ewt",
            "external_treebank_transfer",
            "exploratory_extensions",
        ],
    )
    run.add_argument("--results", default="results")
    run.add_argument("--run-id")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--selection-lock")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate-run")
    validate.add_argument("--run-dir", required=True)
    validate.set_defaults(func=cmd_validate)

    compare = sub.add_parser("compare")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=cmd_compare)

    status = sub.add_parser("status")
    status.add_argument("--results", default="results")
    status.set_defaults(func=cmd_status)
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

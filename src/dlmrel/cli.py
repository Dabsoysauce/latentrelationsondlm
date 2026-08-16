from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import (
    RELATION_NAMES,
    Config,
    DiffusionConfig,
    ModelConfig,
    TreebankConfig,
)

CONFIGS = Path("configs")
RESULTS = Path("results")

COMMON_POOL = [
    "diffusionfamily/diffugpt-s",
    "diffusionfamily/diffullama",
    "Dream-org/Dream-v0-Base-7B",
]

EXPERIMENTS = {
    "head_search": "dlmrel.experiments.head_search",
    "time_curve": "dlmrel.experiments.time_curve",
    "attention_entropy": "dlmrel.experiments.attention_entropy",
    "logit_lens": "dlmrel.experiments.logit_lens",
    "pos_probe": "dlmrel.experiments.pos_probe",
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _model_config(name: str) -> dict:
    return _load_yaml(CONFIGS / "models" / f"{name}.yaml")


def _experiment_config(name: str) -> dict:
    path = CONFIGS / "experiments" / f"{name}.yaml"
    return _load_yaml(path) if path.exists() else {}


def _apply(target, source: dict) -> None:
    """Copy recognised keys from a YAML block onto a dataclass instance.

    Unknown keys are ignored rather than raising: the experiment YAMLs carry a
    few descriptive keys (`classifier`, `measure_bos_mass`) that document the
    run without configuring it.
    """
    for key, value in source.items():
        if value is not None and hasattr(target, key):
            setattr(target, key, value)


def _build_config(model_name: str, model_cfg: dict, exp_cfg: dict) -> Config:
    checkpoint = model_cfg.get("checkpoint")
    if checkpoint is None:
        raise KeyError(
            f"configs/models/{model_name}.yaml has no `checkpoint`. Without it "
            f"the loader would try to resolve '{model_cfg.get('name')}' as a "
            "Hugging Face repo id."
        )

    model = ModelConfig(name=checkpoint, family=model_cfg.get("adapter", "dream"))
    _apply(model, {k: v for k, v in model_cfg.items() if k not in ("name", "checkpoint")})

    diffusion = DiffusionConfig()
    _apply(diffusion, exp_cfg)

    return Config(
        treebank=TreebankConfig(common_pool_models=COMMON_POOL),
        model=model,
        diffusion=diffusion,
        out_dir=str(RESULTS / model_name),
    )


def cmd_prepare_data(args) -> None:
    from .data import build_all_splits, load_tokenizer
    from .evaluation.metrics import build_null_table, offset_distribution
    from .relations import relations_to_records

    model_cfg = _model_config(args.model)
    cfg = _build_config(args.model, model_cfg, {})
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg.model.name)
    splits = build_all_splits(cfg, tokenizer)

    records = []
    for name, examples in splits.items():
        records.extend(relations_to_records(examples, name))
        pd.DataFrame({"split": name, "sentence": [e.text for e in examples]}).to_csv(
            out / f"sentences_{name}.csv", index=False
        )
    frame = pd.DataFrame(records)
    frame.to_csv(out / "relation_instances.csv", index=False)
    cfg.save(out / "config.yaml")

    table = build_null_table(
        frame[frame["split"] == "select"],
        frame[frame["split"] == "test"],
        list(RELATION_NAMES),
        cfg.analysis.offset_range,
        cfg.diffusion.attender_token,
        cfg.analysis.n_bootstrap,
        cfg.analysis.ci,
    )
    table.to_csv(out / "offset_null.csv", index=False)

    print(f"[prepare-data] {len(frame)} relation instances -> {out}")
    print(pd.crosstab(frame["relation"], frame["split"]))
    for relation in RELATION_NAMES:
        dist = offset_distribution(frame, relation)
        if not dist.empty:
            top = ", ".join(f"{int(k):+d}:{v:.0%}" for k, v in dist.nlargest(4).items())
            print(f"  {relation:22s} {top}")


def cmd_run(args) -> None:
    model_cfg = _model_config(args.model)
    exp_cfg = _experiment_config(args.experiment)
    cfg = _build_config(args.model, model_cfg, exp_cfg)

    adapter_mod = importlib.import_module(f"dlmrel.models.{model_cfg['adapter']}")
    model, tokenizer, meta = adapter_mod.load(model_cfg)

    (Path(cfg.out_dir) / "model_meta.json").write_text(json.dumps(meta, indent=2))

    experiment = importlib.import_module(EXPERIMENTS[args.experiment])
    out = Path(cfg.out_dir) / args.experiment

    # pos_probe needs the layer count, and reading it off the adapter differs
    # per family (the wrapped adapters hide the backbone behind the diffusion
    # module). The loader already reports it, so hand it over when accepted.
    extra = {}
    if "meta" in inspect.signature(experiment.run).parameters:
        extra["meta"] = meta
    experiment.run(model, tokenizer, cfg, out, **extra)


def cmd_compare(args) -> None:
    from .evaluation.compare_models import compare

    compare(args.models, args.experiment, RESULTS / "cross_model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dlmrel")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("prepare-data")
    p_data.add_argument("--model", required=True)
    p_data.set_defaults(func=cmd_prepare_data)

    p_run = sub.add_parser("run")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--experiment", required=True)
    p_cmp.add_argument("--models", required=True, nargs="+")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

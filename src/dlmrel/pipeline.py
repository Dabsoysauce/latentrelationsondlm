"""Small coordinator shared by local and Colab execution."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactError, atomic_json, final_artifact_hashes
from .config import RunConfig
from .relation_selection import RelationLockSet, load_relation_locks


def load_adapter(cfg: RunConfig):
    """Load one pinned model adapter and verify its declared capabilities."""
    if cfg.model.family == "fake":
        from .models.fake import FakeAdapter

        return FakeAdapter(), None, {"checkpoint": "fake", "revision": "local-v1"}
    module = importlib.import_module(f"dlmrel.models.{cfg.model.family}")
    model, tokenizer, metadata = module.load(asdict(cfg.model))
    actual = getattr(model, "capabilities", None)
    if actual is not None and actual.__dict__ != asdict(cfg.model.capabilities):
        raise ArtifactError("adapter capabilities differ from the pinned model config")
    return model, tokenizer, metadata


def read_source_locks(path: str | Path, cfg: RunConfig) -> RelationLockSet:
    return load_relation_locks(path, cfg)


def run_real(cfg: RunConfig, run_dir: Path, manifest_hashes: dict[str, str]) -> None:
    """Dispatch one validated configuration to its sole experiment runner."""
    source_locks = (
        read_source_locks(cfg.runtime.selection_lock, cfg) if cfg.runtime.selection_lock else None
    )
    model, tokenizer, model_metadata = load_adapter(cfg)

    if cfg.experiment.type == "head_search":
        from .experiments.head_search import run

        details = run(
            model,
            tokenizer,
            cfg,
            run_dir,
            manifest_hashes=manifest_hashes,
            source_locks=source_locks,
            model_metadata=model_metadata,
        )
    elif cfg.experiment.type == "time_curve":
        if source_locks is None:
            raise ArtifactError("time curves require --selection-lock from EWT head search")
        from .experiments.time_curve import run

        details = run(model, tokenizer, cfg, run_dir, source_locks=source_locks)
    elif cfg.experiment.type == "attention_entropy":
        from .experiments.attention_entropy import run

        details = run(model, tokenizer, cfg, run_dir)
    elif cfg.experiment.type == "logit_lens":
        from .experiments.logit_lens import run

        details = run(model, tokenizer, cfg, run_dir)
    elif cfg.experiment.type == "pos_probe":
        from .experiments.pos_probe import run

        details = run(model, tokenizer, cfg, run_dir)
    else:  # RunConfig validation should make this unreachable.
        raise ArtifactError(f"no runner for experiment {cfg.experiment.type!r}")

    atomic_json(
        run_dir / "summary.json",
        {
            "schema_version": "dlmrel-run-v1",
            "completion_status": "complete",
            "capabilities": asdict(cfg.model.capabilities),
            "model_metadata": model_metadata,
            **details,
        },
    )
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "completion_status": "complete",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "model_revision": cfg.model.revision,
            "tokenizer_revision": cfg.model.tokenizer_revision,
            "remote_code_revision": cfg.model.remote_code_revision,
            "final_artifact_hashes": final_artifact_hashes(run_dir),
        }
    )
    atomic_json(run_dir / "run_metadata.json", metadata)


@torch.no_grad()
def model_smoke_report(model, tokenizer, cfg: RunConfig, metadata: dict[str, Any]) -> dict[str, Any]:
    """Check deterministic shapes, attention normalization, and final-logit parity."""
    ids = tokenizer.encode("The chef cooked dinner.", add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id, *ids]
    input_ids = torch.tensor([ids], device=model.device)
    first = model.forward_attentions(input_ids, output_hidden_states=True)
    second = model.forward_attentions(input_ids, output_hidden_states=True)
    logits, attentions, hidden = first
    second_logits, second_attentions, second_hidden = second
    logits = model.get_logits(hidden[-1]) if logits is None else logits
    second_logits = model.get_logits(second_hidden[-1]) if second_logits is None else second_logits
    row_error = max(
        (layer.float().sum(dim=-1) - 1).abs().max().detach().item()
        for layer in attentions
    )
    determinism = max(
        [(logits.float() - second_logits.float()).abs().max().detach().item()]
        + [
            (left.float() - right.float()).abs().max().detach().item()
            for left, right in zip(attentions, second_attentions, strict=True)
        ]
    )
    if row_error > 1e-3:
        raise ArtifactError("attention rows do not sum to one")
    if determinism > 1e-5:
        raise ArtifactError("model is nondeterministic in evaluation mode")
    return {
        "status": "passed",
        "model": cfg.model.id,
        "checkpoint": cfg.model.name,
        "revision": cfg.model.revision,
        "tokenizer_revision": cfg.model.tokenizer_revision,
        "remote_code_revision": cfg.model.remote_code_revision,
        "capabilities": asdict(cfg.model.capabilities),
        "metadata": metadata,
        "logits_shape": list(logits.shape),
        "hidden_state_shapes": [list(value.shape) for value in hidden],
        "attention_shapes": [list(value.shape) for value in attentions],
        "attention_row_sum_max_error": row_error,
        "determinism_max_abs_error": determinism,
        "final_depth_logit_lens_max_abs_error": (
            logits.float() - model.get_lm_head()(hidden[-1]).float()
        ).abs().max().detach().item(),
    }

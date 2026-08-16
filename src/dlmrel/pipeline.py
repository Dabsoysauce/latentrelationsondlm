"""Small coordinator shared by local and Colab execution."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactError, SelectionLock, atomic_json
from .config import RunConfig


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


def read_source_lock(path: str | Path, cfg: RunConfig) -> SelectionLock:
    lock = SelectionLock(**json.loads(Path(path).read_text(encoding="utf-8")))
    if lock.dataset_id != "ewt":
        raise ArtifactError("selection lock must originate from EWT")
    if lock.model_id != cfg.model.id or lock.model_revision != cfg.model.revision:
        raise ArtifactError("selection lock model/revision mismatch")
    return lock


def _attention_normalization_report(attentions) -> dict[str, Any]:
    """Validate returned probabilities using the precision of their stored dtype."""
    if not attentions:
        raise ArtifactError("model returned no attention tensors")

    worst: dict[str, Any] | None = None
    worst_violation: dict[str, Any] | None = None
    for layer_index, layer in enumerate(attentions):
        if layer is None or not isinstance(layer, torch.Tensor):
            raise ArtifactError(f"attention layer {layer_index} is not a tensor")
        if layer.ndim != 4:
            raise ArtifactError(
                f"attention layer {layer_index} must have shape [batch, heads, query, key]; "
                f"got {list(layer.shape)}"
            )
        if not layer.is_floating_point():
            raise ArtifactError(f"attention layer {layer_index} is not floating point")

        values = layer.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ArtifactError(f"attention layer {layer_index} contains non-finite values")
        minimum = float(values.min())
        if minimum < 0.0:
            raise ArtifactError(
                f"attention layer {layer_index} contains a negative probability: {minimum:.12g}"
            )

        row_sums = values.sum(dim=-1)
        errors = (row_sums - 1.0).abs()
        flat_index = errors.argmax()
        coordinates = [int(value) for value in torch.unravel_index(flat_index, errors.shape)]
        batch_index, head_index, query_index = coordinates
        max_error = float(errors[batch_index, head_index, query_index])
        row_sum = float(row_sums[batch_index, head_index, query_index])

        # Dream returns FP32-softmax probabilities cast back to BF16. The
        # principled bound is unit roundoff for the stored dtype, plus one
        # FP32 epsilon for the softmax calculation itself.
        allowed_error = float(
            torch.finfo(layer.dtype).eps / 2 + torch.finfo(torch.float32).eps
        )
        candidate = {
            "attention_row_sum_max_error": max_error,
            "attention_row_sum_allowed_error": allowed_error,
            "attention_row_sum_value": row_sum,
            "attention_row_sum_dtype": str(layer.dtype),
            "attention_row_sum_location": {
                "layer": layer_index,
                "batch": batch_index,
                "head": head_index,
                "query": query_index,
            },
        }
        if worst is None or max_error > worst["attention_row_sum_max_error"]:
            worst = candidate
        ratio = max_error / allowed_error
        if ratio > 1.0 and (
            worst_violation is None or ratio > worst_violation["error_to_allowed_ratio"]
        ):
            worst_violation = {**candidate, "error_to_allowed_ratio": ratio}

    if worst_violation is not None:
        location = worst_violation["attention_row_sum_location"]
        raise ArtifactError(
            "attention rows do not sum to one: "
            f"max_abs_error={worst_violation['attention_row_sum_max_error']:.12g}, "
            f"allowed_error={worst_violation['attention_row_sum_allowed_error']:.12g}, "
            f"row_sum={worst_violation['attention_row_sum_value']:.12g}, "
            f"dtype={worst_violation['attention_row_sum_dtype']}, "
            f"layer={location['layer']}, batch={location['batch']}, "
            f"head={location['head']}, query={location['query']}"
        )
    assert worst is not None
    return worst


def run_real(cfg: RunConfig, run_dir: Path, manifest_hashes: dict[str, str]) -> None:
    """Dispatch one validated configuration to its sole experiment runner."""
    model, tokenizer, model_metadata = load_adapter(cfg)
    source_lock = read_source_lock(cfg.runtime.selection_lock, cfg) if cfg.runtime.selection_lock else None

    if cfg.experiment.type == "head_search":
        from .experiments.head_search import run

        details = run(
            model,
            tokenizer,
            cfg,
            run_dir,
            manifest_hashes=manifest_hashes,
            source_lock=source_lock,
        )
    elif cfg.experiment.type == "time_curve":
        if source_lock is None:
            raise ArtifactError("time curves require --selection-lock from EWT head search")
        from .experiments.time_curve import run

        details = run(model, tokenizer, cfg, run_dir, source_lock=source_lock)
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
        }
    )
    atomic_json(run_dir / "run_metadata.json", metadata)


@torch.inference_mode()
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
    normalization = _attention_normalization_report(attentions)
    determinism = max(
        [float((logits.float() - second_logits.float()).abs().max())]
        + [
            float((left.float() - right.float()).abs().max())
            for left, right in zip(attentions, second_attentions, strict=True)
        ]
    )
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
        **normalization,
        "determinism_max_abs_error": determinism,
        "final_depth_logit_lens_max_abs_error": float(
            (logits.float() - model.get_lm_head()(hidden[-1]).float()).abs().max()
        ),
    }

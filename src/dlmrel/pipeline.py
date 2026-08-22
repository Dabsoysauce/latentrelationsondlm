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
from .config import RunConfig, is_paper_experiment
from .paper_protocol import PaperLockSet
from .relation_selection import RelationLockSet, load_relation_locks

ATTENTION_ROW_SUM_TOLERANCE = 1e-2
_WORST_ROWS_PER_LAYER = 1
_WORST_ROWS_OVERALL = 5


def _unravel_index(index: int, shape: tuple[int, ...]) -> list[int]:
    coordinates: list[int] = []
    for size in reversed(shape):
        coordinates.append(index % size)
        index //= size
    return list(reversed(coordinates))


def _worst_attention_rows(
    row_sums: torch.Tensor,
    *,
    layer_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    flat_sums = row_sums.reshape(-1)
    flat_errors = (flat_sums - 1.0).abs()
    ranking_errors = torch.where(
        torch.isfinite(flat_errors),
        flat_errors,
        torch.full_like(flat_errors, torch.inf),
    )
    count = min(limit, flat_sums.numel())
    if count == 0:
        return []
    _, indices = torch.topk(ranking_errors, count, largest=True, sorted=True)
    rows = []
    for flat_index in indices.tolist():
        row_sum = flat_sums[flat_index].item()
        error = flat_errors[flat_index].item()
        rows.append(
            {
                "layer": layer_index,
                "row_index": _unravel_index(flat_index, tuple(row_sums.shape)),
                "row_sum": row_sum if torch.isfinite(flat_sums[flat_index]).item() else None,
                "abs_error_from_one": error if torch.isfinite(flat_errors[flat_index]).item() else None,
            }
        )
    return rows


def attention_normalization_diagnostics(
    attentions,
    *,
    sequence_length: int,
    tolerance: float = ATTENTION_ROW_SUM_TOLERANCE,
) -> dict[str, Any]:
    """Summarize attention row normalization without changing or repairing the values."""
    layer_reports: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    total_rows = 0
    total_bad_rows = 0
    global_max_error = 0.0

    for layer_index, layer in enumerate(attentions):
        values = layer.detach().float()
        row_sums = values.sum(dim=-1)
        finite = torch.isfinite(row_sums)
        finite_sums = row_sums[finite]
        errors = (row_sums - 1.0).abs()
        bad = ~finite | (errors > tolerance)
        finite_errors = errors[finite]
        layer_max_error = finite_errors.max().item() if finite_errors.numel() else None
        bad_rows = int(bad.sum().item())
        nonfinite_rows = int((~finite).sum().item())
        layer_worst = _worst_attention_rows(
            row_sums,
            layer_index=layer_index,
            limit=_WORST_ROWS_PER_LAYER,
        )
        layer_reports.append(
            {
                "layer": layer_index,
                "shape": list(layer.shape),
                "dtype": str(layer.dtype).removeprefix("torch."),
                "row_sum_min": finite_sums.min().item() if finite_sums.numel() else None,
                "row_sum_max": finite_sums.max().item() if finite_sums.numel() else None,
                "max_error_from_one": layer_max_error,
                "rows_exceeding_tolerance": bad_rows,
                "nonfinite_rows": nonfinite_rows,
                "total_rows": row_sums.numel(),
                "representative_worst_rows": layer_worst,
            }
        )
        worst_rows.extend(layer_worst)
        total_rows += row_sums.numel()
        total_bad_rows += bad_rows
        if layer_max_error is not None:
            global_max_error = max(global_max_error, layer_max_error)

    worst_rows.sort(
        key=lambda row: (
            row["abs_error_from_one"] is None,
            row["abs_error_from_one"] or 0.0,
        ),
        reverse=True,
    )
    attention_shapes_match_unpadded_input = all(
        len(report["shape"]) >= 2
        and report["shape"][-2] == sequence_length
        and report["shape"][-1] == sequence_length
        for report in layer_reports
    )
    return {
        "tolerance": tolerance,
        "passed": bool(layer_reports) and total_bad_rows == 0,
        "layer_count": len(layer_reports),
        "total_rows": total_rows,
        "rows_exceeding_tolerance": total_bad_rows,
        "max_error_from_one": global_max_error if layer_reports else None,
        "layers": layer_reports,
        "representative_worst_rows": worst_rows[:_WORST_ROWS_OVERALL],
        "mask_padding_assessment": {
            "input_padding_applied": False,
            "input_sequence_length": sequence_length,
            "attention_shapes_match_unpadded_input": attention_shapes_match_unpadded_input,
            "padding_can_explain_failure": False,
            "reason": (
                "The smoke input is one unpadded sequence. Causal or bidirectional key masks may "
                "zero individual attention entries, but valid softmax rows must still sum to one."
            ),
        },
    }


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


def read_source_locks(path: str | Path, cfg: RunConfig) -> RelationLockSet | PaperLockSet:
    if is_paper_experiment(cfg.experiment):
        from .experiments.paper_relation import load_paper_locks

        return load_paper_locks(path, cfg)
    return load_relation_locks(path, cfg)


def run_real(cfg: RunConfig, run_dir: Path, manifest_hashes: dict[str, str]) -> None:
    """Dispatch one validated configuration to its sole experiment runner."""
    source_locks = (
        read_source_locks(cfg.runtime.selection_lock, cfg) if cfg.runtime.selection_lock else None
    )
    model, tokenizer, model_metadata = load_adapter(cfg)

    if is_paper_experiment(cfg.experiment) and cfg.model.family == "fake":
        from .experiments.paper_fake import run

        details = run(cfg, run_dir, manifest_hashes=manifest_hashes, source_locks=source_locks)
    elif cfg.experiment.type == "relation_head_receiver_prediction":
        from .experiments.paper_relation import run_selection_and_test

        details = run_selection_and_test(
            model,
            tokenizer,
            cfg,
            run_dir,
            manifest_hashes=manifest_hashes,
            model_metadata=model_metadata,
        )
    elif cfg.experiment.type == "relation_head_receiver_prediction_over_diffusion_time":
        if source_locks is None:
            raise ArtifactError("paper time curves require --selection-lock")
        from .experiments.paper_relation import run_time_or_transfer

        details = run_time_or_transfer(
            model,
            tokenizer,
            cfg,
            run_dir,
            source_locks=source_locks,
            model_metadata=model_metadata,
        )
    elif cfg.experiment.type == "attention_entropy" and cfg.experiment.id == "attention_entropy":
        from .experiments.paper_entropy import run

        details = run(
            model,
            tokenizer,
            cfg,
            run_dir,
            manifest_hashes=manifest_hashes,
        )
    elif cfg.experiment.type == "pos_token_class_linear_probes":
        from .experiments.paper_pos import run

        details = run(model, tokenizer, cfg, run_dir)
    elif cfg.experiment.type == "final_token_prediction_by_layer":
        from .experiments.paper_native import run_final_token

        details = run_final_token(model, tokenizer, cfg, run_dir)
    elif cfg.experiment.type == "prediction_before_unmasking_timing_analysis":
        from .experiments.paper_native import run_timing

        details = run_timing(model, tokenizer, cfg, run_dir)
    elif cfg.experiment.type == "direct_logit_attribution":
        if source_locks is None:
            raise ArtifactError("Direct Logit Attribution requires --selection-lock")
        from .experiments.paper_causal import run_dla

        details = run_dla(model, tokenizer, cfg, run_dir, source_locks=source_locks)
    elif cfg.experiment.type == "matched_relation_head_ablation":
        if source_locks is None:
            raise ArtifactError("Matched Relation-Head Ablation requires --selection-lock")
        from .experiments.paper_causal import run_ablation

        details = run_ablation(model, tokenizer, cfg, run_dir, source_locks=source_locks)
    elif cfg.experiment.type == "attention_heatmaps_and_trajectories":
        if source_locks is None:
            raise ArtifactError("attention trajectories require --selection-lock")
        from .experiments.paper_visuals import run

        details = run(model, tokenizer, cfg, run_dir, source_locks=source_locks)
    elif cfg.experiment.type == "multilingual_relation_head_transfer":
        if source_locks is None:
            raise ArtifactError("multilingual transfer requires --selection-lock")
        from .experiments.paper_relation import run_time_or_transfer

        details = run_time_or_transfer(
            model,
            tokenizer,
            cfg,
            run_dir,
            source_locks=source_locks,
            transfer=True,
            model_metadata=model_metadata,
        )
    elif cfg.experiment.type == "head_search":
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
            "canonical_paper_protocol": is_paper_experiment(cfg.experiment),
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
    attention_diagnostics = attention_normalization_diagnostics(
        attentions,
        sequence_length=input_ids.shape[-1],
    )
    determinism = max(
        [(logits.float() - second_logits.float()).abs().max().detach().item()]
        + [
            (left.float() - right.float()).abs().max().detach().item()
            for left, right in zip(attentions, second_attentions, strict=True)
        ]
    )
    if not attention_diagnostics["passed"]:
        raise ArtifactError(
            "attention rows do not sum to one; diagnostics="
            + json.dumps(attention_diagnostics, sort_keys=True)
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
        "attention_row_sum_max_error": attention_diagnostics["max_error_from_one"],
        "attention_normalization": attention_diagnostics,
        "determinism_max_abs_error": determinism,
        "final_depth_logit_lens_max_abs_error": (
            logits.float() - model.get_lm_head()(hidden[-1]).float()
        ).abs().max().detach().item(),
    }

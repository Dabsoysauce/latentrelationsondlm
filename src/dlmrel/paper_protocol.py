"""Scientific invariants for the restored DiffuGPT-paper protocol.

This module is deliberately model-agnostic.  Expensive runners call these small,
pure helpers so the important definitions are unit-testable without a 7B model.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ArtifactError, atomic_json, canonical_hash
from .config import RELATION_NAMES

PAPER_SEEDS = (42, 43, 44)
PAPER_TIMESTEPS = tuple(range(64))
RELATIVE_DEPTHS = {"early": 0.20, "middle": 0.50, "late": 0.90}
CANONICAL_EXPERIMENTS = (
    "relation_head_receiver_prediction",
    "relation_head_receiver_prediction_over_diffusion_time",
    "attention_entropy",
    "pos_token_class_linear_probes",
    "final_token_prediction_by_layer",
    "prediction_before_unmasking_timing_analysis",
    "direct_logit_attribution",
    "matched_relation_head_ablation",
    "attention_heatmaps_and_trajectories",
    "multilingual_relation_head_transfer",
)

LOCK_SCHEMA = "dlmrel-paper-selection-lock-v1"
BUNDLE_SCHEMA = "dlmrel-paper-selection-bundle-v1"


def all_progress_points(steps: int = 64) -> list[float]:
    """Return one exact normalized coordinate for every discrete step."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    return [step / (steps - 1) for step in range(steps)]


def map_relative_depths(
    number_of_layers: int,
    relative_depths: dict[str, float] | None = None,
) -> list[dict[str, int | float | str]]:
    """Map named fractions to nearest valid zero-based transformer blocks."""
    if number_of_layers < 1:
        raise ValueError("number_of_layers must be positive")
    configured = relative_depths or RELATIVE_DEPTHS
    if set(configured) != {"early", "middle", "late"}:
        raise ValueError("relative depths must contain early, middle, and late")
    rows = []
    for label in ("early", "middle", "late"):
        fraction = float(configured[label])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"relative depth {label!r} is outside [0, 1]")
        rows.append(
            {
                "relative_label": label,
                "configured_fraction": fraction,
                "actual_layer_index": round(fraction * (number_of_layers - 1)),
                "total_model_layers": number_of_layers,
            }
        )
    return rows


def receiver_source_argmax(
    attention: np.ndarray,
    attender_span: Iterable[int],
    *,
    exclude_bos: bool = True,
) -> np.ndarray:
    """Old-code receiver prediction: final attender piece, one source argmax.

    ``attention`` may be ``[heads, query, source]`` or ``[query, source]``.
    BOS and the *complete* attender span are removed before the argmax.  No
    receiver-word aggregation is performed.
    """
    values = np.asarray(attention, dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("attention must have shape [heads, query, source]")
    span = tuple(int(index) for index in attender_span)
    if not span or min(span) < 0 or max(span) >= values.shape[-2]:
        raise ValueError("attender span is empty or outside the query axis")
    rows = values[:, span[-1], :].copy()
    if exclude_bos:
        rows[:, 0] = -np.inf
    rows[:, list(span)] = -np.inf
    if not np.isfinite(rows).any(axis=1).all():
        raise ValueError("receiver candidate set is empty")
    return rows.argmax(axis=1)


def receiver_is_correct(predicted_source: int, receiver_span: Iterable[int]) -> bool:
    """A predicted source piece is correct iff it lies anywhere in the gold span."""
    return int(predicted_source) in {int(index) for index in receiver_span}


def paper_visibility_group(
    is_visible: Iterable[bool],
    attender_span: Iterable[int],
    receiver_span: Iterable[int],
) -> str:
    """The paper's two-way endpoint visibility split.

    An instance is ``both_masked`` only when *no* piece of either endpoint is
    visible. One revealed piece immediately moves it to ``at_least_one_revealed``.
    """
    visibility = list(bool(value) for value in is_visible)
    endpoints = [int(index) for index in (*attender_span, *receiver_span)]
    if not endpoints or min(endpoints) < 0 or max(endpoints) >= len(visibility):
        raise ValueError("endpoint span is empty or outside the visibility vector")
    return (
        "at_least_one_revealed"
        if any(visibility[index] for index in endpoints)
        else "both_masked"
    )


def attention_entropy_rows(attention: np.ndarray) -> np.ndarray:
    """Old normalized entropy, excluding only the BOS *query* row.

    The BOS source column remains in the distribution.  Return one mean value
    per head after normalization by log(number of source positions).
    """
    probability = np.asarray(attention, dtype=float)
    if probability.ndim != 3:
        raise ValueError("attention must have shape [heads, query, source]")
    probability = probability / np.clip(probability.sum(axis=-1, keepdims=True), 1e-12, None)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, None))).sum(axis=-1)
    queries = entropy[:, 1:]  # exclude BOS query, retain BOS source column above
    denominator = math.log(probability.shape[-1]) if probability.shape[-1] > 1 else 1.0
    return queries.mean(axis=-1) / denominator if queries.shape[-1] else np.zeros(len(entropy))


def summarize_entropy_trajectory(
    values: Iterable[float],
    *,
    early_window: tuple[int, int],
    late_window: tuple[int, int],
    flat_threshold: float = 0.01,
) -> dict[str, float | str]:
    """Summarize the old early/late/delta/slope/direction evidence."""
    trajectory = np.asarray(list(values), dtype=float)
    if trajectory.ndim != 1 or len(trajectory) != 64 or not np.isfinite(trajectory).all():
        raise ValueError("entropy trajectory must contain 64 finite values")
    e0, e1 = early_window
    l0, l1 = late_window
    if not (0 <= e0 <= e1 < l0 <= l1 < len(trajectory)):
        raise ValueError("entropy windows must be ordered inclusive ranges")
    early = float(trajectory[e0 : e1 + 1].mean())
    late = float(trajectory[l0 : l1 + 1].mean())
    delta = late - early
    slope = float(np.polyfit(np.arange(len(trajectory)), trajectory, 1)[0])
    if delta > flat_threshold:
        direction = "increasing"
    elif delta < -flat_threshold:
        direction = "decreasing"
    else:
        direction = "flat"
    return {"early_entropy": early, "late_entropy": late, "delta": delta, "slope": slope,
            "direction": direction}


def prediction_source_index(target_position: int, prediction_offset: int) -> int:
    """Locate the representation that predicts a target under adapter semantics."""
    source = int(target_position) + int(prediction_offset)
    if source < 0:
        raise ValueError("target position has no valid prediction source")
    return source


def timing_record(
    argmax_history: Iterable[Iterable[int]],
    pre_forward_history: Iterable[Iterable[int]],
    final_ids: Iterable[int],
    *,
    mask_token_id: int,
    position: int,
) -> dict[str, int | bool | None]:
    """Compute the preserved found/unmask timing quantities for one position."""
    argmax = [list(step) for step in argmax_history]
    states = [list(step) for step in pre_forward_history]
    final = list(final_ids)
    if len(argmax) != 64 or len(states) != 64:
        raise ValueError("timing histories must cover all 64 pre-forward steps")
    target = int(final[position])
    found = next((step for step, ids in enumerate(argmax) if int(ids[position]) == target), None)
    unmask = next(
        (
            step
            for step, ids in enumerate(states)
            if int(ids[position]) != int(mask_token_id) and int(ids[position]) == target
        ),
        None,
    )
    delta = None if found is None or unmask is None else found - unmask
    return {
        "found_time": found,
        "unmask_time": unmask,
        "found_time_minus_unmask_time": delta,
        "lead_steps": None if delta is None else -delta,
        "predicted_before_unmasking": bool(delta is not None and delta < 0),
        "predicted_exactly_at_unmasking": bool(delta == 0),
    }


def projection_head_slice(
    concatenated_head_output: Any,
    projection_weight: Any,
    *,
    head: int,
    number_of_heads: int,
):
    """Exact per-head additive contribution through a linear output projection."""
    import torch

    values = concatenated_head_output
    weight = projection_weight
    if values.shape[-1] % number_of_heads:
        raise ValueError("attention output width is not divisible by the number of heads")
    head_width = values.shape[-1] // number_of_heads
    if not 0 <= head < number_of_heads or weight.shape[-1] != values.shape[-1]:
        raise ValueError("invalid head or projection shape")
    start, stop = head * head_width, (head + 1) * head_width
    return torch.nn.functional.linear(values[..., start:stop], weight[:, start:stop], None)


def zero_projection_head_input(values: Any, *, head: int, number_of_heads: int):
    """Clone a concatenated head tensor and zero exactly one requested slice."""
    if values.shape[-1] % number_of_heads:
        raise ValueError("attention output width is not divisible by the number of heads")
    head_width = values.shape[-1] // number_of_heads
    if not 0 <= head < number_of_heads:
        raise ValueError("head is outside the projection input")
    result = values.clone()
    result[..., head * head_width : (head + 1) * head_width] = 0
    return result


@dataclass(frozen=True)
class PaperSelectionLock:
    schema_version: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dataset_id: str
    selection_manifest_hash: str
    relation: str
    fully_visible_timestep: int
    scoring_method: str
    layer: int
    head: int
    selection_numerator: int
    selection_denominator: int
    selection_accuracy: float
    config_hash: str
    code_hash: str
    created_at: str


@dataclass(frozen=True)
class PaperLockSet:
    source: Path
    source_kind: str
    locks: dict[str, PaperSelectionLock]

    @property
    def heads(self) -> set[tuple[int, int]]:
        return {(lock.layer, lock.head) for lock in self.locks.values()}

    def resolve(self, relation: str) -> PaperSelectionLock:
        try:
            return self.locks[relation]
        except KeyError as error:
            raise ArtifactError(f"selection bundle has no lock for {relation!r}") from error


def choose_selection_winners(scores: pd.DataFrame) -> pd.DataFrame:
    """Choose each relation directly on fully-visible selection evidence."""
    required = {"relation", "layer", "head", "accuracy", "n_total", "n_correct"}
    if missing := required - set(scores):
        raise ValueError(f"selection scores missing columns: {sorted(missing)}")
    winners = []
    for relation in RELATION_NAMES:
        candidates = scores[scores["relation"] == relation].sort_values(
            ["accuracy", "n_total", "layer", "head"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        if candidates.empty:
            raise ValueError(f"no fully-visible selection evidence for {relation}")
        winners.append(candidates.iloc[0])
    return pd.DataFrame(winners).reset_index(drop=True)


def write_selection_bundle(
    path: str | Path,
    winners: pd.DataFrame,
    *,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    dataset_id: str,
    selection_manifest_hash: str,
    config_hash: str,
    code_hash: str,
    created_at: str | None = None,
) -> PaperLockSet:
    """Atomically publish six immutable selection-only locks."""
    path = Path(path)
    if path.exists():
        return load_selection_bundle(path, model_id=model_id, model_revision=model_revision)
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    (temporary / "locks").mkdir()
    locks: dict[str, PaperSelectionLock] = {}
    for row in winners.itertuples(index=False):
        lock = PaperSelectionLock(
            schema_version=LOCK_SCHEMA,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            dataset_id=dataset_id,
            selection_manifest_hash=selection_manifest_hash,
            relation=str(row.relation),
            fully_visible_timestep=63,
            scoring_method="last_attender_subtoken_single_source_argmax",
            layer=int(row.layer),
            head=int(row.head),
            selection_numerator=int(row.n_correct),
            selection_denominator=int(row.n_total),
            selection_accuracy=float(row.accuracy),
            config_hash=config_hash,
            code_hash=code_hash,
            created_at=timestamp,
        )
        atomic_json(temporary / "locks" / f"{lock.relation}.json", asdict(lock))
        locks[lock.relation] = lock
    if set(locks) != set(RELATION_NAMES):
        raise ArtifactError("selection bundle must contain exactly six relations")
    atomic_json(
        temporary / "selection_bundle.json",
        {
            "schema_version": BUNDLE_SCHEMA,
            "selection_only": True,
            "fully_visible_timestep": 63,
            "test_outcomes_used": False,
            "development_used": False,
            "relations": {
                relation: {
                    "lock": f"locks/{relation}.json",
                    "lock_hash": canonical_hash(asdict(lock)),
                }
                for relation, lock in locks.items()
            },
        },
    )
    os.replace(temporary, path)
    return PaperLockSet(path.resolve(), "paper_selection_only", locks)


def load_selection_bundle(
    path: str | Path,
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> PaperLockSet:
    source = Path(path).resolve()
    if source.is_file():
        source = source.parent
    manifest_path = source / "selection_bundle.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"missing paper selection bundle: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("selection_only") is not True
        or manifest.get("development_used") is not False
        or manifest.get("test_outcomes_used") is not False
        or set(manifest.get("relations", {})) != set(RELATION_NAMES)
    ):
        raise ArtifactError("selection bundle is scientifically incompatible")
    locks = {}
    for relation, record in manifest["relations"].items():
        raw = json.loads((source / record["lock"]).read_text(encoding="utf-8"))
        lock = PaperSelectionLock(**raw)
        if (
            lock.schema_version != LOCK_SCHEMA
            or lock.relation != relation
            or lock.fully_visible_timestep != 63
            or lock.scoring_method != "last_attender_subtoken_single_source_argmax"
            or canonical_hash(raw) != record["lock_hash"]
        ):
            raise ArtifactError(f"invalid or stale selection lock for {relation}")
        if model_id is not None and lock.model_id != model_id:
            raise ArtifactError("selection lock belongs to a different model")
        if model_revision is not None and lock.model_revision != model_revision:
            raise ArtifactError("selection lock belongs to a different model revision")
        locks[relation] = lock
    return PaperLockSet(source, "paper_selection_only", locks)


def write_resolved_selection_locks(run_dir: str | Path, locks: PaperLockSet) -> None:
    atomic_json(
        Path(run_dir) / "selection_locks.resolved.json",
        {
            "schema_version": BUNDLE_SCHEMA,
            "source": str(locks.source),
            "relations": {
                relation: {"layer": lock.layer, "head": lock.head}
                for relation, lock in locks.locks.items()
            },
        },
    )

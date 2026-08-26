"""Primary-condition-only adaptive POS probes for a sub-five-hour CPU budget.

This protocol deliberately narrows the claim scope to normalized progress 0.5
at the middle relative depth.  It keeps the unbiased all-head seed-42 screen,
freezes a favorable-result-resistant candidate set, confirms candidates on
held-out seeds, and computes canonical controls only where the paper still
needs them.  Its artifacts and ranking-only checkpoints are isolated from the
full-depth adaptive protocols.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from ..artifacts import ArtifactError, atomic_json, canonical_hash, dataframe_records
from . import paper_pos
from .paper_pos_adaptive import (
    HEADS,
    MIN_CONFIRMATION_RESERVE_FRACTION,
    PRIMARY_DEPTH,
    PRIMARY_PROGRESS,
    SCREEN_SEED,
    FitKey,
    _aggregate_rankings,
    _confirmation,
    _fit_keys,
    _metric_frame,
    _ResourceMonitor,
    _run_probe_batch,
    _saved_context,
    candidate_set,
    full_head_grid,
    inventory,
)

SCHEMA = "dlmrel-pos-primary-adaptive-v1"
OUTPUT_NAME = "pos_adaptive_primary"
CONFIRMATION_SEEDS = (43, 44)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


class _PrimaryMainCheckpointStore:
    """Atomic ranking-only checkpoints, isolated from canonical controls."""

    def __init__(self, run_dir: Path):
        cfg, manifests, _canonical, _depths = _saved_context(run_dir)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        self.directory = run_dir / OUTPUT_NAME / "main_fit_checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.expected_common = {
            "schema_version": SCHEMA,
            "scientific_config_hash": metadata.get("scientific_config_hash"),
            "manifest_hashes": manifests,
            "fixed_regularization_c": float(cfg.experiment.settings["fixed_regularization_c"]),
            "label_inventory": list(paper_pos.LABELS),
            "ranking_classifier_only": True,
        }

    def _paths(self, key: FitKey):
        slug = f"seed-{key.seed}__p-{key.progress:.6f}__{key.relative_label}__{key.feature_kind}"
        path = self.directory / f"{slug}.parquet"
        return path, path.with_suffix(".meta.json")

    def _expected(self, key: FitKey):
        return {
            **self.expected_common,
            "seed": key.seed,
            "normalized_progress": key.progress,
            "relative_label": key.relative_label,
            "feature_kind": key.feature_kind,
        }

    def load(self, key: FitKey):
        path, meta_path = self._paths(key)
        if not path.is_file() or not meta_path.is_file():
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            evidence = pd.read_parquet(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        expected = self._expected(key)
        if any(metadata.get(name) != value for name, value in expected.items()):
            return None
        if metadata.get("row_count") != len(evidence):
            return None
        return evidence, metadata["metrics"]

    def store(self, key: FitKey, evidence: pd.DataFrame, metrics: dict) -> None:
        path, meta_path = self._paths(key)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        evidence.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        atomic_json(
            meta_path,
            {
                **self._expected(key),
                "row_count": len(evidence),
                "metrics": metrics,
            },
        )


def _run_main_batch(tasks, workers: int):
    started = time.perf_counter()
    context = threadpool_limits(limits=1) if workers > 1 else nullcontext()
    with context:
        if workers <= 1:
            outputs = [
                paper_pos._fit_evaluate_main_only(train, test, **kwargs) for train, test, kwargs in tasks
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(paper_pos._fit_evaluate_main_only, train, test, **kwargs)
                    for train, test, kwargs in tasks
                ]
                outputs = [future.result() for future in futures]
    return outputs, time.perf_counter() - started


def _benchmark(run_dir: Path, cache_root, worker_counts: tuple[int, ...]):
    output = run_dir / OUTPUT_NAME
    path = output / "benchmark.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == SCHEMA
            and existing.get("sequential_parallel_exact") is True
            and existing.get("main_full_exact") is True
            and existing.get("worker_counts") == list(worker_counts)
        ):
            return existing
        raise ArtifactError(
            "saved primary benchmark differs from this runtime; preserve it and resume "
            "with the original worker-count list"
        )
    cfg, _manifests, _store, _depths = _saved_context(run_dir)
    load_started = time.perf_counter()
    selection, test = paper_pos._load_condition_frames(
        run_dir, seed=SCREEN_SEED, progress=PRIMARY_PROGRESS, cache_root=cache_root
    )
    load_seconds = time.perf_counter() - load_started
    identities = [(PRIMARY_DEPTH, f"head_{head}") for head in range(12)]
    selection_groups = selection.groupby(["relative_label", "feature_kind"], observed=True, sort=True)
    test_groups = test.groupby(["relative_label", "feature_kind"], observed=True, sort=True)
    regularization = float(cfg.experiment.settings["fixed_regularization_c"])
    tasks = [
        (
            selection_groups.get_group(identity),
            test_groups.get_group(identity),
            {"seed": SCREEN_SEED, "regularization": regularization},
        )
        for identity in identities
    ]
    sequential, sequential_seconds = _run_probe_batch(tasks[:1], 1)
    full_trials = []
    full_outputs = {}
    for workers in worker_counts:
        with _ResourceMonitor() as resources:
            outputs, seconds = _run_probe_batch(tasks, workers)
        full_outputs[workers] = outputs
        full_trials.append(
            {
                "workers": workers,
                "fits": len(tasks),
                "wall_seconds": seconds,
                "seconds_per_fit_throughput": seconds / len(tasks),
                "fits_per_hour": 3600.0 * len(tasks) / seconds,
                **resources.summary(),
            }
        )
    chosen_full = min(full_trials, key=lambda row: (row["wall_seconds"], row["workers"]))
    full_parallel = full_outputs[int(chosen_full["workers"])][0]
    exact = sequential[0][1] == full_parallel[1] and sequential[0][0].equals(full_parallel[0])
    if not exact:
        raise ArtifactError("full-probe sequential and parallel benchmark results differ")

    main_trials = []
    main_outputs = {}
    for workers in worker_counts:
        with _ResourceMonitor() as resources:
            outputs, seconds = _run_main_batch(tasks, workers)
        main_outputs[workers] = outputs
        main_trials.append(
            {
                "workers": workers,
                "fits": len(tasks),
                "wall_seconds": seconds,
                "seconds_per_fit_throughput": seconds / len(tasks),
                "fits_per_hour": 3600.0 * len(tasks) / seconds,
                **resources.summary(),
            }
        )
    chosen_main = min(main_trials, key=lambda row: (row["wall_seconds"], row["workers"]))
    main_first = main_outputs[int(chosen_main["workers"])][0]
    main_full_exact = main_first[1] == full_parallel[1]
    if main_full_exact:
        main_full_exact = main_first[0]["prediction"].equals(full_parallel[0]["prediction"]) and main_first[
            0
        ]["majority_prediction"].equals(full_parallel[0]["majority_prediction"])
    if not main_full_exact:
        raise ArtifactError("ranking-only and canonical main predictions differ")
    payload = {
        "schema_version": SCHEMA,
        "worker_counts": list(worker_counts),
        "logical_cpus": os.cpu_count(),
        "feature_load_seconds": load_seconds,
        "sequential_representative_full_fit_seconds": sequential_seconds,
        "sequential_parallel_exact": True,
        "main_full_exact": True,
        "full_trials": full_trials,
        "main_trials": main_trials,
        "selected_full_workers": int(chosen_full["workers"]),
        "selected_main_workers": int(chosen_main["workers"]),
        "selected_full_seconds_per_fit": float(chosen_full["seconds_per_fit_throughput"]),
        "selected_main_seconds_per_fit": float(chosen_main["seconds_per_fit_throughput"]),
        "benchmark_full_fits": 1 + len(tasks) * len(worker_counts),
        "benchmark_main_fits": len(tasks) * len(worker_counts),
    }
    atomic_json(path, payload)
    return payload


def _main_inventory(store: _PrimaryMainCheckpointStore, keys: set[FitKey]):
    completed = {}
    for key in sorted(keys, key=lambda value: value.tuple()):
        cached = store.load(key)
        if cached is not None:
            completed[key] = cached
    return completed


def _fit_main_keys(
    run_dir: Path,
    keys: set[FitKey],
    *,
    workers: int,
    cache_root,
    deadline: float,
    seconds_per_fit: float,
    phase: str,
):
    cfg, _manifests, _canonical, _depths = _saved_context(run_dir)
    store = _PrimaryMainCheckpointStore(run_dir)
    completed = _main_inventory(store, keys)
    missing = [key for key in sorted(keys, key=lambda value: value.tuple()) if key not in completed]
    for condition in sorted({(key.seed, key.progress) for key in missing}):
        current = [key for key in missing if (key.seed, key.progress) == condition]
        expected = seconds_per_fit * len(current) + 30.0
        if time.monotonic() + expected > deadline:
            raise TimeoutError(
                f"budget guard stopped before {phase}: {len(current)} ranking fits need ~{expected:.0f}s"
            )
        seed, progress = condition
        selection, test = paper_pos._load_condition_frames(
            run_dir, seed=seed, progress=progress, cache_root=cache_root
        )
        identities = {(key.relative_label, key.feature_kind) for key in current}
        selection_groups = {
            identity: frame
            for identity, frame in selection.groupby(
                ["relative_label", "feature_kind"], observed=True, sort=True
            )
            if identity in identities
        }
        test_groups = {
            identity: frame
            for identity, frame in test.groupby(["relative_label", "feature_kind"], observed=True, sort=True)
            if identity in identities
        }
        if set(selection_groups) != identities or set(test_groups) != identities:
            raise ArtifactError("a requested primary ranking feature group is missing")
        tasks = [
            (
                selection_groups[(key.relative_label, key.feature_kind)],
                test_groups[(key.relative_label, key.feature_kind)],
                {
                    "seed": key.seed,
                    "regularization": float(cfg.experiment.settings["fixed_regularization_c"]),
                },
            )
            for key in current
        ]
        outputs, seconds = _run_main_batch(tasks, workers)
        for index, (key, result) in enumerate(zip(current, outputs, strict=True), start=1):
            store.store(key, *result)
            completed[key] = result
            print(
                f"[{phase}] checkpointed {index}/{len(current)} seed={seed} "
                f"p={progress:.2f}; batch={seconds:.1f}s",
                flush=True,
            )
    return _main_inventory(store, keys)


def _combined_results(run_dir: Path, main, keys: set[FitKey]):
    _cfg, canonical = inventory(run_dir)
    combined = {}
    for key in keys:
        if key in canonical:
            combined[key] = canonical[key]
        elif key in main:
            combined[key] = main[key]
    return combined, canonical


def _coverage(cfg, canonical, main, phase_reason):
    rows = []
    for key in full_head_grid(cfg):
        canonical_full = key in canonical
        main_only = key in main and not canonical_full
        phase, reason = phase_reason.get(
            key,
            ("reused_auxiliary", "pre-existing fit retained but outside primary claim scope"),
        )
        included = canonical_full or main_only
        rows.append(
            {
                "seed": key.seed,
                "normalized_progress": key.progress,
                "relative_label": key.relative_label,
                "feature_kind": key.feature_kind,
                "included": included,
                "evidence_tier": (
                    "canonical_full_controls"
                    if canonical_full
                    else "ranking_main_only"
                    if main_only
                    else "omitted"
                ),
                "phase": phase if included else "omitted",
                "reason": reason if included else "not required by primary-only protocol",
                "supports_primary_claim": key in phase_reason,
            }
        )
    return pd.DataFrame(rows)


def run_primary_adaptive(
    run_dir: str | Path,
    *,
    cache_root=None,
    budget_seconds: int = 5 * 60 * 60,
    validation_reserve_seconds: int = 15 * 60,
    worker_counts: tuple[int, ...] = (6, 8, 10, 12),
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output = run_dir / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    cfg, _manifests, _canonical_store, depths = _saved_context(run_dir)
    if budget_seconds <= validation_reserve_seconds:
        raise ValueError("fit budget must exceed validation reserve")
    if set(map(int, cfg.experiment.seeds)) != {42, 43, 44}:
        raise ArtifactError("primary adaptive protocol requires saved seeds 42, 43, and 44")
    if PRIMARY_PROGRESS not in set(map(float, cfg.experiment.normalized_progress)):
        raise ArtifactError("saved extraction lacks primary progress 0.5")
    started = time.monotonic()
    deadline = started + budget_seconds - validation_reserve_seconds
    preregistration = {
        "schema_version": SCHEMA,
        "screen_seed": SCREEN_SEED,
        "held_out_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "progress": PRIMARY_PROGRESS,
        "relative_depth": PRIMARY_DEPTH,
        "screen_all_32_heads": True,
        "candidate_rule": (
            "top four + bottom four + every head within max(1 percentage point, "
            "binomial 95% half-width) of either core boundary"
        ),
        "confirmation_rule": (
            "held-out seeds 43/44; high and low remain in the corresponding top/bottom "
            "three for every seed; median pairwise Spearman >= 0.5; nonzero margins"
        ),
        "controls_scope": (
            "canonical shuffled-label and random-feature controls for the complete seed-42 "
            "screen, the final high/low pair on held-out seeds, and the three-seed residual"
        ),
        "stability_expansion": (
            "if candidate confirmation fails, rank all 32 heads on held-out seeds without "
            "examining early/late or other progress conditions"
        ),
        "minimum_confirmation_reserve_fraction": MIN_CONFIRMATION_RESERVE_FRACTION,
        "claim_scope": "middle depth at normalized progress 0.5 only",
        "claims_removed": [
            "early-versus-late POS depth comparison",
            "multi-progress POS trajectory",
            "early/late matched causal pairs",
        ],
    }
    preregistration["preregistration_hash"] = canonical_hash(preregistration)
    prereg_path = output / "preregistration.json"
    if prereg_path.is_file():
        if json.loads(prereg_path.read_text(encoding="utf-8")) != preregistration:
            raise ArtifactError("saved primary-only preregistration differs")
    else:
        atomic_json(prereg_path, preregistration)

    benchmark = _benchmark(run_dir, cache_root, worker_counts)
    full_workers = int(benchmark["selected_full_workers"])
    main_workers = int(benchmark["selected_main_workers"])
    full_seconds = float(benchmark["selected_full_seconds_per_fit"])
    main_seconds = float(benchmark["selected_main_seconds_per_fit"])
    phase_reason = {}

    screen_keys = {FitKey(SCREEN_SEED, PRIMARY_PROGRESS, PRIMARY_DEPTH, head) for head in HEADS}
    residual_keys = {FitKey(seed, PRIMARY_PROGRESS, PRIMARY_DEPTH, "residual") for seed in (42, 43, 44)}
    for key in screen_keys:
        phase_reason[key] = (
            "A_unbiased_full_control_screen",
            "all 32 heads fixed before seed-42 outcomes were examined; canonical controls included",
        )
    _fit_keys(
        run_dir,
        screen_keys | residual_keys,
        workers=full_workers,
        cache_root=cache_root,
        deadline=deadline,
        predicted_seconds_per_fit=full_seconds,
        phase="primary-screen-and-residual",
    )
    _cfg, canonical = inventory(run_dir)
    screen = _metric_frame({key: canonical[key] for key in screen_keys})
    candidates, reasons = candidate_set(screen)
    selection = {
        "schema_version": SCHEMA,
        "preregistration_hash": preregistration["preregistration_hash"],
        "candidates": sorted(candidates),
        "reasons": reasons,
    }
    selection["selection_hash"] = canonical_hash(selection)
    selection_path = output / "candidate_selection.json"
    if selection_path.is_file():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ArtifactError("saved primary candidate selection differs")
    else:
        atomic_json(selection_path, selection)

    confirmation_keys = {
        FitKey(seed, PRIMARY_PROGRESS, PRIMARY_DEPTH, head)
        for head in candidates
        for seed in CONFIRMATION_SEEDS
    }
    for key in confirmation_keys:
        phase_reason[key] = (
            "B_held_out_ranking_confirmation",
            "candidate set frozen from seed 42 before held-out seeds were read",
        )
    reserve = len(confirmation_keys) / len(screen_keys | confirmation_keys)
    if reserve < MIN_CONFIRMATION_RESERVE_FRACTION:
        raise ArtifactError("primary allocation violates held-out confirmation reserve")
    main = _fit_main_keys(
        run_dir,
        confirmation_keys,
        workers=main_workers,
        cache_root=cache_root,
        deadline=deadline,
        seconds_per_fit=main_seconds,
        phase="candidate-confirmation",
    )
    protocol_keys = screen_keys | confirmation_keys
    combined, canonical = _combined_results(run_dir, main, protocol_keys)
    metrics = _metric_frame(combined)
    confirmation = _confirmation(PRIMARY_DEPTH, metrics, set(candidates))
    expanded = False
    if not confirmation["confirmed"]:
        expanded = True
        expansion_keys = {
            FitKey(seed, PRIMARY_PROGRESS, PRIMARY_DEPTH, head)
            for seed in CONFIRMATION_SEEDS
            for head in HEADS
        }
        for key in expansion_keys:
            phase_reason[key] = (
                "C_preregistered_stability_expansion",
                "candidate confirmation was unstable; all held-out heads ranked fail-safely",
            )
        main = _fit_main_keys(
            run_dir,
            expansion_keys,
            workers=main_workers,
            cache_root=cache_root,
            deadline=deadline,
            seconds_per_fit=main_seconds,
            phase="stability-expansion",
        )
        protocol_keys = screen_keys | expansion_keys
        combined, canonical = _combined_results(run_dir, main, protocol_keys)
        metrics = _metric_frame(combined)
        candidates = set(HEADS)
        confirmation = _confirmation(PRIMARY_DEPTH, metrics, candidates)

    confirmed = bool(confirmation["confirmed"])
    if not confirmed:
        rankings = _aggregate_rankings(metrics)
        _atomic_csv(output / "pos_head_rankings_primary_provisional.csv", rankings)
        manifest = {
            "schema_version": SCHEMA,
            "status": "blocked_unstable_rankings",
            "adaptive_reduced_grid": True,
            "matched_causal_ablation_allowed": False,
            "confirmed_depths": [],
            "confirmation": confirmation,
            "expanded_to_all_held_out_heads": expanded,
            "elapsed_seconds": time.monotonic() - started,
            "preregistration_hash": preregistration["preregistration_hash"],
            "candidate_selection_hash": selection["selection_hash"],
        }
        atomic_json(output / "adaptive_manifest.json", manifest)
        raise ArtifactError("primary-condition rankings failed held-out confirmation")

    high = str(confirmation["high_feature_kind"])
    low = str(confirmation["low_feature_kind"])
    selected_control_keys = {
        FitKey(seed, PRIMARY_PROGRESS, PRIMARY_DEPTH, head) for seed in (42, 43, 44) for head in (high, low)
    }
    for key in selected_control_keys:
        phase_reason[key] = (
            "D_selected_pair_full_controls",
            "confirmed high/low pair receives canonical shuffled-label and random-feature controls",
        )
    _fit_keys(
        run_dir,
        selected_control_keys,
        workers=full_workers,
        cache_root=cache_root,
        deadline=deadline,
        predicted_seconds_per_fit=full_seconds,
        phase="selected-pair-controls",
    )
    combined, canonical = _combined_results(run_dir, main, protocol_keys)
    metrics = _metric_frame(combined)
    for key in selected_control_keys & set(main):
        canonical_metrics = canonical[key][1]
        main_metrics = main[key][1]
        for name in (
            "accuracy",
            "macro_f1",
            "majority_accuracy",
            "n_positions",
            "class_counts",
        ):
            if canonical_metrics[name] != main_metrics[name]:
                raise ArtifactError("ranking-only and canonical selected-pair metrics differ")

    layer = int(depths.set_index("relative_label").loc[PRIMARY_DEPTH, "actual_layer_index"])
    choices = pd.DataFrame(
        [
            {
                "relative_label": PRIMARY_DEPTH,
                "actual_layer_index": layer,
                **confirmation,
            }
        ]
    )
    rankings = _aggregate_rankings(metrics)
    rankings["primary_condition_only"] = True
    residual_rows = []
    for key in sorted(residual_keys, key=lambda value: value.tuple()):
        residual_rows.append(
            {
                "seed": key.seed,
                "normalized_progress": key.progress,
                "relative_label": key.relative_label,
                "feature_kind": key.feature_kind,
                **canonical[key][1],
            }
        )
    residual = pd.DataFrame(residual_rows)
    coverage = _coverage(cfg, canonical, main, phase_reason)
    _atomic_csv(output / "pos_head_choices.csv", choices)
    _atomic_csv(output / "pos_head_rankings_primary.csv", rankings)
    _atomic_csv(output / "per_seed_metrics_primary.csv", metrics)
    _atomic_csv(output / "primary_residual_metrics.csv", residual)
    _atomic_csv(output / "coverage_manifest.csv", coverage)
    saved_choices = pd.read_csv(output / "pos_head_choices.csv")
    saved_rankings = pd.read_csv(output / "pos_head_rankings_primary.csv")
    saved_coverage = pd.read_csv(output / "coverage_manifest.csv")
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": SCHEMA,
        "status": "confirmed",
        "adaptive_reduced_grid": True,
        "primary_condition_only": True,
        "matched_causal_ablation_allowed": True,
        "confirmed_depths": [PRIMARY_DEPTH],
        "progress_values": [PRIMARY_PROGRESS],
        "screen_seed": SCREEN_SEED,
        "held_out_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "full_workers": full_workers,
        "main_workers": main_workers,
        "budget_seconds": budget_seconds,
        "validation_reserve_seconds": validation_reserve_seconds,
        "elapsed_seconds": elapsed,
        "screen_full_control_fits": len(screen_keys),
        "confirmation_ranking_fits": len(protocol_keys - screen_keys),
        "selected_pair_full_control_fits": len(selected_control_keys),
        "primary_residual_fits": len(residual_keys),
        "expanded_to_all_held_out_heads": expanded,
        "confirmation": confirmation,
        "preregistration_hash": preregistration["preregistration_hash"],
        "candidate_selection_hash": selection["selection_hash"],
        "choices_file_hash": canonical_hash(dataframe_records(saved_choices)),
        "ranking_file_hash": canonical_hash(dataframe_records(saved_rankings)),
        "coverage_manifest_hash": canonical_hash(dataframe_records(saved_coverage)),
        "claim_scope": preregistration["claim_scope"],
        "claims_removed": preregistration["claims_removed"],
    }
    atomic_json(output / "adaptive_manifest.json", manifest)
    return manifest


def validate_primary_adaptive(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output = run_dir if (run_dir / "adaptive_manifest.json").is_file() else run_dir / OUTPUT_NAME
    manifest_path = output / "adaptive_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError("primary adaptive manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "confirmed"
        or manifest.get("matched_causal_ablation_allowed") is not True
        or manifest.get("primary_condition_only") is not True
    ):
        raise ArtifactError("primary adaptive rankings have not passed confirmation")
    required = {
        "benchmark.json",
        "preregistration.json",
        "candidate_selection.json",
        "coverage_manifest.csv",
        "per_seed_metrics_primary.csv",
        "primary_residual_metrics.csv",
        "pos_head_rankings_primary.csv",
        "pos_head_choices.csv",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise ArtifactError(f"primary adaptive bundle is missing: {', '.join(missing)}")
    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    if not benchmark.get("sequential_parallel_exact") or not benchmark.get("main_full_exact"):
        raise ArtifactError("primary benchmark equivalence did not pass")
    coverage = pd.read_csv(output / "coverage_manifest.csv")
    metrics = pd.read_csv(output / "per_seed_metrics_primary.csv")
    residual = pd.read_csv(output / "primary_residual_metrics.csv")
    rankings = pd.read_csv(output / "pos_head_rankings_primary.csv")
    choices = pd.read_csv(output / "pos_head_choices.csv")
    selection = json.loads((output / "candidate_selection.json").read_text(encoding="utf-8"))
    if (
        len(coverage) != 1152
        or coverage[["seed", "normalized_progress", "relative_label", "feature_kind"]].duplicated().any()
    ):
        raise ArtifactError("coverage is not the exact 1,152-fit universe")
    if set(choices["relative_label"].astype(str)) != {PRIMARY_DEPTH}:
        raise ArtifactError("primary choices must contain only the middle depth")
    if (
        set(residual["seed"].astype(int)) != {42, 43, 44}
        or set(residual["normalized_progress"].astype(float)) != {PRIMARY_PROGRESS}
        or set(residual["relative_label"].astype(str)) != {PRIMARY_DEPTH}
        or set(residual["feature_kind"].astype(str)) != {"residual"}
    ):
        raise ArtifactError("primary residual controls are incomplete")
    screen = metrics[
        (metrics.seed == SCREEN_SEED)
        & (metrics.normalized_progress == PRIMARY_PROGRESS)
        & (metrics.relative_label == PRIMARY_DEPTH)
    ]
    if set(screen.feature_kind.astype(str)) != set(HEADS):
        raise ArtifactError("primary screen does not cover all 32 heads")
    screen_coverage = coverage[
        (coverage.seed.astype(int) == SCREEN_SEED)
        & (coverage.normalized_progress.astype(float) == PRIMARY_PROGRESS)
        & (coverage.relative_label.astype(str) == PRIMARY_DEPTH)
    ]
    if set(screen_coverage.feature_kind.astype(str)) != set(HEADS) or set(
        screen_coverage.evidence_tier.astype(str)
    ) != {"canonical_full_controls"}:
        raise ArtifactError("seed-42 screen is missing canonical control coverage")
    candidates = set(map(str, selection.get("candidates", [])))
    candidate_counts = (
        metrics[metrics.feature_kind.astype(str).isin(candidates)].groupby("feature_kind")["seed"].nunique()
    )
    if (
        not candidates
        or set(candidate_counts.index.astype(str)) != candidates
        or not candidate_counts.eq(3).all()
    ):
        raise ArtifactError("frozen candidates lack three-seed confirmation coverage")
    selected = {
        str(choices.iloc[0].high_feature_kind),
        str(choices.iloc[0].low_feature_kind),
    }
    selected_controls = coverage[
        (coverage.normalized_progress.astype(float) == PRIMARY_PROGRESS)
        & (coverage.relative_label.astype(str) == PRIMARY_DEPTH)
        & coverage.feature_kind.astype(str).isin(selected)
        & coverage.seed.astype(int).isin({42, 43, 44})
    ]
    if len(selected_controls) != 6 or set(selected_controls.evidence_tier.astype(str)) != {
        "canonical_full_controls"
    }:
        raise ArtifactError("confirmed high/low pair lacks full three-seed controls")
    rebuilt = _aggregate_rankings(metrics)
    rebuilt["primary_condition_only"] = True
    try:
        pd.testing.assert_frame_equal(
            rankings.sort_values(["relative_label", "feature_kind"]).reset_index(drop=True),
            rebuilt.sort_values(["relative_label", "feature_kind"]).reset_index(drop=True),
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as error:
        raise ArtifactError("primary rankings are not deterministic") from error
    hashes = {
        "coverage_manifest_hash": canonical_hash(dataframe_records(coverage)),
        "ranking_file_hash": canonical_hash(dataframe_records(rankings)),
        "choices_file_hash": canonical_hash(dataframe_records(choices)),
    }
    if any(manifest.get(name) != value for name, value in hashes.items()):
        raise ArtifactError("primary bundle hashes do not match the manifest")
    return {
        "valid": True,
        "status": "confirmed",
        "primary_condition_only": True,
        "confirmed_depths": [PRIMARY_DEPTH],
        "matched_causal_ablation_allowed": True,
        "rankings_deterministic": True,
        "sequential_parallel_exact": True,
        "main_full_exact": True,
    }

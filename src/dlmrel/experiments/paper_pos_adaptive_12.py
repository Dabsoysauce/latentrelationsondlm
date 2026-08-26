"""Twelve-candidate POS protocol with three-depth and three-progress coverage.

The protocol is deliberately separate from both the canonical grid and the
uncertainty-expanding adaptive protocol. Ranking-only probes have their own
atomic checkpoint namespace. Canonical full probes, including shuffled-label
and random-feature controls, are run only for the final causal heads and remain
reusable by a later full-grid completion.
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
    PRIMARY_DEPTH,
    PRIMARY_PROGRESS,
    SCREEN_SEED,
    FitKey,
    _confirmation,
    _fit_keys,
    _metric_frame,
    _ResourceMonitor,
    _saved_context,
    inventory,
)

SCHEMA = "dlmrel-pos-adaptive-12-v1"
OUTPUT_NAME = "pos_adaptive_12"
PROGRESS_VALUES = (0.25, 0.5, 0.75)
DEPTHS = ("early", "middle", "late")
CANDIDATES_PER_DEPTH = 12


def _head_inventory(run_dir: Path) -> tuple[str, ...]:
    """Read the model-specific head inventory from an extracted feature shard."""
    shards = sorted(
        (run_dir / "checkpoints").glob("paper-pos-selection-features*.parquet")
    )
    if not shards:
        raise ArtifactError("no extracted POS selection features are available")
    feature_kinds = pd.read_parquet(shards[0], columns=["feature_kind"])[
        "feature_kind"
    ].astype(str)
    indices = sorted(
        {
            int(kind.removeprefix("head_"))
            for kind in feature_kinds.unique()
            if kind.startswith("head_") and kind.removeprefix("head_").isdigit()
        }
    )
    if len(indices) < CANDIDATES_PER_DEPTH or indices != list(range(len(indices))):
        raise ArtifactError("extracted POS features have an invalid attention-head inventory")
    return tuple(f"head_{index}" for index in indices)


def _full_head_grid(cfg, heads: tuple[str, ...]) -> list[FitKey]:
    return [
        FitKey(seed, float(progress), depth, head)
        for seed in cfg.experiment.seeds
        for progress in cfg.experiment.normalized_progress
        for depth in DEPTHS
        for head in heads
    ]


def candidate_set_12(screen: pd.DataFrame) -> tuple[set[str], dict[str, list[str]]]:
    """Select a symmetric, frozen 12-head set from the seed-42 screen.

    The set contains top four, bottom four, the next two below the top boundary,
    and the next two above the bottom boundary. No held-out seed is inspected.
    """
    ordered = screen.sort_values(["accuracy", "feature_kind"], ascending=[False, True])
    heads = tuple(sorted(set(ordered.feature_kind), key=lambda value: int(value.split("_")[1])))
    expected = tuple(f"head_{index}" for index in range(len(heads)))
    if len(heads) < CANDIDATES_PER_DEPTH or heads != expected or len(ordered) != len(heads):
        raise ArtifactError("12-candidate selection requires one result for every model head")
    top = list(ordered.iloc[:4].feature_kind)
    high_boundary = list(ordered.iloc[4:6].feature_kind)
    low_boundary = list(ordered.iloc[-6:-4].feature_kind)
    bottom = list(ordered.iloc[-4:].feature_kind)
    chosen = set(top + high_boundary + low_boundary + bottom)
    if len(chosen) != CANDIDATES_PER_DEPTH:
        raise ArtifactError("symmetric 12-candidate rule did not produce exactly 12 heads")
    reasons = {}
    for head in chosen:
        tags = []
        if head in top:
            tags.append("top_core_rank_1_4")
        if head in high_boundary:
            tags.append("high_boundary_rank_5_6")
        if head in low_boundary:
            tags.append(f"low_boundary_rank_{len(heads) - 5}_{len(heads) - 4}")
        if head in bottom:
            tags.append(f"bottom_core_rank_{len(heads) - 3}_{len(heads)}")
        reasons[head] = tags
    return chosen, reasons


class _MainCheckpointStore:
    def __init__(self, run_dir: Path):
        cfg, manifests, _canonical, _depths = _saved_context(run_dir)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        self.directory = run_dir / OUTPUT_NAME / "main_fit_checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.expected_common = {
            "schema_version": SCHEMA,
            "scientific_config_hash": metadata.get("scientific_config_hash"),
            "manifest_hashes": manifests,
            "fixed_regularization_c": float(
                cfg.experiment.settings["fixed_regularization_c"]
            ),
            "label_inventory": list(paper_pos.LABELS),
            "ranking_classifier_only": True,
        }

    def _paths(self, key: FitKey):
        slug = (
            f"seed-{key.seed}__p-{key.progress:.6f}__"
            f"{key.relative_label}__{key.feature_kind}"
        )
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
        except (OSError, ValueError):
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
    thread_limit = threadpool_limits(limits=1) if workers > 1 else None
    context = thread_limit if thread_limit is not None else nullcontext()
    with context:
        if workers <= 1:
            outputs = [
                paper_pos._fit_evaluate_main_only(train, test, **kwargs)
                for train, test, kwargs in tasks
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        paper_pos._fit_evaluate_main_only, train, test, **kwargs
                    )
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
            and existing.get("worker_counts") == list(worker_counts)
        ):
            return existing
    cfg, _manifests, _store, _depths = _saved_context(run_dir)
    load_started = time.perf_counter()
    selection, test = paper_pos._load_condition_frames(
        run_dir, seed=SCREEN_SEED, progress=PRIMARY_PROGRESS, cache_root=cache_root
    )
    load_seconds = time.perf_counter() - load_started
    identities = [(PRIMARY_DEPTH, f"head_{head}") for head in range(12)]
    selection_groups = selection.groupby(
        ["relative_label", "feature_kind"], observed=True, sort=True
    )
    test_groups = test.groupby(["relative_label", "feature_kind"], observed=True, sort=True)
    tasks = [
        (
            selection_groups.get_group(identity),
            test_groups.get_group(identity),
            {
                "seed": SCREEN_SEED,
                "regularization": float(
                    cfg.experiment.settings["fixed_regularization_c"]
                ),
            },
        )
        for identity in identities
    ]
    sequential, sequential_seconds = _run_main_batch(tasks[:1], 1)
    trials = []
    trial_outputs = {}
    for workers in worker_counts:
        with _ResourceMonitor() as resources:
            outputs, seconds = _run_main_batch(tasks, workers)
        trial_outputs[workers] = outputs
        trials.append(
            {
                "workers": workers,
                "fits": len(tasks),
                "wall_seconds": seconds,
                "seconds_per_main_fit_throughput": seconds / len(tasks),
                "main_fits_per_hour": 3600.0 * len(tasks) / seconds,
                **resources.summary(),
            }
        )
    chosen = min(trials, key=lambda row: (row["wall_seconds"], row["workers"]))
    parallel_first = trial_outputs[int(chosen["workers"])][0]
    exact = sequential[0][1] == parallel_first[1] and sequential[0][0].equals(
        parallel_first[0]
    )
    if not exact:
        raise ArtifactError("main-only sequential and parallel benchmark results differ")
    payload = {
        "schema_version": SCHEMA,
        "worker_counts": list(worker_counts),
        "logical_cpus": os.cpu_count(),
        "feature_load_seconds": load_seconds,
        "sequential_representative_main_fit_seconds": sequential_seconds,
        "sequential_parallel_exact": True,
        "trials": trials,
        "selected_workers": int(chosen["workers"]),
        "selected_seconds_per_main_fit": float(
            chosen["seconds_per_main_fit_throughput"]
        ),
        "selected_main_fits_per_hour": float(chosen["main_fits_per_hour"]),
        "estimated_full_logical_seconds": float(
            chosen["seconds_per_main_fit_throughput"] * 3.5
        ),
        "benchmark_main_fits": 1 + len(tasks) * len(worker_counts),
    }
    atomic_json(path, payload)
    return payload


def _main_inventory(store: _MainCheckpointStore, keys: set[FitKey]):
    completed = {}
    for key in sorted(keys, key=lambda value: value.tuple()):
        cached = store.load(key)
        if cached is not None:
            completed[key] = cached
    return completed


def _main_metric_frame(completed):
    rows = []
    for key, (_evidence, metrics) in sorted(completed.items(), key=lambda item: item[0].tuple()):
        rows.append(
            {
                "seed": key.seed,
                "normalized_progress": key.progress,
                "relative_label": key.relative_label,
                "feature_kind": key.feature_kind,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


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
    store = _MainCheckpointStore(run_dir)
    completed = _main_inventory(store, keys)
    missing = [key for key in sorted(keys, key=lambda value: value.tuple()) if key not in completed]
    for condition in sorted({(key.seed, key.progress) for key in missing}):
        current = [key for key in missing if (key.seed, key.progress) == condition]
        expected = seconds_per_fit * len(current) + 30.0
        if time.monotonic() + expected > deadline:
            raise TimeoutError(
                f"budget guard stopped before {phase}: {len(current)} main fits need ~{expected:.0f}s"
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
            for identity, frame in test.groupby(
                ["relative_label", "feature_kind"], observed=True, sort=True
            )
            if identity in identities
        }
        if set(selection_groups) != identities or set(test_groups) != identities:
            raise ArtifactError("a requested main-only feature group is missing")
        tasks = [
            (
                selection_groups[(key.relative_label, key.feature_kind)],
                test_groups[(key.relative_label, key.feature_kind)],
                {
                    "seed": key.seed,
                    "regularization": float(
                        cfg.experiment.settings["fixed_regularization_c"]
                    ),
                },
            )
            for key in current
        ]
        outputs, seconds = _run_main_batch(tasks, workers)
        for index, (key, result) in enumerate(zip(current, outputs, strict=True), start=1):
            store.store(key, *result)
            completed[key] = result
            print(
                f"[{phase}] {index}/{len(current)} seed={seed} p={progress:.2f} "
                f"batch={seconds:.1f}s",
                flush=True,
            )
    return _main_inventory(store, keys)


def _coverage(cfg, heads, canonical, main, phase_reason):
    rows = []
    for key in _full_head_grid(cfg, heads):
        canonical_full = key in canonical
        main_only = key in main and not canonical_full
        phase, reason = phase_reason.get(
            key,
            ("reused_existing", "pre-existing canonical fit retained as auxiliary evidence"),
        )
        rows.append(
            {
                "seed": key.seed,
                "normalized_progress": key.progress,
                "relative_label": key.relative_label,
                "feature_kind": key.feature_kind,
                "included": canonical_full or main_only,
                "evidence_tier": (
                    "canonical_full_controls"
                    if canonical_full
                    else "ranking_main_only" if main_only else "omitted"
                ),
                "phase": phase if canonical_full or main_only else "omitted",
                "reason": reason if canonical_full or main_only else "not in fixed protocol",
            }
        )
    return pd.DataFrame(rows)


def run_adaptive_12(
    run_dir: str | Path,
    *,
    cache_root=None,
    budget_seconds: int = 8 * 60 * 60,
    validation_reserve_seconds: int = 15 * 60,
    worker_counts: tuple[int, ...] = (6, 8, 10, 12),
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output = run_dir / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    cfg, _manifests, _canonical_store, depth_mapping = _saved_context(run_dir)
    heads = _head_inventory(run_dir)
    if not set(PROGRESS_VALUES).issubset(set(map(float, cfg.experiment.normalized_progress))):
        raise ArtifactError("saved extraction lacks one of progress .25/.50/.75")
    started = time.monotonic()
    deadline = started + budget_seconds - validation_reserve_seconds
    preregistration = {
        "schema_version": SCHEMA,
        "screen_seed": SCREEN_SEED,
        "held_out_confirmation_seeds": [43, 44],
        "screen_progress": PRIMARY_PROGRESS,
        "trajectory_progress": list(PROGRESS_VALUES),
        "depths": list(DEPTHS),
        "candidates_per_depth": CANDIDATES_PER_DEPTH,
        "head_inventory": list(heads),
        "candidate_rule": (
            "top 4 + ranks 5-6 + the two ranks immediately above the bottom 4 + "
            "bottom 4 from the seed-42 p=.5 screen"
        ),
        "controls_scope": (
            "canonical shuffled-label and random-feature controls for final high/low "
            "heads at p=.5 across all seeds/depths; ranking-only probes otherwise"
        ),
        "claim_scope": (
            "three-depth primary ranking plus descriptive .25/.50/.75 trajectories "
            "for the confirmed high/low heads"
        ),
    }
    preregistration["preregistration_hash"] = canonical_hash(preregistration)
    prereg_path = output / "preregistration.json"
    if prereg_path.is_file():
        if json.loads(prereg_path.read_text(encoding="utf-8")) != preregistration:
            raise ArtifactError("saved 12-candidate preregistration differs")
    else:
        atomic_json(prereg_path, preregistration)

    benchmark = _benchmark(run_dir, cache_root, worker_counts)
    workers = int(benchmark["selected_workers"])
    main_seconds = float(benchmark["selected_seconds_per_main_fit"])
    full_seconds = float(benchmark["estimated_full_logical_seconds"])
    phase_reason = {}

    screen_keys = {
        FitKey(SCREEN_SEED, PRIMARY_PROGRESS, depth, head)
        for depth in DEPTHS
        for head in heads
    }
    for key in screen_keys:
        phase_reason[key] = ("A_all_head_screen", "fixed seed-42 all-head p=.5 screen")
    screen = _fit_main_keys(
        run_dir,
        screen_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=deadline,
        seconds_per_fit=main_seconds,
        phase="screen",
    )
    screen_metrics = _main_metric_frame(screen)
    candidates_by_depth = {}
    reasons_by_depth = {}
    for depth in DEPTHS:
        candidates, reasons = candidate_set_12(
            screen_metrics[screen_metrics.relative_label == depth]
        )
        candidates_by_depth[depth] = sorted(candidates)
        reasons_by_depth[depth] = reasons
    selection_payload = {
        "schema_version": SCHEMA,
        "preregistration_hash": preregistration["preregistration_hash"],
        "candidates_by_depth": candidates_by_depth,
        "reasons_by_depth": reasons_by_depth,
    }
    selection_payload["selection_hash"] = canonical_hash(selection_payload)
    atomic_json(output / "candidate_selection.json", selection_payload)

    confirmation_keys = {
        FitKey(seed, PRIMARY_PROGRESS, depth, head)
        for depth, heads in candidates_by_depth.items()
        for head in heads
        for seed in (43, 44)
    }
    for key in confirmation_keys:
        phase_reason[key] = (
            "B_held_out_confirmation",
            "fixed 12-head set chosen before held-out seeds were read",
        )
    primary_keys = screen_keys | confirmation_keys
    primary = _fit_main_keys(
        run_dir,
        primary_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=deadline,
        seconds_per_fit=main_seconds,
        phase="confirmation",
    )
    primary_metrics = _main_metric_frame(primary)
    confirmations = {
        depth: _confirmation(depth, primary_metrics, set(candidates_by_depth[depth]))
        for depth in DEPTHS
    }
    if not all(result["confirmed"] for result in confirmations.values()):
        manifest = {
            "schema_version": SCHEMA,
            "status": "blocked_unstable_rankings",
            "matched_causal_ablation_allowed": False,
            "confirmations": confirmations,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(output / "adaptive_manifest.json", manifest)
        raise ArtifactError("12-candidate held-out ranking gate failed; no trajectories published")

    choices_rows = []
    layers = depth_mapping.set_index("relative_label")["actual_layer_index"].astype(int)
    for depth in DEPTHS:
        result = confirmations[depth]
        choices_rows.append(
            {
                "relative_label": depth,
                "actual_layer_index": int(layers.loc[depth]),
                **result,
            }
        )
    choices = pd.DataFrame(choices_rows)
    selected_by_depth = {
        row.relative_label: {row.high_feature_kind, row.low_feature_kind}
        for row in choices.itertuples(index=False)
    }
    trajectory_keys = {
        FitKey(seed, progress, depth, head)
        for depth, heads in selected_by_depth.items()
        for head in heads
        for seed in (42, 43, 44)
        for progress in PROGRESS_VALUES
    }
    for key in trajectory_keys:
        phase_reason[key] = (
            "C_selected_head_trajectory",
            "confirmed high/low head measured at preregistered progress .25/.50/.75",
        )
    all_main_keys = primary_keys | trajectory_keys
    main = _fit_main_keys(
        run_dir,
        all_main_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=deadline,
        seconds_per_fit=main_seconds,
        phase="trajectory",
    )
    main_metrics = _main_metric_frame(main)

    control_keys = {
        FitKey(seed, PRIMARY_PROGRESS, depth, head)
        for depth, heads in selected_by_depth.items()
        for head in heads
        for seed in (42, 43, 44)
    }
    _fit_keys(
        run_dir,
        control_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=deadline,
        predicted_seconds_per_fit=full_seconds,
        phase="selected-controls",
    )
    residual_keys = {
        FitKey(seed, PRIMARY_PROGRESS, PRIMARY_DEPTH, "residual")
        for seed in (42, 43, 44)
    }
    residual = _fit_keys(
        run_dir,
        residual_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=deadline,
        predicted_seconds_per_fit=full_seconds,
        phase="primary-residual",
    )

    _cfg, canonical = inventory(run_dir)
    canonical_metrics = _metric_frame(canonical)
    selected_primary_main = main_metrics[
        main_metrics.apply(
            lambda row: FitKey(
                int(row.seed),
                float(row.normalized_progress),
                str(row.relative_label),
                str(row.feature_kind),
            )
            in control_keys,
            axis=1,
        )
    ]
    selected_primary_full = canonical_metrics[
        canonical_metrics.apply(
            lambda row: FitKey(
                int(row.seed),
                float(row.normalized_progress),
                str(row.relative_label),
                str(row.feature_kind),
            )
            in control_keys,
            axis=1,
        )
    ]
    comparison_columns = [
        "seed",
        "normalized_progress",
        "relative_label",
        "feature_kind",
        "accuracy",
        "macro_f1",
        "majority_accuracy",
        "n_positions",
    ]
    pd.testing.assert_frame_equal(
        selected_primary_main[comparison_columns]
        .sort_values(comparison_columns[:4])
        .reset_index(drop=True),
        selected_primary_full[comparison_columns]
        .sort_values(comparison_columns[:4])
        .reset_index(drop=True),
        check_exact=True,
    )

    rankings = main_metrics[main_metrics.normalized_progress == PRIMARY_PROGRESS].copy()
    rankings = rankings.groupby(["relative_label", "feature_kind"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        n_seeds=("seed", "nunique"),
    )
    rankings["fixed_12_candidate_protocol"] = True
    trajectory = main_metrics[
        main_metrics.apply(
            lambda row: str(row.feature_kind)
            in selected_by_depth.get(str(row.relative_label), set()),
            axis=1,
        )
        & main_metrics.normalized_progress.isin(PROGRESS_VALUES)
    ].copy()
    coverage = _coverage(cfg, heads, canonical, main, phase_reason)
    residual_rows = []
    for key, (_evidence, metrics) in sorted(residual.items(), key=lambda item: item[0].tuple()):
        residual_rows.append(
            {
                "seed": key.seed,
                "normalized_progress": key.progress,
                "relative_label": key.relative_label,
                "feature_kind": key.feature_kind,
                **metrics,
            }
        )
    residual_frame = pd.DataFrame(residual_rows)
    choices.to_csv(output / "pos_head_choices.csv", index=False)
    rankings.to_csv(output / "pos_head_rankings_adaptive_12.csv", index=False)
    trajectory.to_csv(output / "selected_head_progress_trajectories.csv", index=False)
    main_metrics.to_csv(output / "main_probe_metrics.csv", index=False)
    selected_primary_full.to_csv(output / "selected_head_control_metrics.csv", index=False)
    residual_frame.to_csv(output / "primary_residual_metrics.csv", index=False)
    coverage.to_csv(output / "coverage_manifest.csv", index=False)
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": SCHEMA,
        "status": "confirmed",
        "adaptive_reduced_grid": True,
        "fixed_candidates_per_depth": CANDIDATES_PER_DEPTH,
        "head_inventory": list(heads),
        "head_count": len(heads),
        "matched_causal_ablation_allowed": True,
        "confirmed_depths": list(DEPTHS),
        "progress_values": list(PROGRESS_VALUES),
        "screen_seed": SCREEN_SEED,
        "held_out_confirmation_seeds": [43, 44],
        "workers": workers,
        "budget_seconds": budget_seconds,
        "validation_reserve_seconds": validation_reserve_seconds,
        "elapsed_seconds": elapsed,
        "main_probe_fits": len(main),
        "canonical_selected_control_fits": len(control_keys),
        "primary_residual_fits": len(residual),
        "confirmations": confirmations,
        "preregistration_hash": preregistration["preregistration_hash"],
        "candidate_selection_hash": selection_payload["selection_hash"],
        "choices_file_hash": canonical_hash(dataframe_records(choices)),
        "ranking_file_hash": canonical_hash(dataframe_records(rankings)),
        "trajectory_file_hash": canonical_hash(dataframe_records(trajectory)),
        "coverage_manifest_hash": canonical_hash(dataframe_records(coverage)),
        "claim_scope": preregistration["claim_scope"],
    }
    atomic_json(output / "adaptive_manifest.json", manifest)
    return manifest


def validate_adaptive_12(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output = run_dir if (run_dir / "adaptive_manifest.json").is_file() else run_dir / OUTPUT_NAME
    manifest = json.loads((output / "adaptive_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "confirmed"
        or manifest.get("matched_causal_ablation_allowed") is not True
    ):
        raise ArtifactError("12-candidate POS bundle is not confirmed")
    required = {
        "pos_head_choices.csv",
        "pos_head_rankings_adaptive_12.csv",
        "selected_head_progress_trajectories.csv",
        "main_probe_metrics.csv",
        "selected_head_control_metrics.csv",
        "primary_residual_metrics.csv",
        "coverage_manifest.csv",
        "benchmark.json",
        "preregistration.json",
        "candidate_selection.json",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise ArtifactError("12-candidate bundle missing: " + ", ".join(missing))
    choices = pd.read_csv(output / "pos_head_choices.csv")
    rankings = pd.read_csv(output / "pos_head_rankings_adaptive_12.csv")
    trajectory = pd.read_csv(output / "selected_head_progress_trajectories.csv")
    coverage = pd.read_csv(output / "coverage_manifest.csv")
    candidates = json.loads((output / "candidate_selection.json").read_text(encoding="utf-8"))
    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    if set(choices.relative_label.astype(str)) != set(DEPTHS):
        raise ArtifactError("confirmed choices do not cover early/middle/late")
    if any(len(heads) != CANDIDATES_PER_DEPTH for heads in candidates["candidates_by_depth"].values()):
        raise ArtifactError("candidate selection is not exactly 12 heads per depth")
    if set(trajectory.normalized_progress.astype(float)) != set(PROGRESS_VALUES):
        raise ArtifactError("selected-head trajectories do not cover .25/.50/.75")
    trajectory_counts = trajectory.groupby(
        ["relative_label", "feature_kind", "normalized_progress"]
    ).seed.nunique()
    if len(trajectory_counts) != 18 or not trajectory_counts.eq(3).all():
        raise ArtifactError("selected-head trajectory lacks three-seed coverage")
    heads = tuple(manifest.get("head_inventory", ()))
    expected_rows = 3 * 4 * len(DEPTHS) * len(heads)
    if (
        len(heads) < CANDIDATES_PER_DEPTH
        or manifest.get("head_count") != len(heads)
        or len(coverage) != expected_rows
        or coverage.iloc[:, :4].duplicated().any()
        or set(coverage.feature_kind.astype(str)) != set(heads)
    ):
        raise ArtifactError("coverage manifest is not the exact model-head universe")
    hashes = {
        "choices_file_hash": canonical_hash(dataframe_records(choices)),
        "ranking_file_hash": canonical_hash(dataframe_records(rankings)),
        "trajectory_file_hash": canonical_hash(dataframe_records(trajectory)),
        "coverage_manifest_hash": canonical_hash(dataframe_records(coverage)),
    }
    if any(manifest.get(name) != value for name, value in hashes.items()):
        raise ArtifactError("12-candidate artifact hash validation failed")
    if benchmark.get("sequential_parallel_exact") is not True:
        raise ArtifactError("real-data sequential/parallel main-probe equality failed")
    return {
        "valid": True,
        "status": "confirmed",
        "candidates_per_depth": CANDIDATES_PER_DEPTH,
        "depths": list(DEPTHS),
        "progress_values": list(PROGRESS_VALUES),
        "main_probe_fits": int(manifest["main_probe_fits"]),
        "selected_control_fits": int(manifest["canonical_selected_control_fits"]),
        "selected_workers": int(manifest["workers"]),
        "matched_causal_ablation_allowed": True,
    }

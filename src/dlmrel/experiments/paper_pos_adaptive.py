"""Budgeted, preregistered CPU-only completion of the paper POS head probes.

This module deliberately does not finalize the canonical POS run. It reuses and
adds ordinary atomic v1 fit checkpoints, while publishing reduced-grid outputs
under ``pos_adaptive/`` with an explicit coverage manifest. Matched causal
ablation accepts those rankings only after the confirmation gate passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..artifacts import ArtifactError, atomic_json, canonical_hash, dataframe_records
from ..config import RunConfig
from . import paper_pos

SCHEMA = "dlmrel-pos-adaptive-v1"
SCREEN_SEED = 42
PRIMARY_PROGRESS = 0.5
PRIMARY_DEPTH = "middle"
TOP_CORE = 4
BOTTOM_CORE = 4
MIN_CONFIRMATION_RESERVE_FRACTION = 0.15
HEADS = tuple(f"head_{head}" for head in range(32))


@dataclass(frozen=True)
class FitKey:
    seed: int
    progress: float
    relative_label: str
    feature_kind: str

    def tuple(self):
        return self.seed, self.progress, self.relative_label, self.feature_kind


def _saved_context(run_dir: Path):
    raw = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    cfg = RunConfig.from_dict(raw)
    if cfg.experiment.type != "pos_token_class_linear_probes":
        raise ArtifactError("adaptive POS fitting requires a saved POS token-class run")
    manifests = json.loads((run_dir / "manifest_refs.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    store = paper_pos._FitCheckpointStore(
        run_dir,
        scientific_config_hash=metadata.get("scientific_config_hash"),
        manifest_hashes=manifests,
        regularization=float(cfg.experiment.settings["fixed_regularization_c"]),
        label_inventory=list(paper_pos.LABELS),
    )
    depths = pd.read_csv(run_dir / "relative_depth_mapping.csv")
    expected_depths = {"early", "middle", "late"}
    if set(depths["relative_label"].astype(str)) != expected_depths:
        raise ArtifactError("adaptive POS fitting requires early/middle/late depth mapping")
    return cfg, manifests, store, depths


def full_head_grid(cfg: RunConfig) -> list[FitKey]:
    return [
        FitKey(seed, float(progress), depth, head)
        for seed in cfg.experiment.seeds
        for progress in cfg.experiment.normalized_progress
        for depth in ("early", "middle", "late")
        for head in HEADS
    ]


def inventory(run_dir: str | Path) -> tuple[RunConfig, dict[FitKey, tuple[pd.DataFrame, dict]]]:
    run_dir = Path(run_dir)
    cfg, _manifests, store, _depths = _saved_context(run_dir)
    completed = {}
    for key in full_head_grid(cfg):
        cached = store.load(*key.tuple())
        if cached is not None:
            completed[key] = cached
    return cfg, completed


def _metric_frame(completed: dict[FitKey, tuple[pd.DataFrame, dict]]) -> pd.DataFrame:
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


def candidate_set(screen: pd.DataFrame) -> tuple[set[str], dict[str, list[str]]]:
    """Apply the frozen favorable-result-resistant candidate rule.

    Core top/bottom quartets are always retained. The uncertainty band uses a
    conservative binomial 95% half-width at each boundary. Sentence dependence
    makes that width anti-conservative, so it is floored at one percentage point.
    This intentionally expands rather than shrinks the confirmation set.
    """
    ordered = screen.sort_values(["accuracy", "feature_kind"], ascending=[False, True])
    if set(ordered["feature_kind"]) != set(HEADS) or len(ordered) != len(HEADS):
        raise ArtifactError("candidate selection requires one valid screen result for all 32 heads")
    top = ordered.head(TOP_CORE)
    bottom = ordered.tail(BOTTOM_CORE)
    top_boundary = top.iloc[-1]
    bottom_boundary = bottom.iloc[0]

    def half_width(row) -> float:
        n = max(int(row.n_positions), 1)
        p = float(row.accuracy)
        return max(0.01, 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n))

    top_cut = float(top_boundary.accuracy) - half_width(top_boundary)
    bottom_cut = float(bottom_boundary.accuracy) + half_width(bottom_boundary)
    top_ambiguous = ordered[ordered["accuracy"] >= top_cut]
    bottom_ambiguous = ordered[ordered["accuracy"] <= bottom_cut]
    chosen = set(top_ambiguous.feature_kind) | set(bottom_ambiguous.feature_kind)
    reasons = {}
    for head in HEADS:
        tags = []
        if head in set(top.feature_kind):
            tags.append("top_core")
        if head in set(bottom.feature_kind):
            tags.append("bottom_core")
        if head in set(top_ambiguous.feature_kind) - set(top.feature_kind):
            tags.append("top_boundary_uncertainty")
        if head in set(bottom_ambiguous.feature_kind) - set(bottom.feature_kind):
            tags.append("bottom_boundary_uncertainty")
        if tags:
            reasons[head] = tags
    return chosen, reasons


class _ResourceMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = None

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            return self

        process = psutil.Process()

        def sample():
            psutil.cpu_percent(interval=None)
            while not self.stop.wait(0.25):
                self.samples.append(
                    {
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "rss_bytes": process.memory_info().rss,
                    }
                )

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {"mean_cpu_percent": None, "peak_rss_bytes": None}
        return {
            "mean_cpu_percent": float(np.mean([row["cpu_percent"] for row in self.samples])),
            "peak_rss_bytes": int(max(row["rss_bytes"] for row in self.samples)),
        }


def _run_probe_batch(tasks, workers: int):
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if workers == 1:
            outputs = [paper_pos._fit_evaluate_probe(train, test, **kwargs) for train, test, kwargs in tasks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(paper_pos._fit_evaluate_probe, train, test, **kwargs)
                    for train, test, kwargs in tasks
                ]
                outputs = [future.result() for future in futures]
    return outputs, time.perf_counter() - started


def benchmark(
    run_dir: str | Path,
    *,
    cache_root: str | Path | None,
    worker_counts: tuple[int, ...] = (6, 8, 10, 12),
) -> dict[str, Any]:
    """Benchmark full-data fits without creating or replacing a fit checkpoint."""
    run_dir = Path(run_dir)
    cfg, _manifests, _store, _depths = _saved_context(run_dir)
    load_started = time.perf_counter()
    selection, test = paper_pos._load_condition_frames(
        run_dir,
        seed=SCREEN_SEED,
        progress=PRIMARY_PROGRESS,
        cache_root=cache_root,
    )
    load_seconds = time.perf_counter() - load_started
    sample_heads = (0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 27, 31)
    select_groups = {
        identity: group
        for identity, group in selection.groupby(
            ["relative_label", "feature_kind"], observed=True, sort=True
        )
    }
    test_groups = {
        identity: group
        for identity, group in test.groupby(
            ["relative_label", "feature_kind"], observed=True, sort=True
        )
    }
    tasks = []
    for head in sample_heads:
        identity = (PRIMARY_DEPTH, f"head_{head}")
        tasks.append(
            (
                select_groups[identity],
                test_groups[identity],
                {"seed": SCREEN_SEED, "regularization": 1.0},
            )
        )

    def fingerprint(frame: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        identity_columns = [
            column
            for column in ("sentence_id", "word_index", "label", "feature_kind")
            if column in frame
        ]
        digest.update(
            pd.util.hash_pandas_object(frame[identity_columns], index=True).to_numpy().tobytes()
        )
        digest.update(np.stack(frame["feature"].map(np.asarray)).tobytes())
        return digest.hexdigest()

    sequential, sequential_seconds = _run_probe_batch(tasks[:1], 1)
    trials = []
    best_outputs = None
    for workers in worker_counts:
        with _ResourceMonitor() as resources:
            outputs, seconds = _run_probe_batch(tasks, workers)
        trials.append(
            {
                "workers": workers,
                "fits": len(tasks),
                "wall_seconds": seconds,
                "seconds_per_fit_throughput": seconds / len(tasks),
                "fits_per_hour": 3600.0 * len(tasks) / seconds,
                **resources.summary(),
            }
        )
        if best_outputs is None or seconds < min(row["wall_seconds"] for row in trials[:-1]):
            best_outputs = outputs

    # The first task is deliberately repeated to compare numerical identity.
    parallel_first, _ = _run_probe_batch(tasks[:1], max(worker_counts))
    left_evidence, left_metrics = sequential[0]
    right_evidence, right_metrics = parallel_first[0]
    exact = left_metrics == right_metrics and left_evidence.equals(right_evidence)
    if not exact:
        raise ArtifactError("sequential and one-thread-per-fit benchmark results differ")

    scratch_parent = Path(cache_root) if cache_root else Path(tempfile.gettempdir())
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dlmrel-pos-write-", dir=scratch_parent) as directory:
        local_path = Path(directory) / "checkpoint.parquet"
        write_started = time.perf_counter()
        left_evidence.to_parquet(local_path, index=False)
        local_write_seconds = time.perf_counter() - write_started
    drive_scratch = run_dir / "pos_adaptive" / ".benchmark-write.parquet.tmp"
    drive_scratch.parent.mkdir(parents=True, exist_ok=True)
    drive_scratch.unlink(missing_ok=True)
    write_started = time.perf_counter()
    left_evidence.to_parquet(drive_scratch, index=False)
    drive_write_seconds = time.perf_counter() - write_started
    drive_scratch.unlink(missing_ok=True)

    total_memory = None
    try:
        import psutil

        total_memory = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    safe = [
        row
        for row in trials
        if row["peak_rss_bytes"] is None
        or total_memory is None
        or row["peak_rss_bytes"] <= 0.8 * total_memory
    ]
    if not safe:
        safe = trials
    chosen = min(safe, key=lambda row: (row["wall_seconds"], row["workers"]))
    payload = {
        "schema_version": SCHEMA,
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "feature_load_seconds": load_seconds,
        "local_checkpoint_write_seconds": local_write_seconds,
        "drive_checkpoint_write_seconds": drive_write_seconds,
        "sequential_representative_fit_seconds": sequential_seconds,
        "sequential_parallel_exact": exact,
        "selection_rows": len(tasks[0][0]),
        "test_rows": len(tasks[0][1]),
        "selection_row_label_feature_hash": fingerprint(tasks[0][0]),
        "test_row_label_feature_hash": fingerprint(tasks[0][1]),
        "solver": "sklearn.linear_model.LogisticRegression default lbfgs",
        "fixed_regularization_c": 1.0,
        "max_iterations": 2000,
        "random_seed": SCREEN_SEED,
        "trials": trials,
        "selected_workers": chosen["workers"],
        "selected_fits_per_hour": chosen["fits_per_hour"],
        "selected_seconds_per_fit_throughput": chosen["seconds_per_fit_throughput"],
    }
    atomic_json(run_dir / "pos_adaptive" / "benchmark.json", payload)
    return payload


def _fit_keys(
    run_dir: Path,
    keys: set[FitKey],
    *,
    workers: int,
    cache_root: str | Path | None,
    deadline: float,
    predicted_seconds_per_fit: float,
    phase: str,
) -> dict[FitKey, tuple[pd.DataFrame, dict]]:
    cfg, _manifests, store, _depths = _saved_context(run_dir)
    regularization = float(cfg.experiment.settings["fixed_regularization_c"])
    resolved = {}
    missing = []
    for key in sorted(keys, key=lambda value: value.tuple()):
        cached = store.load(*key.tuple())
        if cached is None:
            missing.append(key)
        else:
            resolved[key] = cached
    for condition in sorted({(key.seed, key.progress) for key in missing}):
        current = [key for key in missing if (key.seed, key.progress) == condition]
        expected = predicted_seconds_per_fit * len(current) + 30.0
        if time.monotonic() + expected > deadline:
            raise TimeoutError(
                f"budget guard stopped before {phase}: {len(current)} fits need about {expected:.0f}s"
            )
        seed, progress = condition
        selection, test = paper_pos._load_condition_frames(
            run_dir, seed=seed, progress=progress, cache_root=cache_root
        )
        requested_identities = {(key.relative_label, key.feature_kind) for key in current}
        select_groups = {
            identity: group
            for identity, group in selection.groupby(
                ["relative_label", "feature_kind"], observed=True, sort=True
            )
            if identity in requested_identities
        }
        test_groups = {
            identity: group
            for identity, group in test.groupby(
                ["relative_label", "feature_kind"], observed=True, sort=True
            )
            if identity in requested_identities
        }
        if set(select_groups) != requested_identities or set(test_groups) != requested_identities:
            raise ArtifactError("requested adaptive feature group is missing from extraction")
        tasks = [
            (
                select_groups[(key.relative_label, key.feature_kind)],
                test_groups[(key.relative_label, key.feature_kind)],
                {"seed": key.seed, "regularization": regularization},
            )
            for key in current
        ]
        outputs, seconds = _run_probe_batch(tasks, workers)
        for index, (key, output) in enumerate(zip(current, outputs, strict=True), start=1):
            store.store(*key.tuple(), *output)
            resolved[key] = output
            completed = len(resolved)
            eta = max(0.0, predicted_seconds_per_fit * (len(keys) - completed))
            print(
                f"[{phase}] checkpointed {index}/{len(current)} for seed={seed} "
                f"progress={progress:.2f}; batch={seconds:.1f}s ETA~{eta/60:.1f}m",
                flush=True,
            )
        del selection, test, select_groups, test_groups, tasks, outputs
    return resolved


def _confirmation(depth: str, metrics: pd.DataFrame, candidates: set[str]) -> dict[str, Any]:
    current = metrics[
        (metrics.normalized_progress == PRIMARY_PROGRESS)
        & (metrics.relative_label == depth)
        & metrics.feature_kind.isin(candidates)
    ].copy()
    counts = current.groupby("feature_kind")["seed"].nunique()
    if set(counts.index) != candidates or not counts.eq(3).all():
        return {"confirmed": False, "reason": "candidate seed coverage incomplete"}
    pivot = current.pivot(index="feature_kind", columns="seed", values="accuracy")
    means = pivot.mean(axis=1).sort_values(ascending=False)
    high, low = means.index[0], means.index[-1]
    high_ranks = pivot.rank(axis=0, ascending=False, method="min").loc[high]
    low_ranks = pivot.rank(axis=0, ascending=True, method="min").loc[low]
    correlations = pivot.corr(method="spearman").to_numpy()
    correlations = correlations[np.triu_indices_from(correlations, k=1)]
    median_correlation = float(np.median(correlations))
    confirmed = bool(
        high_ranks.max() <= 3
        and low_ranks.max() <= 3
        and median_correlation >= 0.5
        and means.iloc[0] > means.iloc[1]
        and means.iloc[-1] < means.iloc[-2]
    )
    return {
        "confirmed": confirmed,
        "high_feature_kind": high,
        "low_feature_kind": low,
        "high_mean_accuracy": float(means.iloc[0]),
        "low_mean_accuracy": float(means.iloc[-1]),
        "high_worst_seed_rank": float(high_ranks.max()),
        "low_worst_seed_rank": float(low_ranks.max()),
        "median_pairwise_seed_spearman": median_correlation,
        "reason": None if confirmed else "rank or cross-seed stability criterion failed",
    }


def _aggregate_rankings(metrics: pd.DataFrame) -> pd.DataFrame:
    primary_metrics = metrics[metrics.normalized_progress == PRIMARY_PROGRESS].copy()
    rankings = primary_metrics.groupby(
        ["relative_label", "feature_kind"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        n_seeds=("seed", "nunique"),
    )
    rankings["adaptive_reduced_grid"] = True
    rankings["primary_progress_only"] = True
    return rankings


def _write_coverage(
    output: Path,
    cfg: RunConfig,
    completed: dict[FitKey, tuple[pd.DataFrame, dict]],
    phase_reason: dict[FitKey, tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for key in full_head_grid(cfg):
        included = key in completed
        phase, reason = phase_reason.get(
            key,
            (
                "reused_existing",
                "valid pre-existing atomic fit retained as auxiliary coverage",
            ),
        )
        rows.append(
            {
                **dict(
                    zip(
                        ("seed", "normalized_progress", "relative_label", "feature_kind"),
                        key.tuple(),
                        strict=True,
                    )
                ),
                "included": included,
                "phase": phase if included else "omitted",
                "reason": reason if included else "not required by preregistered adaptive allocation",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "coverage_manifest.csv", index=False)
    return frame


def _decision_table(
    *,
    completed_before: int,
    candidate_proxy: int,
    fits_per_hour: float,
    condition_load_seconds: float,
) -> pd.DataFrame:
    """Quantitative comparison using the measured current-runtime throughput."""
    candidate_proxy = min(max(candidate_proxy, TOP_CORE + BOTTOM_CORE), 32)
    recommended_new = 96 + 2 * candidate_proxy * 3
    rows = [
        {
            "option": "Full original grid",
            "fits": 1152,
            "reused": completed_before,
            "new_fits": 1152 - completed_before,
            "conditions_loaded": 12,
            "seed_coverage": "3/3 for every head/condition",
            "head_coverage": "32/32 everywhere",
            "progress_depth_coverage": "4/4 progress x 3/3 depths",
            "claims_retained": "all canonical POS grid and causal-ablation inputs",
            "claims_narrowed_or_removed": "none",
            "causal_head_risk": "lowest; canonical definition",
        },
        {
            "option": "Primary-condition-only full-head grid",
            "fits": 96,
            "reused": 0,
            "new_fits": 96,
            "conditions_loaded": 3,
            "seed_coverage": "3/3",
            "head_coverage": "32/32",
            "progress_depth_coverage": "progress .5, middle only",
            "claims_retained": "primary midpoint POS comparison",
            "claims_narrowed_or_removed": "remove multi-depth/progress claim and causal early/late pairs",
            "causal_head_risk": "high outside middle depth",
        },
        {
            "option": "Full-head screen plus top/bottom/ambiguous confirmation",
            "fits": 32 + 2 * candidate_proxy,
            "reused": 0,
            "new_fits": 32 + 2 * candidate_proxy,
            "conditions_loaded": 3,
            "seed_coverage": "seed 42 all heads; seeds 43/44 candidates",
            "head_coverage": "32/32 screen; uncertainty candidates confirmed",
            "progress_depth_coverage": "progress .5, middle only",
            "claims_retained": "primary high-vs-low midpoint ranking",
            "claims_narrowed_or_removed": "remove early/late causal pairs and grid trend",
            "causal_head_risk": "low at middle; unsupported elsewhere",
        },
        {
            "option": "Stratified 50% grid",
            "fits": 576,
            "reused": completed_before,
            "new_fits": 576 - completed_before,
            "conditions_loaded": 12,
            "seed_coverage": "balanced but incomplete per cell",
            "head_coverage": "predeclared 16/32 per cell",
            "progress_depth_coverage": "4/4 x 3/3",
            "claims_retained": "coarse progress/depth trends",
            "claims_narrowed_or_removed": "cannot claim exhaustive head rankings",
            "causal_head_risk": "moderate; best omitted head can change selection",
        },
        {
            "option": "Stratified 25% grid",
            "fits": 288,
            "reused": completed_before,
            "new_fits": 288 - completed_before,
            "conditions_loaded": 12,
            "seed_coverage": "balanced but sparse",
            "head_coverage": "predeclared 8/32 per cell",
            "progress_depth_coverage": "4/4 x 3/3",
            "claims_retained": "descriptive sampled trends only",
            "claims_narrowed_or_removed": "remove exhaustive rankings and strong grid comparisons",
            "causal_head_risk": "high",
        },
        {
            "option": "Recommended adaptive design",
            "fits": completed_before + recommended_new,
            "reused": completed_before,
            "new_fits": recommended_new,
            "conditions_loaded": 3,
            "seed_coverage": "seed 42 all heads at 3 depths; seeds 43/44 candidates",
            "head_coverage": "32/32 screen at each depth; uncertainty candidates confirmed",
            "progress_depth_coverage": "primary progress .5 x 3/3 depths; existing p0 early auxiliary",
            "claims_retained": "primary condition and confirmed high-vs-low causal inputs at all depths",
            "claims_narrowed_or_removed": "narrow four-progress full-head grid to primary-progress inference",
            "causal_head_risk": "fail-closed; expands to full held-out heads when unstable",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["percent_original"] = 100.0 * frame["fits"] / 1152.0
    fit_seconds = 3600.0 * frame["new_fits"].clip(lower=0) / fits_per_hour
    estimates = fit_seconds + frame["conditions_loaded"] * condition_load_seconds
    frame["estimated_wall_minutes"] = estimates / 60.0
    frame["estimated_wall_low_minutes"] = estimates * 0.8 / 60.0
    frame["estimated_wall_high_minutes"] = estimates * 1.2 / 60.0
    return frame


def run_adaptive(
    run_dir: str | Path,
    *,
    cache_root: str | Path | None,
    budget_seconds: int = 10800,
    validation_reserve_seconds: int = 900,
    worker_counts: tuple[int, ...] = (6, 8, 10, 12),
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output = run_dir / "pos_adaptive"
    output.mkdir(parents=True, exist_ok=True)
    cfg, _manifests, _store, depths = _saved_context(run_dir)
    _inventory_cfg, completed_before_map = inventory(run_dir)
    completed_before = len(completed_before_map)
    if budget_seconds <= validation_reserve_seconds:
        raise ValueError("fit budget must exceed validation reserve")
    overall_start = time.monotonic()
    preregistration = {
        "schema_version": SCHEMA,
        "screen_seed": SCREEN_SEED,
        "held_out_confirmation_seeds": [43, 44],
        "primary_progress": PRIMARY_PROGRESS,
        "primary_depth": PRIMARY_DEPTH,
        "primary_condition_rationale": (
            "the repository's single-condition masked POS probe fixes normalized progress 0.5 "
            "and uses the midpoint hidden state"
        ),
        "screen_all_32_heads": True,
        "screen_depth_order": ["middle", "early", "late"],
        "candidate_rule": {
            "top_core": TOP_CORE,
            "bottom_core": BOTTOM_CORE,
            "uncertainty": (
                "retain every head within max(1 percentage point, binomial 95% "
                "half-width) of either core boundary"
            ),
        },
        "confirmation_rule": (
            "three seeds; selected high and low each rank in the corresponding three extreme "
            "positions for every seed; median pairwise Spearman >= 0.5; nonzero mean margins"
        ),
        "minimum_confirmation_reserve_fraction": MIN_CONFIRMATION_RESERVE_FRACTION,
        "canonical_full_grid_unchanged": True,
    }
    preregistration["preregistration_hash"] = canonical_hash(preregistration)
    prereg_path = output / "preregistration.json"
    if prereg_path.is_file():
        existing = json.loads(prereg_path.read_text(encoding="utf-8"))
        if existing != preregistration:
            raise ArtifactError("adaptive preregistration differs from the existing saved protocol")
    else:
        atomic_json(prereg_path, preregistration)

    benchmark_payload = benchmark(
        run_dir, cache_root=cache_root, worker_counts=worker_counts
    )
    workers = int(benchmark_payload["selected_workers"])
    seconds_per_fit = float(benchmark_payload["selected_seconds_per_fit_throughput"])
    preexisting_metrics = _metric_frame(completed_before_map)
    candidate_proxy = 20
    if not preexisting_metrics.empty:
        group_sizes = preexisting_metrics.groupby(
            ["seed", "normalized_progress", "relative_label"]
        ).size()
        complete_groups = group_sizes[group_sizes == 32]
        if len(complete_groups):
            seed, progress, depth = complete_groups.index[0]
            proxy_screen = preexisting_metrics[
                (preexisting_metrics.seed == seed)
                & (preexisting_metrics.normalized_progress == progress)
                & (preexisting_metrics.relative_label == depth)
            ]
            candidate_proxy = len(candidate_set(proxy_screen)[0])
    decision_table = _decision_table(
        completed_before=completed_before,
        candidate_proxy=candidate_proxy,
        fits_per_hour=float(benchmark_payload["selected_fits_per_hour"]),
        condition_load_seconds=float(benchmark_payload["feature_load_seconds"]),
    )
    decision_table.to_csv(output / "decision_table.csv", index=False)
    full_row = decision_table[decision_table.option == "Full original grid"].iloc[0]
    benchmark_payload["full_grid_estimated_wall_minutes"] = float(
        full_row.estimated_wall_minutes
    )
    benchmark_payload["full_grid_fits_within_165_minutes"] = bool(
        full_row.estimated_wall_high_minutes <= 165.0
    )
    atomic_json(output / "benchmark.json", benchmark_payload)
    fit_deadline = overall_start + budget_seconds - validation_reserve_seconds
    phase_reason = {}

    # Preserve the repository's explicit single-condition POS result: the
    # midpoint residual stream at progress .5, evaluated for all three seeds.
    # These three logical probes are additional to the 1,152 head-fit universe.
    primary_residual_keys = {
        FitKey(seed, PRIMARY_PROGRESS, PRIMARY_DEPTH, "residual")
        for seed in cfg.experiment.seeds
    }
    primary_residual = _fit_keys(
        run_dir,
        primary_residual_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=fit_deadline,
        predicted_seconds_per_fit=seconds_per_fit,
        phase="primary-residual",
    )
    primary_residual_rows = []
    for key, (_evidence, metrics_row) in sorted(
        primary_residual.items(), key=lambda item: item[0].tuple()
    ):
        primary_residual_rows.append(
            {
                "seed": key.seed,
                "normalized_progress": key.progress,
                "relative_label": key.relative_label,
                "feature_kind": key.feature_kind,
                **metrics_row,
            }
        )
    pd.DataFrame(primary_residual_rows).to_csv(
        output / "primary_residual_metrics.csv", index=False
    )

    screen_keys = {
        FitKey(SCREEN_SEED, PRIMARY_PROGRESS, depth, head)
        for depth in (PRIMARY_DEPTH, "early", "late")
        for head in HEADS
    }
    for key in screen_keys:
        phase = "A_primary_screen" if key.relative_label == PRIMARY_DEPTH else "D_depth_screen"
        phase_reason[key] = (phase, "unbiased all-head screen on fixed seed 42")
    _fit_keys(
        run_dir,
        screen_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=fit_deadline,
        predicted_seconds_per_fit=seconds_per_fit,
        phase="screen",
    )
    _cfg, completed = inventory(run_dir)
    metrics = _metric_frame(completed)
    candidates_by_depth = {}
    candidate_reasons = {}
    for depth in ("middle", "early", "late"):
        screen = metrics[
            (metrics.seed == SCREEN_SEED)
            & (metrics.normalized_progress == PRIMARY_PROGRESS)
            & (metrics.relative_label == depth)
        ]
        candidates, reasons = candidate_set(screen)
        candidates_by_depth[depth] = sorted(candidates)
        candidate_reasons[depth] = reasons
    selection_payload = {
        "schema_version": SCHEMA,
        "preregistration_hash": preregistration["preregistration_hash"],
        "candidates_by_depth": candidates_by_depth,
        "reasons_by_depth": candidate_reasons,
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
            "C_held_out_confirmation",
            "fixed candidate selected from seed-42 screen before seeds 43/44 were read",
        )
    reserve_fraction = len(confirmation_keys) / max(len(screen_keys | confirmation_keys), 1)
    if reserve_fraction < MIN_CONFIRMATION_RESERVE_FRACTION:
        raise ArtifactError("adaptive allocation violates the held-out confirmation reserve")
    _fit_keys(
        run_dir,
        confirmation_keys,
        workers=workers,
        cache_root=cache_root,
        deadline=fit_deadline,
        predicted_seconds_per_fit=seconds_per_fit,
        phase="confirmation",
    )

    _cfg, completed = inventory(run_dir)
    metrics = _metric_frame(completed)
    confirmations = {
        depth: _confirmation(depth, metrics, set(candidates_by_depth[depth]))
        for depth in ("early", "middle", "late")
    }
    unstable = [depth for depth, result in confirmations.items() if not result["confirmed"]]
    for depth in unstable:
        expansion = {
            FitKey(seed, PRIMARY_PROGRESS, depth, head)
            for seed in (43, 44)
            for head in HEADS
        }
        for key in expansion:
            phase_reason[key] = (
                "E_stability_expansion",
                "preregistered fail-safe expansion after candidate ranking instability",
            )
        try:
            _fit_keys(
                run_dir,
                expansion,
                workers=workers,
                cache_root=cache_root,
                deadline=fit_deadline,
                predicted_seconds_per_fit=seconds_per_fit,
                phase=f"stability-{depth}",
            )
        except TimeoutError:
            break

    _cfg, completed = inventory(run_dir)
    metrics = _metric_frame(completed)
    for depth in unstable:
        all_head_coverage = set(
            metrics[
                (metrics.normalized_progress == PRIMARY_PROGRESS)
                & (metrics.relative_label == depth)
                & (metrics.seed.isin([42, 43, 44]))
            ].feature_kind
        ) == set(HEADS)
        if all_head_coverage:
            candidates_by_depth[depth] = list(HEADS)
    confirmations = {
        depth: _confirmation(depth, metrics, set(candidates_by_depth[depth]))
        for depth in ("early", "middle", "late")
    }
    confirmed = all(result["confirmed"] for result in confirmations.values())

    selected_rows = []
    layer_by_depth = depths.set_index("relative_label")["actual_layer_index"].astype(int)
    for depth, result in confirmations.items():
        if result.get("high_feature_kind") and result.get("low_feature_kind"):
            selected_rows.append(
                {
                    "relative_label": depth,
                    "actual_layer_index": int(layer_by_depth.loc[depth]),
                    **result,
                }
            )
    choices = pd.DataFrame(selected_rows)
    rankings = _aggregate_rankings(metrics)
    if confirmed:
        choices.to_csv(output / "pos_head_choices.csv", index=False)
        rankings.to_csv(output / "pos_head_rankings_adaptive.csv", index=False)
    else:
        choices.to_csv(output / "pos_head_choices_provisional.csv", index=False)
        rankings.to_csv(output / "pos_head_rankings_provisional.csv", index=False)
    metrics.to_csv(output / "per_seed_metrics_adaptive.csv", index=False)
    coverage = _write_coverage(output, cfg, completed, phase_reason)
    elapsed = time.monotonic() - overall_start
    manifest = {
        "schema_version": SCHEMA,
        "status": "confirmed" if confirmed else "blocked_unstable_rankings",
        "adaptive_reduced_grid": True,
        "canonical_full_grid_complete": len(completed) == len(full_head_grid(cfg)),
        "preregistration_hash": preregistration["preregistration_hash"],
        "candidate_selection_hash": selection_payload["selection_hash"],
        "coverage_manifest_hash": canonical_hash(dataframe_records(coverage)),
        "ranking_file_hash": (
            canonical_hash(dataframe_records(rankings)) if confirmed else None
        ),
        "choices_file_hash": canonical_hash(dataframe_records(choices)) if confirmed else None,
        "screen_seed": SCREEN_SEED,
        "held_out_confirmation_seeds": [43, 44],
        "primary_progress": PRIMARY_PROGRESS,
        "primary_depth": PRIMARY_DEPTH,
        "workers": workers,
        "fit_budget_seconds": budget_seconds,
        "validation_reserve_seconds": validation_reserve_seconds,
        "adaptive_elapsed_seconds": elapsed,
        "completed_head_fits": len(completed),
        "primary_residual_fits": len(primary_residual),
        "primary_residual_all_three_seeds": len(primary_residual) == 3,
        "original_head_fits": len(full_head_grid(cfg)),
        "completed_ratio": len(completed) / len(full_head_grid(cfg)),
        "confirmations": confirmations,
        "matched_causal_ablation_allowed": confirmed,
        "causal_claim_scope": (
            "confirmed high-vs-low POS decoding at normalized progress 0.5; not the "
            "canonical four-progress average"
        ),
    }
    atomic_json(output / "adaptive_manifest.json", manifest)
    if not confirmed:
        missing_primary = 0
        for depth in ("early", "middle", "late"):
            for seed in (43, 44):
                for head in HEADS:
                    if FitKey(seed, PRIMARY_PROGRESS, depth, head) not in completed:
                        missing_primary += 1
        manifest["smallest_additional_runtime_seconds_estimate"] = (
            missing_primary * seconds_per_fit
        )
        atomic_json(output / "adaptive_manifest.json", manifest)
    return manifest


def validate_adaptive(run_dir: str | Path) -> dict[str, Any]:
    """Validate a confirmed reduced bundle without loading a model or feature table."""
    run_dir = Path(run_dir)
    output = run_dir if (run_dir / "adaptive_manifest.json").is_file() else run_dir / "pos_adaptive"
    manifest_path = output / "adaptive_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError("adaptive POS manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "confirmed" or not manifest.get(
        "matched_causal_ablation_allowed"
    ):
        raise ArtifactError("adaptive POS bundle has not passed held-out confirmation")
    required = {
        "coverage_manifest.csv",
        "per_seed_metrics_adaptive.csv",
        "primary_residual_metrics.csv",
        "pos_head_rankings_adaptive.csv",
        "pos_head_choices.csv",
        "benchmark.json",
        "preregistration.json",
        "candidate_selection.json",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise ArtifactError(f"adaptive POS bundle is missing: {', '.join(missing)}")
    coverage = pd.read_csv(output / "coverage_manifest.csv")
    metrics = pd.read_csv(output / "per_seed_metrics_adaptive.csv")
    rankings = pd.read_csv(output / "pos_head_rankings_adaptive.csv")
    choices = pd.read_csv(output / "pos_head_choices.csv")
    primary_residual = pd.read_csv(output / "primary_residual_metrics.csv")
    if len(coverage) != 1152 or coverage[
        ["seed", "normalized_progress", "relative_label", "feature_kind"]
    ].duplicated().any():
        raise ArtifactError("adaptive coverage manifest is not the exact 1,152-fit universe")
    if int(coverage["included"].astype(bool).sum()) != int(manifest["completed_head_fits"]):
        raise ArtifactError("adaptive coverage count disagrees with its manifest")
    if (
        set(primary_residual["seed"].astype(int)) != {42, 43, 44}
        or set(primary_residual["normalized_progress"].astype(float)) != {PRIMARY_PROGRESS}
        or set(primary_residual["relative_label"].astype(str)) != {PRIMARY_DEPTH}
        or set(primary_residual["feature_kind"].astype(str)) != {"residual"}
    ):
        raise ArtifactError("primary residual condition is not complete across all three seeds")
    hashes = {
        "coverage_manifest_hash": canonical_hash(dataframe_records(coverage)),
        "ranking_file_hash": canonical_hash(dataframe_records(rankings)),
        "choices_file_hash": canonical_hash(dataframe_records(choices)),
    }
    if any(manifest.get(key) != value for key, value in hashes.items()):
        raise ArtifactError("adaptive POS bundle hash validation failed")
    rebuilt = _aggregate_rankings(metrics)
    try:
        pd.testing.assert_frame_equal(
            rankings.sort_values(["relative_label", "feature_kind"]).reset_index(drop=True),
            rebuilt.sort_values(["relative_label", "feature_kind"]).reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ArtifactError("adaptive rankings are not deterministic from the saved fit set") from error
    if set(choices["relative_label"].astype(str)) != {"early", "middle", "late"}:
        raise ArtifactError("adaptive choices do not cover all three relative depths")
    benchmark_payload = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    if benchmark_payload.get("sequential_parallel_exact") is not True:
        raise ArtifactError("real-data sequential/parallel equivalence did not pass")
    return {
        "valid": True,
        "status": "confirmed",
        "completed_head_fits": int(manifest["completed_head_fits"]),
        "coverage_ratio": float(manifest["completed_ratio"]),
        "selected_workers": int(manifest["workers"]),
        "matched_causal_ablation_allowed": True,
        "rankings_deterministic": True,
        "sequential_parallel_exact": True,
    }

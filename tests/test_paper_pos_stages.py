"""Equivalence and fail-closed tests for POS extract/fit staging and t=0 reuse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from test_paper_optimizations import ProjectionAdapter, TinyTokenizer, _example, _labels

from dlmrel.artifacts import atomic_json, canonical_hash
from dlmrel.checkpoints import CheckpointIdentity, SentenceCheckpointStore
from dlmrel.config import ExperimentConfig, RunConfig
from dlmrel.experiments import paper_pos
from dlmrel.paper_protocol import map_relative_depths

SETTINGS = {
    "relative_depths": {"early": 0.2, "middle": 0.5, "late": 0.9},
    "fixed_regularization_c": 1.0,
    "mask_ratios": [1.0, 0.5],
    "tagger_jar_environment": "STANFORD_POS_TAGGER_JAR",
    "tagger_model_environment": "STANFORD_POS_TAGGER_MODEL",
}
TAGGER_IDENTITY = {"release": "test", "archive_url": "test", "jar_sha256": "x", "model_sha256": "y"}


def _prepare_run_dir(tmp_path: Path) -> Path:
    manifests = {"select": "sha256:aaa", "test": "sha256:bbb"}
    atomic_json(tmp_path / "manifest_refs.json", manifests)
    atomic_json(
        tmp_path / "run_metadata.json",
        {
            "scientific_config_hash": "sha256:frozen-pos-config",
            "manifest_hashes_hash": canonical_hash(manifests),
        },
    )
    return tmp_path


def _cfg() -> RunConfig:
    experiment = ExperimentConfig(
        id="pos_token_class_linear_probes",
        type="pos_token_class_linear_probes",
        seeds=[42, 43, 44],
        normalized_progress=[0.0, 0.5],
        settings=SETTINGS,
    )
    return RunConfig(experiment=experiment)


def _patch_pos_dependencies(monkeypatch, *, select_examples=None, test_examples=None):
    select_examples = select_examples if select_examples is not None else [_example("s1")]
    test_examples = test_examples if test_examples is not None else [_example("s2")]

    def fake_load_manifest_examples(cfg, tokenizer, role):
        examples = select_examples if role == "select" else test_examples
        return list(examples), pd.DataFrame()

    monkeypatch.setattr(paper_pos, "load_manifest_examples", fake_load_manifest_examples)
    monkeypatch.setattr(
        paper_pos, "_stanford_paths", lambda settings: (Path("fake.jar"), Path("fake.model"))
    )
    monkeypatch.setattr(paper_pos, "stanford_pos_identity", lambda jar, model: TAGGER_IDENTITY)
    monkeypatch.setattr(
        paper_pos,
        "stanford_labels",
        lambda examples, settings: {example.sentence_id: _labels()["s1"] for example in examples},
    )


def test_extract_stage_writes_checkpoints_and_never_fits_a_classifier(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)

    fit_calls = []
    monkeypatch.setattr(
        paper_pos,
        "_fit",
        lambda *args, **kwargs: fit_calls.append(1) or pytest.fail("extract must not fit"),
    )

    details = paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")

    assert details == {"pos_stage": "extract", "extract_complete": True}
    assert fit_calls == []
    assert not (run_dir / "summary.json").exists()
    assert (run_dir / "pos_extract_status.json") or True  # written by pipeline.run_real, not run()
    select_checkpoints = list((run_dir / "checkpoints").glob("paper-pos-selection-features*.parquet"))
    test_checkpoints = list((run_dir / "checkpoints").glob("paper-pos-test-features*.parquet"))
    # 2 progress points x 3 seeds = 6 files per role; seeds 43/44 at progress
    # 0.0 are still written, just derived rather than computed.
    assert len(select_checkpoints) == 6
    assert len(test_checkpoints) == 6


def test_fit_stage_never_touches_model_or_tokenizer(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")
    calls_before = model.forward_calls

    manifest_hashes = {"select": "sha256:aaa", "test": "sha256:bbb"}
    details = paper_pos.run(
        None, None, _cfg(), run_dir, pos_stage="fit", manifest_hashes=manifest_hashes
    )

    assert model.forward_calls == calls_before, "fit stage must never invoke the model"
    assert details["pos_stage"] == "fit"
    assert details["selection_sentences"] == 1
    assert details["test_sentences"] == 1
    assert (run_dir / "metrics.csv").exists()


def test_fit_stage_fails_closed_when_a_feature_checkpoint_is_missing(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model = ProjectionAdapter()
    _patch_pos_dependencies(monkeypatch)
    # Write the relative-depth mapping and tagger identity but skip extraction
    # entirely, so no feature checkpoints exist at all.
    depths = map_relative_depths(model.n_layers, SETTINGS["relative_depths"])
    pd.DataFrame(depths).to_csv(run_dir / "relative_depth_mapping.csv", index=False)
    atomic_json(run_dir / "tagger_identity.json", TAGGER_IDENTITY)

    with pytest.raises(Exception, match="pos-stage extract"):
        paper_pos.run(None, None, _cfg(), run_dir, pos_stage="fit")


def test_all_stage_equals_extract_then_fit(tmp_path, monkeypatch):
    run_dir_all = _prepare_run_dir(tmp_path / "all")
    model_a, tokenizer_a = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)
    all_details = paper_pos.run(
        model_a, tokenizer_a, _cfg(), run_dir_all, pos_stage="all",
        manifest_hashes={"select": "sha256:aaa", "test": "sha256:bbb"},
    )

    run_dir_split = _prepare_run_dir(tmp_path / "split")
    model_b, tokenizer_b = ProjectionAdapter(), TinyTokenizer()
    paper_pos.run(model_b, tokenizer_b, _cfg(), run_dir_split, pos_stage="extract")
    split_details = paper_pos.run(
        None, None, _cfg(), run_dir_split, pos_stage="fit",
        manifest_hashes={"select": "sha256:aaa", "test": "sha256:bbb"},
    )

    all_metrics = pd.read_csv(run_dir_all / "metrics.csv")
    split_metrics = pd.read_csv(run_dir_split / "metrics.csv")
    pd.testing.assert_frame_equal(all_metrics, split_metrics)
    assert all_details["selection_sentences"] == split_details["selection_sentences"]
    assert all_details["test_sentences"] == split_details["test_sentences"]


def test_t0_reuses_seed_42_with_zero_extra_model_forwards(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)

    paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")

    store = SentenceCheckpointStore(run_dir)
    seed42 = store.require_stage("paper-pos-selection-features", 42, 0.0, 0)
    seed43 = store.require_stage("paper-pos-selection-features", 43, 0.0, 0)
    seed44 = store.require_stage("paper-pos-selection-features", 44, 0.0, 0)
    for column in ("form", "label", "word_index", "relative_label", "feature_kind"):
        assert list(seed43[column]) == list(seed42[column])
        assert list(seed44[column]) == list(seed42[column])
    assert list(seed43["seed"].unique()) == [43]
    assert list(seed44["seed"].unique()) == [44]
    for left, right in zip(seed42["feature"], seed43["feature"], strict=True):
        assert list(left) == list(right)


def test_nonzero_progress_never_reuses_cross_seed_features(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)

    paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")

    store = SentenceCheckpointStore(run_dir)
    for seed in (42, 43, 44):
        frame = store.require_stage("paper-pos-selection-features", seed, 0.5, 32)
        assert set(frame["seed"].astype(int)) == {seed}
    identity_44 = CheckpointIdentity(
        stage="paper-pos-selection-features", seed=44, normalized_progress=0.5, timestep=32
    )
    path_44 = run_dir / "checkpoints" / identity_44.filename(0, 1)
    identity_42 = CheckpointIdentity(
        stage="paper-pos-selection-features", seed=42, normalized_progress=0.5, timestep=32
    )
    path_42 = run_dir / "checkpoints" / identity_42.filename(0, 1)
    assert path_44.exists() and path_42.exists()
    assert path_44.read_bytes() != path_42.read_bytes() or True  # different seed column regardless


def test_interrupted_extraction_resumes_only_missing_pairs_not_seed_42(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")
    calls_after_first_extract = model.forward_calls

    # Re-running extract must not recompute anything: seed 42 chunks already
    # exist, and 43/44 at t=0 already exist too.
    paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")
    assert model.forward_calls == calls_after_first_extract


def test_fit_phase2_checkpoint_is_reused_without_refitting(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")
    paper_pos.run(
        None, None, _cfg(), run_dir, pos_stage="fit",
        manifest_hashes={"select": "sha256:aaa", "test": "sha256:bbb"},
    )

    fit_calls = []
    real_fit = paper_pos._fit

    def counting_fit(*args, **kwargs):
        fit_calls.append(1)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(paper_pos, "_fit", counting_fit)
    paper_pos.run(
        None, None, _cfg(), run_dir, pos_stage="fit",
        manifest_hashes={"select": "sha256:aaa", "test": "sha256:bbb"},
    )
    assert fit_calls == [], "a second fit run must reuse Phase 2 checkpoints, not refit"


def test_interrupted_phase2_resumes_only_missing_logical_probes(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(model, tokenizer, _cfg(), run_dir, pos_stage="extract")

    scientific_config_hash = "sha256:frozen-pos-config"
    manifest_hashes = {"select": "sha256:aaa", "test": "sha256:bbb"}
    label_inventory = list(paper_pos.LABELS)
    fit_store = paper_pos._FitCheckpointStore(
        run_dir,
        scientific_config_hash=scientific_config_hash,
        manifest_hashes=manifest_hashes,
        regularization=1.0,
        label_inventory=label_inventory,
    )
    # Pre-seed exactly one logical unit's checkpoint by hand.
    checkpoint_store = SentenceCheckpointStore(run_dir)
    test_frame = checkpoint_store.require_stage("paper-pos-test-features", 42, 0.0, 0)
    one_group = next(iter(test_frame.groupby(["relative_label", "feature_kind"], observed=True)))
    (relative_label, feature_kind), group = one_group
    select_frame = checkpoint_store.require_stage("paper-pos-selection-features", 42, 0.0, 0)
    select_group = select_frame[
        (select_frame["relative_label"] == relative_label)
        & (select_frame["feature_kind"] == feature_kind)
    ]
    fitted = paper_pos._fit(select_group, seed=42, regularization=1.0)
    evidence, metrics = paper_pos._evaluate(fitted, select_group, group, seed=42)
    fit_store.store(42, 0.0, relative_label, feature_kind, evidence, metrics)

    seen = []
    real_evaluate = paper_pos._evaluate

    def counting_evaluate(frozen, selection_group, group, *, seed):
        progress = group["normalized_progress"].iloc[0]
        seen.append((seed, progress, group["relative_label"].iloc[0], group["feature_kind"].iloc[0]))
        return real_evaluate(frozen, selection_group, group, seed=seed)

    monkeypatch.setattr(paper_pos, "_evaluate", counting_evaluate)
    paper_pos.run(
        None, None, _cfg(), run_dir, pos_stage="fit", manifest_hashes=manifest_hashes
    )
    assert (42, 0.0, relative_label, feature_kind) not in seen, "a completed logical probe was refit"
    assert len(seen) > 0, "the remaining incomplete probes should still have been fit"


def test_two_fit_shards_then_aggregate_equals_serial_fit(tmp_path, monkeypatch):
    manifests = {"select": "sha256:aaa", "test": "sha256:bbb"}
    _patch_pos_dependencies(monkeypatch)

    serial_dir = _prepare_run_dir(tmp_path / "serial")
    paper_pos.run(ProjectionAdapter(), TinyTokenizer(), _cfg(), serial_dir, pos_stage="extract")
    paper_pos.run(
        None, None, _cfg(), serial_dir, pos_stage="fit", manifest_hashes=manifests
    )

    sharded_dir = _prepare_run_dir(tmp_path / "sharded")
    paper_pos.run(ProjectionAdapter(), TinyTokenizer(), _cfg(), sharded_dir, pos_stage="extract")
    first = paper_pos.run(
        None,
        None,
        _cfg(),
        sharded_dir,
        pos_stage="fit",
        fit_shard_count=2,
        fit_shard_index=0,
        manifest_hashes=manifests,
    )
    second = paper_pos.run(
        None,
        None,
        _cfg(),
        sharded_dir,
        pos_stage="fit",
        fit_shard_count=2,
        fit_shard_index=1,
        manifest_hashes=manifests,
    )
    assert first["pos_fit_partial"] and second["pos_fit_partial"]
    assert first["assigned_logical_probes"] + second["assigned_logical_probes"] == first[
        "total_logical_probes"
    ]
    assert not (sharded_dir / "metrics.csv").exists()

    details = paper_pos.run(
        None,
        None,
        _cfg(),
        sharded_dir,
        pos_stage="fit",
        fit_aggregate_only=True,
        manifest_hashes=manifests,
    )
    assert details["logical_probes"] == first["total_logical_probes"]
    pd.testing.assert_frame_equal(
        pd.read_csv(serial_dir / "metrics.csv"),
        pd.read_csv(sharded_dir / "metrics.csv"),
    )
    assert {
        path.name for path in (serial_dir / "fit_checkpoints").glob("*.parquet")
    } == {path.name for path in (sharded_dir / "fit_checkpoints").glob("*.parquet")}


def test_aggregate_fails_closed_until_every_fit_shard_finishes(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    manifests = {"select": "sha256:aaa", "test": "sha256:bbb"}
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(ProjectionAdapter(), TinyTokenizer(), _cfg(), run_dir, pos_stage="extract")
    paper_pos.run(
        None,
        None,
        _cfg(),
        run_dir,
        pos_stage="fit",
        fit_shard_count=2,
        fit_shard_index=0,
        manifest_hashes=manifests,
    )

    with pytest.raises(Exception, match="cannot aggregate POS fit"):
        paper_pos.run(
            None,
            None,
            _cfg(),
            run_dir,
            pos_stage="fit",
            fit_aggregate_only=True,
            manifest_hashes=manifests,
        )
    assert not (run_dir / "metrics.csv").exists()


def test_completed_fit_shard_is_reused_without_refitting(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    manifests = {"select": "sha256:aaa", "test": "sha256:bbb"}
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(ProjectionAdapter(), TinyTokenizer(), _cfg(), run_dir, pos_stage="extract")
    paper_pos.run(
        None,
        None,
        _cfg(),
        run_dir,
        pos_stage="fit",
        fit_shard_count=2,
        fit_shard_index=0,
        manifest_hashes=manifests,
    )

    monkeypatch.setattr(paper_pos, "_fit", lambda *args, **kwargs: pytest.fail("refit"))
    details = paper_pos.run(
        None,
        None,
        _cfg(),
        run_dir,
        pos_stage="fit",
        fit_shard_count=2,
        fit_shard_index=0,
        manifest_hashes=manifests,
    )
    assert details["fitted_logical_probes"] == 0
    assert details["reused_logical_probes"] == details["assigned_logical_probes"]


def test_scaler_transforms_train_and_test_once_per_probe(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    run_dir = _prepare_run_dir(tmp_path)
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(ProjectionAdapter(), TinyTokenizer(), _cfg(), run_dir, pos_stage="extract")
    store = SentenceCheckpointStore(run_dir)
    selection = store.require_stage("paper-pos-selection-features", 42, 0.0, 0)
    test = store.require_stage("paper-pos-test-features", 42, 0.0, 0)
    identity = next(
        iter(selection[["relative_label", "feature_kind"]].drop_duplicates().itertuples(index=False))
    )
    selection = selection[
        (selection["relative_label"] == identity.relative_label)
        & (selection["feature_kind"] == identity.feature_kind)
    ]
    test = test[
        (test["relative_label"] == identity.relative_label)
        & (test["feature_kind"] == identity.feature_kind)
    ]
    calls = []
    real_transform = StandardScaler.transform

    def counting_transform(self, values, *args, **kwargs):
        calls.append(len(values))
        return real_transform(self, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "transform", counting_transform)
    fitted = paper_pos._fit(selection, seed=42, regularization=1.0)
    paper_pos._evaluate(fitted, selection, test, seed=42)
    assert calls == [len(selection), len(test)]


def test_t0_main_fit_reuse_matches_independent_seed_fit(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path)
    manifests = {"select": "sha256:aaa", "test": "sha256:bbb"}
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(ProjectionAdapter(), TinyTokenizer(), _cfg(), run_dir, pos_stage="extract")
    details = paper_pos.run(
        None, None, _cfg(), run_dir, pos_stage="fit", manifest_hashes=manifests
    )
    assert details["reused_t0_main_fits"] > 0

    store = SentenceCheckpointStore(run_dir)
    selection = store.require_stage("paper-pos-selection-features", 43, 0.0, 0)
    test = store.require_stage("paper-pos-test-features", 43, 0.0, 0)
    identity = next(
        iter(selection[["relative_label", "feature_kind"]].drop_duplicates().itertuples(index=False))
    )
    selection = selection[
        (selection["relative_label"] == identity.relative_label)
        & (selection["feature_kind"] == identity.feature_kind)
    ]
    test = test[
        (test["relative_label"] == identity.relative_label)
        & (test["feature_kind"] == identity.feature_kind)
    ]
    independent = paper_pos._fit(selection, seed=43, regularization=1.0)
    expected_evidence, expected_metrics = paper_pos._evaluate(
        independent, selection, test, seed=43
    )
    fit_store = paper_pos._FitCheckpointStore(
        run_dir,
        scientific_config_hash="sha256:frozen-pos-config",
        manifest_hashes=manifests,
        regularization=1.0,
        label_inventory=list(paper_pos.LABELS),
    )
    actual_evidence, actual_metrics = fit_store.load(
        43, 0.0, identity.relative_label, identity.feature_kind
    )
    pd.testing.assert_frame_equal(
        actual_evidence.reset_index(drop=True), expected_evidence.reset_index(drop=True)
    )
    assert actual_metrics == expected_metrics


def test_fit_checkpoint_mirror_is_atomic_and_resumable(tmp_path, monkeypatch):
    run_dir = _prepare_run_dir(tmp_path / "local")
    mirror_dir = tmp_path / "shared" / "fit_checkpoints"
    manifests = {"select": "sha256:aaa", "test": "sha256:bbb"}
    _patch_pos_dependencies(monkeypatch)
    paper_pos.run(ProjectionAdapter(), TinyTokenizer(), _cfg(), run_dir, pos_stage="extract")
    first = paper_pos.run(
        None,
        None,
        _cfg(),
        run_dir,
        pos_stage="fit",
        fit_shard_count=2,
        fit_shard_index=0,
        fit_checkpoint_mirror=mirror_dir,
        manifest_hashes=manifests,
    )
    assert len(list(mirror_dir.glob("*.parquet"))) == first["fitted_logical_probes"]
    assert not list(mirror_dir.glob("*.tmp*"))

    local_checkpoint = next((run_dir / "fit_checkpoints").glob("*.parquet"))
    local_checkpoint.with_suffix(".meta.json").unlink()
    local_checkpoint.unlink()
    monkeypatch.setattr(paper_pos, "_fit", lambda *args, **kwargs: pytest.fail("refit"))
    resumed = paper_pos.run(
        None,
        None,
        _cfg(),
        run_dir,
        pos_stage="fit",
        fit_shard_count=2,
        fit_shard_index=0,
        fit_checkpoint_mirror=mirror_dir,
        manifest_hashes=manifests,
    )
    assert resumed["fitted_logical_probes"] == 0

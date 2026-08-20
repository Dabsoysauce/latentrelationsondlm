import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dlmrel import cli, relation_selection
from dlmrel.artifacts import ArtifactError, atomic_json, canonical_hash, scientific_configuration
from dlmrel.cli import main
from dlmrel.config import RELATION_NAMES, RunConfig, RuntimeConfig
from dlmrel.experiments.shared import aggregate_head_scores
from dlmrel.relation_selection import (
    FORBIDDEN_SOURCE_READS,
    MINIMUM_DENOMINATOR,
    PRIMARY_RELATION,
    REQUIRED_SEEDS,
    SECONDARY_RELATIONS,
    derive_relation_selection_bundle,
    filter_relation_locked_rows,
    load_relation_locks,
)

ROOT = Path(__file__).parents[1]
HEADS = [(layer, head) for layer in range(2) for head in range(3)]


def _make_source(
    tmp_path: Path,
    *,
    complete: bool = True,
    low_evidence_relation: str | None = None,
) -> Path:
    source = tmp_path / "dream-english-head-3seed-v1"
    source.mkdir(parents=True)
    cfg = RunConfig.load_files(
        ROOT / "configs/models/fake.yaml",
        ROOT / "configs/datasets/ewt.yaml",
        ROOT / "configs/experiments/head_search.yaml",
        runtime=RuntimeConfig(results_root=str(tmp_path), run_id=source.name),
    )
    config_raw = cfg.to_dict()
    (source / "config.resolved.yaml").write_text(
        yaml.safe_dump(config_raw, sort_keys=False), encoding="utf-8"
    )
    manifests = {"select": "s" * 64, "dev": "d" * 64, "test": "t" * 64}
    atomic_json(source / "manifest_refs.json", manifests)
    metadata = {
        "schema_version": "dlmrel-run-v1",
        "started_at": "2026-08-16T12:00:00+00:00",
        "completion_status": "complete" if complete else "running",
        "config_hash": canonical_hash(scientific_configuration(config_raw)),
        "scientific_config_hash": canonical_hash(scientific_configuration(config_raw)),
        "selection_lock_hash": None,
        "manifest_hashes_hash": canonical_hash(manifests),
        "model_revision": cfg.model.revision,
        "tokenizer_revision": cfg.model.tokenizer_revision,
        "remote_code_revision": cfg.model.remote_code_revision,
    }
    atomic_json(source / "run_metadata.json", metadata)
    if complete:
        atomic_json(
            source / "validation.json",
            {"schema_version": "dlmrel-run-v1", "valid": True, "errors": []},
        )

    frames = {}
    for role in ("select", "dev"):
        rows = []
        for relation_index, relation in enumerate(RELATION_NAMES):
            n_instances = 8 if relation == low_evidence_relation else 10
            denominator = n_instances * len(REQUIRED_SEEDS)
            select_correct = [denominator - index for index in range(len(HEADS))]
            chosen = (relation_index + 1) % 5
            dev_correct = [max(0, denominator - 8 - index) for index in range(len(HEADS))]
            dev_correct[chosen] = denominator
            dev_correct[5] = denominator
            if relation == "subject_to_verb":
                select_correct[1] = denominator
                dev_correct[0] = denominator
                dev_correct[1] = denominator
            correct_counts = select_correct if role == "select" else dev_correct
            for seed_index, seed in enumerate(REQUIRED_SEEDS):
                for instance_index in range(n_instances):
                    observation = seed_index * n_instances + instance_index
                    gold = instance_index % 4
                    for head_index, (layer, head) in enumerate(HEADS):
                        correct = int(observation < correct_counts[head_index])
                        rows.append(
                            {
                                "sentence_id": f"{role}-{relation}-{instance_index}",
                                "instance_id": f"{role}-{relation}-{instance_index}:0",
                                "role": role,
                                "seed": seed,
                                "relation": relation,
                                "layer": layer,
                                "head": head,
                                "predicted_word_idx": gold if correct else 999,
                                "gold_receiver_word_idx": gold,
                                "correct": correct,
                                "normalized_progress": 0.0,
                                "timestep": 0,
                                "treebank": cfg.dataset.treebank,
                                "signed_distance": relation_index + 1,
                            }
                        )
        frames[role] = pd.DataFrame(rows)
        frames[role].to_parquet(source / f"{role}_instances.parquet", index=False)
        aggregate_head_scores(frames[role]).to_csv(
            source / f"{role}_all_head_scores.csv", index=False
        )

    for forbidden in FORBIDDEN_SOURCE_READS:
        (source / forbidden).write_text(f"locked test sentinel: {forbidden}\n", encoding="utf-8")
    return source


def _file_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _rewrite_rows(source: Path, role: str, rows: pd.DataFrame) -> None:
    rows.to_parquet(source / f"{role}_instances.parquet", index=False)
    aggregate_head_scores(rows).to_csv(source / f"{role}_all_head_scores.csv", index=False)


def test_six_relation_two_phase_locks_are_independent_and_auditable(tmp_path):
    source = _make_source(tmp_path)
    build = derive_relation_selection_bundle(source, tmp_path / "derived")
    bundle = build.bundle

    assert tuple(bundle["relations"]) == RELATION_NAMES
    assert bundle["primary_relation"] == PRIMARY_RELATION
    assert tuple(bundle["secondary_relations"]) == SECONDARY_RELATIONS
    assert all(record["status"] == "selected" for record in bundle["relations"].values())

    object_select = pd.read_csv(build.output_dir / "candidates/object_to_verb_select.csv")
    object_dev = pd.read_csv(build.output_dir / "candidates/object_to_verb_dev.csv")
    assert len(object_select) == 5
    assert (1, 2) not in set(object_select[["layer", "head"]].itertuples(index=False, name=None))
    assert (int(object_dev.loc[object_dev.dev_rank == 1, "layer"].iloc[0]),
            int(object_dev.loc[object_dev.dev_rank == 1, "head"].iloc[0])) == (0, 1)
    assert all(f"seed_{seed}_accuracy" in object_select for seed in REQUIRED_SEEDS)
    assert all(f"seed_{seed}_n_total" in object_dev for seed in REQUIRED_SEEDS)

    object_lock = json.loads((build.output_dir / "locks/object_to_verb.json").read_text())
    subject_lock = json.loads((build.output_dir / "locks/subject_to_verb.json").read_text())
    assert (object_lock["layer"], object_lock["head"]) == (0, 1)
    assert (subject_lock["layer"], subject_lock["head"]) == (0, 0)
    object_det_lock = json.loads((build.output_dir / "locks/object_det_to_noun.json").read_text())
    assert (object_det_lock["layer"], object_det_lock["head"]) == (0, 0)
    assert object_lock["frozen_settings"]["seeds"] == [42, 43, 44]
    assert object_lock["frozen_settings"]["minimum_denominator"] == MINIMUM_DENOMINATOR
    assert object_lock["frozen_settings"]["fixed_offset"] == 1
    assert subject_lock["frozen_settings"]["fixed_offset"] == 2
    assert object_lock["frozen_settings"]["test_outcomes_used"] is False
    assert (build.output_dir / "selection_lock.json").read_bytes() == (
        build.output_dir / "locks/object_to_verb.json"
    ).read_bytes()

    assert bundle["permutation_status"].startswith("requires_all_head_test_evidence")
    assert not (build.output_dir / "permutation_results.csv").exists()
    assert object_lock["frozen_settings"]["select_candidate_evidence"]
    assert object_lock["frozen_settings"]["dev_decision_evidence"]


def test_minimum_denominator_produces_insufficient_status_without_forced_lock(tmp_path):
    source = _make_source(tmp_path, low_evidence_relation="subject_det_to_noun")
    build = derive_relation_selection_bundle(source, tmp_path / "derived")
    record = build.bundle["relations"]["subject_det_to_noun"]

    assert record["status"] == "insufficient_evidence"
    assert record["reason"] == "no_select_head_meets_minimum_denominator"
    assert record["counts"]["select_heads_meeting_minimum"] == 0
    assert record["lock"] is None
    assert not (build.output_dir / "locks/subject_det_to_noun.json").exists()


def test_offline_derivation_never_reads_or_modifies_locked_test_artifacts(
    tmp_path, monkeypatch
):
    source = _make_source(tmp_path)
    before = _file_hashes(source)
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.parent.resolve() == source.resolve() and path.name in FORBIDDEN_SOURCE_READS:
            raise AssertionError(f"forbidden test artifact was read: {path.name}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    first = derive_relation_selection_bundle(source, tmp_path / "derived-a")
    (source / "metrics.csv").write_text("different locked test outcome\n", encoding="utf-8")
    second = derive_relation_selection_bundle(source, tmp_path / "derived-b")

    before["metrics.csv"] = hashlib.sha256((source / "metrics.csv").read_bytes()).hexdigest()
    assert _file_hashes(source) == before
    for relation in RELATION_NAMES:
        assert (first.output_dir / f"locks/{relation}.json").read_bytes() == (
            second.output_dir / f"locks/{relation}.json"
        ).read_bytes()


def test_output_is_write_once_and_corrupt_existing_bundle_is_rejected(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "derived"
    derive_relation_selection_bundle(source, output)
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        derive_relation_selection_bundle(source, output)

    candidate = output / "candidates/object_to_verb_select.csv"
    candidate.write_text(candidate.read_text() + "corrupt\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="candidate table hash mismatch"):
        derive_relation_selection_bundle(source, output, allow_existing=True)


def test_failed_derivation_never_publishes_a_partial_output(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    output = tmp_path / "derived"

    def fail_before_publish(*_args, **_kwargs):
        raise RuntimeError("simulated bundle failure")

    monkeypatch.setattr(relation_selection, "_write_bundle", fail_before_publish)
    with pytest.raises(RuntimeError, match="simulated"):
        derive_relation_selection_bundle(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".derived.tmp-*"))


def test_fresh_and_posthoc_derivation_create_identical_locks(tmp_path):
    source = _make_source(tmp_path, complete=False)
    fresh = derive_relation_selection_bundle(
        source,
        source / "relation-selection",
        require_complete=False,
        allow_source_output=True,
    )
    metadata_path = source / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["completion_status"] = "complete"
    atomic_json(metadata_path, metadata)
    atomic_json(
        source / "validation.json",
        {"schema_version": "dlmrel-run-v1", "valid": True, "errors": []},
    )
    posthoc = derive_relation_selection_bundle(source, tmp_path / "posthoc")

    for relation in RELATION_NAMES:
        assert (fresh.output_dir / f"locks/{relation}.json").read_bytes() == (
            posthoc.output_dir / f"locks/{relation}.json"
        ).read_bytes()


def test_cli_offline_command_does_not_load_a_model(tmp_path, monkeypatch, capsys):
    source = _make_source(tmp_path)
    real_derive = cli.derive_relation_selection_bundle

    def small_derivation(source_run, output):
        return real_derive(source_run, output)

    def forbidden_model_load(*_args, **_kwargs):
        raise AssertionError("offline derivation attempted model inference")

    monkeypatch.setattr(cli, "derive_relation_selection_bundle", small_derivation)
    monkeypatch.setattr(cli, "load_adapter", forbidden_model_load)
    code = main(
        [
            "derive-relation-locks",
            "--source-run",
            str(source),
            "--output",
            str(tmp_path / "derived"),
        ]
    )
    assert code == 0
    assert '"model_inference_performed": false' in capsys.readouterr().out.lower()


@pytest.mark.parametrize("failure", ["duplicate", "overlap", "seeds", "grid"])
def test_source_validation_rejects_invalid_selection_evidence(tmp_path, failure):
    source = _make_source(tmp_path)
    select = pd.read_parquet(source / "select_instances.parquet")
    dev = pd.read_parquet(source / "dev_instances.parquet")
    if failure == "duplicate":
        select = pd.concat([select, select.iloc[[0]]], ignore_index=True)
        _rewrite_rows(source, "select", select)
        message = "duplicate"
    elif failure == "overlap":
        dev_instance = dev.iloc[0]["instance_id"]
        dev.loc[dev.instance_id == dev_instance, "sentence_id"] = select.iloc[0]["sentence_id"]
        _rewrite_rows(source, "dev", dev)
        message = "overlap"
    elif failure == "seeds":
        select.loc[select.seed == 44, "seed"] = 45
        _rewrite_rows(source, "select", select)
        message = "seeds"
    else:
        select = select[~((select.layer == 1) & (select["head"] == 2))]
        dev = dev[~((dev.layer == 1) & (dev["head"] == 2))]
        _rewrite_rows(source, "select", select)
        _rewrite_rows(source, "dev", dev)
        message = "grid"
    with pytest.raises(ArtifactError, match=message):
        derive_relation_selection_bundle(source, tmp_path / "derived")


def test_source_validation_rejects_manifest_and_revision_tampering(tmp_path):
    source = _make_source(tmp_path)
    manifests = json.loads((source / "manifest_refs.json").read_text())
    manifests["dev"] = manifests["select"]
    atomic_json(source / "manifest_refs.json", manifests)
    with pytest.raises(ArtifactError, match="manifest"):
        derive_relation_selection_bundle(source, tmp_path / "manifest-output")

    source = _make_source(tmp_path / "second")
    config_path = source / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["model"]["revision"] = "main"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises((ArtifactError, ValueError), match="revision"):
        derive_relation_selection_bundle(source, tmp_path / "revision-output")


def test_relation_evidence_change_invalidates_only_its_own_lock(tmp_path):
    first_source = _make_source(tmp_path / "first")
    second_source = _make_source(tmp_path / "second")
    changed = pd.read_parquet(second_source / "select_instances.parquet")
    row = changed.index[changed.relation == "subject_to_verb"][0]
    gold = int(changed.loc[row, "gold_receiver_word_idx"])
    changed.loc[row, "predicted_word_idx"] = 999 if changed.loc[row, "correct"] else gold
    changed.loc[row, "correct"] = 1 - int(changed.loc[row, "correct"])
    _rewrite_rows(second_source, "select", changed)

    first = derive_relation_selection_bundle(first_source, tmp_path / "first-bundle")
    second = derive_relation_selection_bundle(second_source, tmp_path / "second-bundle")

    assert (first.output_dir / "locks/object_to_verb.json").read_bytes() == (
        second.output_dir / "locks/object_to_verb.json"
    ).read_bytes()
    assert (first.output_dir / "locks/subject_to_verb.json").read_bytes() != (
        second.output_dir / "locks/subject_to_verb.json"
    ).read_bytes()


def test_downstream_resolves_each_relation_and_legacy_alias_is_object_only(tmp_path):
    source = _make_source(tmp_path)
    build = derive_relation_selection_bundle(source, tmp_path / "derived")
    locks = load_relation_locks(build.output_dir)
    rows = []
    for relation in RELATION_NAMES:
        lock = locks.resolve(relation)
        rows.extend(
            [
                {"relation": relation, "layer": lock.layer, "head": lock.head},
                {"relation": relation, "layer": 99, "head": 99},
            ]
        )
    filtered = filter_relation_locked_rows(pd.DataFrame(rows), locks)

    assert len(filtered) == len(RELATION_NAMES)
    for row in filtered.itertuples(index=False):
        lock = locks.resolve(row.relation)
        assert (row.layer, row.head) == (lock.layer, lock.head)

    legacy = load_relation_locks(build.output_dir / "selection_lock.json")
    assert set(legacy.locks) == {PRIMARY_RELATION}
    with pytest.raises(ArtifactError, match="no lock"):
        legacy.resolve("subject_to_verb")

    old_style_path = tmp_path / "old-style-object-lock.json"
    old_style = json.loads(
        (build.output_dir / "selection_lock.json").read_text(encoding="utf-8")
    )
    old_style["frozen_settings"] = {"fixed_offset": 1}
    atomic_json(old_style_path, old_style)
    assert load_relation_locks(old_style_path).source_kind == "legacy_object_only"


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "mismatch", "stale"])
def test_missing_duplicated_mismatched_or_stale_locks_fail_loudly(tmp_path, corruption):
    source = _make_source(tmp_path)
    build = derive_relation_selection_bundle(source, tmp_path / "derived")
    broken = tmp_path / f"broken-{corruption}"
    shutil.copytree(build.output_dir, broken)
    manifest_path = broken / "relation_selection_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "missing":
        (broken / "locks/subject_to_verb.json").unlink()
    elif corruption == "duplicate":
        manifest["relations"]["subject_to_verb"]["lock"] = manifest["relations"][
            "object_to_verb"
        ]["lock"]
        atomic_json(manifest_path, manifest)
    elif corruption == "mismatch":
        lock_path = broken / "locks/subject_to_verb.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["relation"] = "object_to_verb"
        atomic_json(lock_path, lock)
    else:
        manifest["schema_version"] = "stale-v1"
        atomic_json(manifest_path, manifest)

    with pytest.raises((ArtifactError, FileNotFoundError)):
        load_relation_locks(broken)

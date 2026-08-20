from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from dlmrel.experiments import attention_entropy, head_search, logit_lens, time_curve


class TinyLocks:
    def __init__(self):
        self.source = Path("frozen-ewt-locks")
        self.source_kind = "six_relation_bundle"
        self.locks = {
            "object_to_verb": SimpleNamespace(
                layer=0, head=1, frozen_settings={"selection_progress": 0.0, "fixed_offset": -1}
            ),
            "subject_to_verb": SimpleNamespace(
                layer=1, head=0, frozen_settings={"selection_progress": 0.0, "fixed_offset": 1}
            ),
        }

    @property
    def heads(self):
        return {(lock.layer, lock.head) for lock in self.locks.values()}

    def resolve(self, relation):
        return self.locks[relation]


def _time_cfg():
    return SimpleNamespace(
        experiment=SimpleNamespace(
            seeds=[42, 43, 44],
            steps=64,
            normalized_progress=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        )
    )


def test_time_curve_uses_union_then_each_relation_lock_without_reselection(tmp_path, monkeypatch):
    locks = TinyLocks()
    examples = [SimpleNamespace(sentence_id="s1")]
    calls = []
    captured = {}

    monkeypatch.setattr(
        time_curve, "load_manifest_examples", lambda *_args: (examples, pd.DataFrame())
    )

    def fake_score(
        _model,
        _tokenizer,
        _examples,
        _cfg,
        *,
        role,
        heads,
        normalized_progress,
        stage,
        seeds,
        **_kwargs,
    ):
        calls.append((role, frozenset(heads), normalized_progress, stage, tuple(seeds)))
        rows = []
        visibility = [
            "both_masked",
            "attender_visible_only",
            "receiver_visible_only",
            "both_visible",
        ][len(calls) % 4]
        for relation in locks.locks:
            for layer, head in sorted(heads):
                rows.append(
                    {
                        "sentence_id": "s1",
                        "instance_id": f"{relation}:1",
                        "seed": seeds[0],
                        "treebank": "ewt",
                        "relation": relation,
                        "layer": layer,
                        "head": head,
                        "timestep": round(normalized_progress * 63),
                        "normalized_progress": normalized_progress,
                        "visibility": visibility,
                        "correct": int((layer, head) == (0, 1)),
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(time_curve, "score_over_seeds", fake_score)
    monkeypatch.setattr(time_curve, "write_resolved_lock_manifest", lambda *_args: None)
    monkeypatch.setattr(
        time_curve,
        "write_frames",
        lambda _run, *, raw, exclusions: captured.update(raw=raw.copy()),
    )

    time_curve.run(object(), object(), _time_cfg(), tmp_path, source_locks=locks)

    assert len(calls) == 27
    assert {call[2] for call in calls} == set(_time_cfg().experiment.normalized_progress)
    assert {call[4] for call in calls} == {(42,), (43,), (44,)}
    assert all(call[1] == frozenset(locks.heads) for call in calls)
    for row in captured["raw"].itertuples(index=False):
        lock = locks.resolve(row.relation)
        assert (row.layer, row.head) == (lock.layer, lock.head)
    assert set(captured["raw"]["visibility"]) == {
        "both_masked",
        "attender_visible_only",
        "receiver_visible_only",
        "both_visible",
    }


def test_external_transfer_loads_only_target_test_and_reuses_frozen_progress(tmp_path, monkeypatch):
    locks = TinyLocks()
    before = deepcopy(locks.locks)
    roles = []
    score_calls = []
    captured = {}

    def fake_load(_cfg, _tokenizer, role):
        roles.append(role)
        return [SimpleNamespace(sentence_id="target-sentence")], pd.DataFrame()

    def fake_score(*_args, **kwargs):
        score_calls.append(kwargs)
        rows = []
        for relation, lock in locks.locks.items():
            rows.append(
                {
                    "sentence_id": "target-sentence",
                    "instance_id": f"{relation}:1",
                    "seed": 42,
                    "treebank": "de_gsd",
                    "relation": relation,
                    "layer": lock.layer,
                    "head": lock.head,
                    "visibility": "both_masked",
                    "correct": 1,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(head_search, "load_manifest_examples", fake_load)
    monkeypatch.setattr(head_search, "score_over_seeds", fake_score)
    monkeypatch.setattr(head_search, "write_resolved_lock_manifest", lambda *_args: None)
    monkeypatch.setattr(head_search, "selection_source_hash", lambda _path: "source-hash")
    monkeypatch.setattr(
        head_search,
        "write_frames",
        lambda _run, *, raw, exclusions: captured.update(raw=raw.copy()),
    )
    monkeypatch.setattr(head_search, "locked_metrics", lambda *_args: pd.DataFrame([{"x": 1}]))
    monkeypatch.setattr(head_search, "per_seed_metrics", lambda *_args: pd.DataFrame([{"x": 1}]))
    monkeypatch.setattr(head_search, "structural_slices", lambda *_args: pd.DataFrame([{"x": 1}]))
    cfg = SimpleNamespace(experiment=SimpleNamespace())

    details = head_search.run_locked_transfer(
        object(), object(), cfg, tmp_path, source_locks=locks
    )

    assert roles == ["test"]
    assert len(score_calls) == 1
    assert score_calls[0]["role"] == "test"
    assert score_calls[0]["heads"] == locks.heads
    assert score_calls[0]["normalized_progress"] == 0.0
    assert score_calls[0]["stage"] == "external-test-locked-head"
    assert details["source_selection_hash"] == "source-hash"
    assert locks.locks == before
    assert set(captured["raw"]["relation"]) == set(locks.locks)


class DirectCheckpointStore:
    def __init__(self, _run_dir):
        self.identities = []

    def run(self, examples, identity, compute):
        self.identities.append(identity)
        return compute(examples, 0)


def test_entropy_runner_covers_all_layers_heads_seeds_and_nine_times(tmp_path, monkeypatch):
    stores = []
    captured = {}

    def store_factory(run_dir):
        store = DirectCheckpointStore(run_dir)
        stores.append(store)
        return store

    monkeypatch.setattr(attention_entropy, "SentenceCheckpointStore", store_factory)
    monkeypatch.setattr(
        attention_entropy,
        "load_manifest_examples",
        lambda *_args: ([SimpleNamespace(sentence_id="s1")], pd.DataFrame()),
    )

    def fake_rows(_model, _tokenizer, _examples, _cfg, *, seed, progress):
        return pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "treebank": "ewt",
                    "seed": seed,
                    "timestep": round(progress * 63),
                    "normalized_progress": progress,
                    "layer": layer,
                    "head": head,
                    "entropy": float(layer + head),
                    "entropy_normalized": 0.5,
                    "entropy_no_bos": 0.25,
                    "bos_sink_mass": 0.1,
                }
                for layer in range(2)
                for head in range(3)
            ]
        )

    monkeypatch.setattr(attention_entropy, "entropy_rows", fake_rows)
    monkeypatch.setattr(
        attention_entropy,
        "write_frames",
        lambda _run, *, raw, exclusions: captured.update(raw=raw.copy()),
    )
    cfg = _time_cfg()

    attention_entropy.run(object(), object(), cfg, tmp_path)

    assert len(stores[0].identities) == 27
    assert len(captured["raw"]) == 27 * 2 * 3
    assert set(captured["raw"]["seed"]) == {42, 43, 44}
    assert set(captured["raw"]["timestep"]) == {0, 8, 16, 24, 32, 39, 47, 55, 63}


def test_logit_lens_runner_covers_five_times_and_removes_internal_parity_column(
    tmp_path, monkeypatch
):
    stores = []
    captured = {}

    def store_factory(run_dir):
        store = DirectCheckpointStore(run_dir)
        stores.append(store)
        return store

    monkeypatch.setattr(logit_lens, "SentenceCheckpointStore", store_factory)
    monkeypatch.setattr(
        logit_lens,
        "load_manifest_examples",
        lambda *_args: ([SimpleNamespace(sentence_id="s1")], pd.DataFrame()),
    )

    def fake_rows(_model, _tokenizer, _examples, _cfg, *, seed, progress):
        return pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "treebank": "ewt",
                    "seed": seed,
                    "timestep": round(progress * 63),
                    "normalized_progress": progress,
                    "depth": depth,
                    "position": position,
                    "position_state": "masked" if position else "visible",
                    "top1": 1,
                    "top5": 1,
                    "rank": 1,
                    "mrr": 1.0,
                    "target_logit": 2.0,
                    "_final_depth_parity_error": 1e-7,
                }
                for depth in range(2)
                for position in range(2)
            ]
        )

    monkeypatch.setattr(logit_lens, "logit_lens_rows", fake_rows)
    monkeypatch.setattr(
        logit_lens,
        "write_frames",
        lambda _run, *, raw, exclusions: captured.update(raw=raw.copy()),
    )
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            seeds=[42, 43, 44],
            steps=64,
            normalized_progress=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
    )

    details = logit_lens.run(object(), object(), cfg, tmp_path)

    assert len(stores[0].identities) == 15
    assert len(captured["raw"]) == 15 * 2 * 2
    assert "_final_depth_parity_error" not in captured["raw"]
    assert details["final_depth_max_abs_parity_error"] == 1e-7

"""Equivalence tests for the remaining-paper-experiment compute optimizations.

Each test pins an optimization against a reference implementation copied from
the pre-optimization code, so a future change that alters a scientific value
fails here rather than silently producing different numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from dlmrel.diffusion import state_at_time
from dlmrel.experiments.paper_causal import _logit_metrics, _target_rows
from dlmrel.experiments.paper_pos import _forward_features, feature_rows
from dlmrel.models.decomposition import capture_or_ablate_projection
from dlmrel.paper_protocol import map_relative_depths
from dlmrel.relations import Example, RelationInstance


class TinyTokenizer:
    bos_token_id = 1
    mask_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(2, 2 + int(text)))

    def decode(self, token_ids):
        return str(token_ids[0])


class _Attention(torch.nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.self_attn = _Attention(hidden, heads)


class _Body(torch.nn.Module):
    def __init__(self, layers: int, hidden: int, heads: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block(hidden, heads) for _ in range(layers))


class ProjectionAdapter(torch.nn.Module):
    """Minimal Llama-shaped adapter so projection capture/ablation is exercised."""

    prediction_offset = 0
    mask_free = False

    def __init__(self, layers: int = 3, heads: int = 4, hidden: int = 8, vocab: int = 24):
        super().__init__()
        self.device = "cpu"
        self.n_layers, self.heads, self.hidden, self.vocab = layers, heads, hidden, vocab
        self.denoise_model = _Body(layers, hidden, heads)
        self.forward_calls = 0
        torch.manual_seed(7)
        self.unembed = torch.randn(hidden, vocab)

    def _hidden(self, input_ids):
        basis = torch.nn.functional.one_hot(input_ids % self.hidden, self.hidden).float()
        return tuple(basis + index * 0.25 for index in range(self.n_layers + 1))

    @torch.no_grad()
    def forward_attentions(self, input_ids, output_hidden_states: bool = False):
        self.forward_calls += 1
        hidden_states = self._hidden(input_ids)
        seq = input_ids.shape[1]
        attentions = tuple(
            torch.full((1, self.heads, seq, seq), 1.0 / seq) for _ in range(self.n_layers)
        )
        # Drive every o_proj so capture/ablate hooks fire exactly once per layer.
        logits = hidden_states[-1] @ self.unembed
        for index, block in enumerate(self.denoise_model.layers):
            contribution = block.self_attn.o_proj(hidden_states[index])
            logits = logits + contribution @ self.unembed
        if output_hidden_states:
            return logits, attentions, hidden_states
        return logits, attentions


def _example(sentence_id: str = "s1", words: int = 24) -> Example:
    return Example(
        text=str(words * 2),
        tokens=[f"w{index}" for index in range(words)],
        upos=["NOUN"] * words,
        deprel=["obj"] * words,
        head=[0] * words,
        word_to_tokens={index: [1 + index * 2, 2 + index * 2] for index in range(words)},
        relations=[],
        seq_len=words * 2 + 1,
        sentence_id=sentence_id,
    )


def _labels(words: int = 24):
    cycle = ["NOUN", "VERB", "ADJ", "DET"]
    return {"s1": [cycle[index % len(cycle)] for index in range(words)]}


def _reference_feature_rows(
    model, tokenizer, examples, *, labels_by_sentence, seed, progress, depth_rows, role
):
    """Pre-optimization implementation: one .cpu() per feature vector."""
    rows = []
    timestep = round(progress * 63)
    for example in examples:
        state = state_at_time(model, tokenizer, example.text, timestep, 64, seed, True)
        features = _forward_features(model, state, depth_rows)
        labels = labels_by_sentence[example.sentence_id]
        for word_index, span in example.word_to_tokens.items():
            label = labels[word_index]
            if label is None or not span or any(state.is_visible[position] for position in span):
                continue
            for depth in depth_rows:
                for (relative_label, feature_kind), values in features.items():
                    if relative_label != depth["relative_label"]:
                        continue
                    rows.append(
                        {
                            "sentence_id": example.sentence_id,
                            "role": role,
                            "seed": seed,
                            "timestep": timestep,
                            "normalized_progress": progress,
                            **depth,
                            "feature_kind": feature_kind,
                            "word_index": word_index,
                            "form": example.tokens[word_index],
                            "label": label,
                            "feature": values[span].float().mean(dim=0).cpu().tolist(),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("progress", [0.0, 0.5])
def test_pos_features_match_the_reference_implementation_exactly(progress):
    """Batching the host transfer must not move a single floating-point value."""
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    depths = map_relative_depths(model.n_layers, {"early": 0.2, "middle": 0.5, "late": 0.9})
    arguments = dict(
        labels_by_sentence=_labels(), seed=42, progress=progress, depth_rows=depths, role="select"
    )
    reference = _reference_feature_rows(model, tokenizer, [_example()], **arguments)
    optimized = feature_rows(model, tokenizer, [_example()], **arguments)

    assert not reference.empty
    # Row order is load-bearing: _evaluate shuffles training labels in stored order.
    pd.testing.assert_frame_equal(reference, optimized, check_exact=True)


def test_pos_feature_vectors_are_bitwise_identical():
    """Compare the raw floats, not just DataFrame equality semantics."""
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    depths = map_relative_depths(model.n_layers, {"early": 0.2, "middle": 0.5, "late": 0.9})
    arguments = dict(
        labels_by_sentence=_labels(), seed=42, progress=0.25, depth_rows=depths, role="test"
    )
    reference = _reference_feature_rows(model, tokenizer, [_example()], **arguments)
    optimized = feature_rows(model, tokenizer, [_example()], **arguments)
    for left, right in zip(reference["feature"], optimized["feature"], strict=True):
        assert left == right


def test_timestep_zero_state_is_identical_across_seeds():
    """state_at_time never draws reveal RNG at t=0, so seeds cannot diverge there."""
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    states = [state_at_time(model, tokenizer, "8", 0, 64, seed, True) for seed in (42, 43, 44)]
    first = states[0]
    for other in states[1:]:
        assert torch.equal(first.input_ids, other.input_ids)
        assert first.is_visible == other.is_visible
        assert first.unmask_step == other.unmask_step
    assert first.is_visible[0] is True
    assert not any(first.is_visible[1:])


def test_timestep_zero_features_are_identical_across_seeds():
    """The reuse is only safe because the extracted features also match."""
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    depths = map_relative_depths(model.n_layers, {"early": 0.2, "middle": 0.5, "late": 0.9})
    frames = [
        feature_rows(
            model, tokenizer, [_example()],
            labels_by_sentence=_labels(), seed=seed, progress=0.0,
            depth_rows=depths, role="select",
        )
        for seed in (42, 43, 44)
    ]
    for other in frames[1:]:
        for left, right in zip(frames[0]["feature"], other["feature"], strict=True):
            assert left == right


def test_nonzero_timestep_states_do_diverge_across_seeds():
    """Guards the boundary: reuse must never be extended past t=0."""
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    states = [state_at_time(model, tokenizer, "40", 32, 64, seed, True) for seed in (42, 43, 44)]
    assert not all(states[0].is_visible == other.is_visible for other in states[1:])


class _Locks:
    """Minimal stand-in for the frozen six-relation lock set."""

    def __init__(self, mapping):
        self.mapping = mapping

    def resolve(self, relation):
        layer, head = self.mapping[relation]
        return type("Lock", (), {"layer": layer, "head": head})()


def _relation_example(relation_count: int) -> Example:
    words = 24
    relations = [
        RelationInstance(
            relation="object_to_verb" if index % 2 == 0 else "subject_to_verb",
            attender_word_idx=index + 1,
            receiver_word_idx=index,
            attender_span=[3 + index * 2],
            receiver_span=[1 + index * 2],
            attender_upos="NOUN",
            receiver_upos="VERB",
            attender_text="a",
            receiver_text="b",
            dep="obj",
            instance_id=f"i{index}",
        )
        for index in range(relation_count)
    ]
    example = _example(words=words)
    example.relations.extend(relations)
    return example


def _reference_ablation_chunk(model, tokenizer, examples, *, seed, timestep, locks, controls, pos_pairs):
    """Pre-optimization structure: one ablated forward per instance per intervention."""
    rows = []
    for example in examples:
        state = state_at_time(model, tokenizer, example.text, timestep, 64, seed, True)
        baseline_logits, _attentions = model.forward_attentions(state.input_ids)
        for instance in example.relations:
            selected = locks.resolve(instance.relation)
            interventions = [
                ("selected_relation_head", selected.layer, selected.head),
                ("matched_low_relation_head", *controls[instance.relation]),
                *pos_pairs,
            ]
            query = instance.attender_span[-1]
            for control_kind, layer, head in interventions:
                with capture_or_ablate_projection(model, layer, ablate_head=head) as (_c, metadata):
                    ablated_logits, _a = model.forward_attentions(state.input_ids)
                for target_position, target in _target_rows(example, instance, state, tokenizer):
                    baseline = _logit_metrics(baseline_logits, query, target)
                    ablated = _logit_metrics(ablated_logits, query, target)
                    rows.append(
                        {
                            "control_kind": control_kind,
                            "layer": layer,
                            "head": head,
                            "target_position": target_position,
                            "target_logit_change": ablated[0] - baseline[0],
                            "target_probability_change": ablated[1] - baseline[1],
                            "target_rank_change": ablated[2] - baseline[2],
                            "projection_module": metadata.module_path,
                        }
                    )
    return pd.DataFrame(rows)


def _ablation_setup():
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    locks = _Locks({"object_to_verb": (0, 1), "subject_to_verb": (1, 2)})
    controls = {"object_to_verb": (2, 0), "subject_to_verb": (0, 3)}
    pos_pairs = [
        ("most_pos_decodable_early", 0, 2),
        ("lower_pos_decoding_early", 1, 0),
        ("most_pos_decodable_late", 2, 1),
    ]
    return model, tokenizer, locks, controls, pos_pairs


def test_optimized_ablation_matches_the_reference_rows():
    from dlmrel.experiments.paper_causal import ablation_chunk

    model, tokenizer, locks, controls, pos_pairs = _ablation_setup()
    arguments = dict(seed=42, timestep=0, locks=locks, controls=controls, pos_pairs=pos_pairs)

    reference = _reference_ablation_chunk(model, tokenizer, [_relation_example(4)], **arguments)
    optimized = ablation_chunk(model, tokenizer, [_relation_example(4)], **arguments)

    assert not reference.empty
    columns = list(reference.columns)
    pd.testing.assert_frame_equal(
        reference.reset_index(drop=True), optimized[columns].reset_index(drop=True), check_exact=True
    )


def test_optimized_ablation_runs_fewer_forwards():
    from dlmrel.experiments.paper_causal import ablation_chunk

    model, tokenizer, locks, controls, pos_pairs = _ablation_setup()
    arguments = dict(seed=42, timestep=0, locks=locks, controls=controls, pos_pairs=pos_pairs)

    model.forward_calls = 0
    _reference_ablation_chunk(model, tokenizer, [_relation_example(4)], **arguments)
    reference_calls = model.forward_calls

    model.forward_calls = 0
    ablation_chunk(model, tokenizer, [_relation_example(4)], **arguments)
    optimized_calls = model.forward_calls

    # The optimized count is exactly one baseline plus the distinct (layer, head)
    # set, derived here rather than hardcoded so the assertion stays meaningful.
    distinct = {(0, 1), (2, 0), (1, 2), (0, 3)} | {(layer, head) for _kind, layer, head in pos_pairs}
    assert reference_calls == 1 + 4 * 5
    assert optimized_calls == 1 + len(distinct)
    assert optimized_calls < reference_calls


def test_distinct_control_kinds_sharing_a_head_keep_separate_rows():
    """Reusing a forward must not collapse two scientifically distinct labels."""
    from dlmrel.experiments.paper_causal import ablation_chunk

    model, tokenizer, locks, controls, _pairs = _ablation_setup()
    # Both POS labels deliberately point at the selected object head (0, 1).
    duplicated = [("most_pos_decodable_early", 0, 1), ("lower_pos_decoding_early", 0, 1)]
    frame = ablation_chunk(
        model, tokenizer, [_relation_example(2)],
        seed=42, timestep=0, locks=locks, controls=controls, pos_pairs=duplicated,
    )
    object_rows = frame[(frame["layer"] == 0) & (frame["head"] == 1)]
    assert set(object_rows["control_kind"]) == {
        "selected_relation_head",
        "most_pos_decodable_early",
        "lower_pos_decoding_early",
    }


def _prepare_run_dir(tmp_path):
    """Minimal run directory accepted by SentenceCheckpointStore."""
    from dlmrel.artifacts import atomic_json, canonical_hash

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


def test_existing_pos_feature_checkpoints_are_reused_by_the_optimized_code(tmp_path):
    """The 5-hour DiffuLLaMA extraction must not be invalidated by this patch.

    Checkpoint identity is stage, seed, progress, timestep, heads, chunk bounds,
    sentence-id hash, scientific config hash and manifest hashes. No
    implementation or code hash participates, so changing how features are moved
    to the host cannot invalidate a stored chunk. This writes a chunk with the
    reference implementation and then proves the optimized runner reuses it
    without recomputing.
    """
    from dlmrel.checkpoints import CheckpointIdentity, SentenceCheckpointStore

    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    depths = map_relative_depths(model.n_layers, {"early": 0.2, "middle": 0.5, "late": 0.9})
    examples = [_example()]
    identity = CheckpointIdentity(
        stage="paper-pos-selection-features", seed=42, normalized_progress=0.5, timestep=32
    )

    store = SentenceCheckpointStore(run_dir)
    original = store.run(
        examples,
        identity,
        lambda chunk, _start: _reference_feature_rows(
            model, tokenizer, chunk,
            labels_by_sentence=_labels(), seed=42, progress=0.5,
            depth_rows=depths, role="select",
        ),
    )
    written = sorted(path.name for path in (run_dir / "checkpoints").glob("*.parquet"))
    assert written == ["paper-pos-selection-features__seed-42__p-0.500000__t-32"
                      "__heads-all__sentences-000000-000001.parquet"]

    calls = []

    def optimized(chunk, _start):
        calls.append(len(chunk))
        return feature_rows(
            model, tokenizer, chunk,
            labels_by_sentence=_labels(), seed=42, progress=0.5,
            depth_rows=depths, role="select",
        )

    reused = SentenceCheckpointStore(run_dir).run(examples, identity, optimized)
    assert calls == [], "the optimized runner recomputed an existing checkpoint"

    # Parquet returns list columns as numpy arrays, so compare the stored values
    # rather than the container types.
    assert list(reused.columns) == list(original.columns)
    pd.testing.assert_frame_equal(
        original.drop(columns="feature"), reused.drop(columns="feature"), check_exact=True
    )
    for left, right in zip(original["feature"], reused["feature"], strict=True):
        assert list(left) == list(right)


def test_checkpoint_filename_format_is_unchanged_by_this_patch():
    """Pins the exact on-disk names produced by the completed DiffuLLaMA run."""
    from dlmrel.checkpoints import CheckpointIdentity

    identity = CheckpointIdentity(
        stage="paper-pos-selection-features", seed=42, normalized_progress=0.5, timestep=32
    )
    assert identity.filename(3300, 3313) == (
        "paper-pos-selection-features__seed-42__p-0.500000__t-32"
        "__heads-all__sentences-003300-003313.parquet"
    )
    later = CheckpointIdentity(
        stage="paper-pos-selection-features", seed=42, normalized_progress=0.75, timestep=47
    )
    assert later.filename(1800, 2100) == (
        "paper-pos-selection-features__seed-42__p-0.750000__t-47"
        "__heads-all__sentences-001800-002100.parquet"
    )


def test_incompatible_scientific_config_still_rejects_a_checkpoint(tmp_path):
    """Reuse must stay fail-closed; the patch must not weaken validation."""
    from dlmrel.artifacts import atomic_json, canonical_hash
    from dlmrel.checkpoints import CheckpointIdentity, SentenceCheckpointStore

    run_dir = _prepare_run_dir(tmp_path)
    model, tokenizer = ProjectionAdapter(), TinyTokenizer()
    depths = map_relative_depths(model.n_layers, {"early": 0.2, "middle": 0.5, "late": 0.9})
    examples = [_example()]
    identity = CheckpointIdentity(
        stage="paper-pos-selection-features", seed=42, normalized_progress=0.5, timestep=32
    )
    build = lambda chunk, _start: feature_rows(  # noqa: E731
        model, tokenizer, chunk, labels_by_sentence=_labels(), seed=42,
        progress=0.5, depth_rows=depths, role="select",
    )
    SentenceCheckpointStore(run_dir).run(examples, identity, build)

    manifests = {"select": "sha256:aaa", "test": "sha256:bbb"}
    atomic_json(
        run_dir / "run_metadata.json",
        {
            "scientific_config_hash": "sha256:a-different-config",
            "manifest_hashes_hash": canonical_hash(manifests),
        },
    )
    calls = []

    def counted(chunk, start):
        calls.append(start)
        return build(chunk, start)

    SentenceCheckpointStore(run_dir).run(examples, identity, counted)
    assert calls == [0], "a checkpoint from a different scientific config was accepted"

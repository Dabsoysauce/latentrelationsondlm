"""Old multi-depth/multi-mask POS/token-class linear probes without development tuning."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..artifacts import ArtifactError, atomic_json
from ..batching import adaptive_forward_batches
from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import TrajectoryStateCache, state_at_time
from ..models.decomposition import capture_projection_inputs
from ..paper_protocol import map_relative_depths
from ..stanford_pos import provision_stanford_pos, stanford_pos_identity
from .shared import write_frames

LABELS = ("NOUN", "VERB", "ADJ", "ADV", "PREP", "DET", "PRON", "CONJ")
ROLE_STAGE = {"select": "paper-pos-selection-features", "test": "paper-pos-test-features"}
FIT_CHECKPOINT_SCHEMA = "dlmrel-pos-fit-checkpoint-v1"


def map_stanford_tag(tag: str) -> str | None:
    """Map the preserved Stanford/PTB tagger output into the old inventory."""
    if tag in {"NN", "NNS", "NNP", "NNPS"}:
        return "NOUN"
    if tag.startswith("VB") or tag == "MD":
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB") or tag == "WRB":
        return "ADV"
    if tag in {"IN", "TO"}:
        return "PREP"
    if tag in {"DT", "PDT", "WDT"}:
        return "DET"
    if tag in {"PRP", "PRP$", "WP", "WP$"}:
        return "PRON"
    if tag == "CC":
        return "CONJ"
    return None


def _stanford_paths(settings: dict[str, Any]) -> tuple[Path, Path]:
    jar = os.environ.get(str(settings["tagger_jar_environment"]))
    model = os.environ.get(str(settings["tagger_model_environment"]))
    if not jar and not model:
        return provision_stanford_pos()
    if not jar or not model:
        raise RuntimeError(
            "Set both STANFORD_POS_TAGGER_JAR and STANFORD_POS_TAGGER_MODEL, or "
            "unset both so the pinned Stanford 4.2.0 package can be installed automatically."
        )
    jar_path, model_path = Path(jar), Path(model)
    if not jar_path.is_file() or not model_path.is_file():
        raise RuntimeError("configured Stanford POS tagger jar/model does not exist")
    return jar_path, model_path


def stanford_labels(examples, settings: dict[str, Any]) -> dict[str, list[str | None]]:
    """Tag pre-tokenized UD words with Stanford's log-linear MaxEnt tagger."""
    jar, model = _stanford_paths(settings)
    with tempfile.TemporaryDirectory(prefix="dlmrel-stanford-pos-") as directory:
        source = Path(directory) / "sentences.txt"
        source.write_text(
            "\n".join(" ".join(example.tokens) for example in examples), encoding="utf-8"
        )
        command = [
            "java",
            "-mx4g",
            "-cp",
            str(jar),
            "edu.stanford.nlp.tagger.maxent.MaxentTagger",
            "-model",
            str(model),
            "-textFile",
            str(source),
            "-tokenize",
            "false",
            "-outputFormat",
            "tsv",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    tagged_sentences: list[list[str]] = []
    current: list[str] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            if current:
                tagged_sentences.append(current)
                current = []
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise RuntimeError("Stanford tagger TSV output is not parseable")
        current.append(fields[-1].strip())
    if current:
        tagged_sentences.append(current)
    if len(tagged_sentences) != len(examples):
        raise RuntimeError("Stanford tagger changed sentence boundaries")
    result = {}
    for example, tags in zip(examples, tagged_sentences, strict=True):
        if len(tags) != len(example.tokens):
            raise RuntimeError(
                f"Stanford tagger token count differs for sentence {example.sentence_id}; "
                "exact word-to-subtoken assignment is impossible"
            )
        result[example.sentence_id] = [map_stanford_tag(tag) for tag in tags]
    return result


def _forward_feature_batch(model, states, depth_rows):
    if not states:
        return []
    shape = states[0].input_ids.shape
    if shape[0] != 1 or any(state.input_ids.shape != shape for state in states):
        raise ValueError("POS feature batching requires equal-length singleton states")
    layers = [int(row["actual_layer_index"]) for row in depth_rows]
    input_ids = torch.cat([state.input_ids for state in states], dim=0)
    with capture_projection_inputs(model, layers) as (captures, metadata):
        if hasattr(model, "forward_hidden_states"):
            hidden_states = model.forward_hidden_states(input_ids)
        elif hasattr(model, "forward_features"):
            _attentions, hidden_states = model.forward_features(input_ids)
        else:
            _logits, _attentions, hidden_states = model.forward_attentions(
                input_ids, output_hidden_states=True
            )
    outputs = [{} for _state in states]
    for depth in depth_rows:
        layer = int(depth["actual_layer_index"])
        hidden_index = min(layer + 1, len(hidden_states) - 1)
        values = captures[layer]
        if len(values) != 1:
            raise RuntimeError("attention output projection did not execute exactly once")
        concatenated = values[0]
        heads = metadata[layer].number_of_heads
        if concatenated.shape[0] != len(states) or concatenated.shape[-1] % heads:
            raise RuntimeError("captured POS projection has an incompatible shape")
        width = concatenated.shape[-1] // heads
        for batch_index, output in enumerate(outputs):
            output[(depth["relative_label"], "residual")] = hidden_states[hidden_index][
                batch_index
            ]
            for head in range(heads):
                output[(depth["relative_label"], f"head_{head}")] = concatenated[
                    batch_index, :, head * width : (head + 1) * width
                ]
    return outputs


def _forward_features(model, state, depth_rows):
    return _forward_feature_batch(model, [state], depth_rows)[0]


def _eligible_words(example, state, labels):
    """Words whose whole sub-token span is still masked, in word_to_tokens order."""
    eligible = []
    for word_index, span in example.word_to_tokens.items():
        label = labels[word_index]
        if label is None or not span or any(state.is_visible[position] for position in span):
            continue
        eligible.append((word_index, span, label))
    return eligible


def _span_means(features, spans):
    """Reduce every span on-device, then move each feature tensor across once.

    The per-span arithmetic is untouched: each entry is still
    ``values[span].float().mean(dim=0)`` over the same sub-token indices in the
    same order, so results are bitwise identical to computing them one at a
    time. Only the host transfer changes. Previously every individual feature
    vector was moved with its own ``.cpu()``, which forces a full device
    synchronization per (word, depth, feature kind) -- on the order of a
    thousand synchronizations per sentence. Now one transfer carries every
    word's vector for a given feature tensor.
    """
    reduced = {}
    for key, values in features.items():
        if not spans:
            reduced[key] = []
            continue
        stacked = torch.stack([values[span].float().mean(dim=0) for span in spans])
        reduced[key] = stacked.cpu().tolist()
    return reduced


def _depth_feature_order(features, depth_rows):
    """Feature keys per depth, preserving the insertion order of `features`.

    Row order is load-bearing: `_evaluate` shuffles the training labels in the
    order they are stored, so the shuffled-label control changes if rows are
    emitted in a different sequence. This reproduces the original
    word -> depth -> (residual, head_0, head_1, ...) ordering while removing the
    quadratic rescan of every feature key for every depth.
    """
    order = {str(depth["relative_label"]): [] for depth in depth_rows}
    for relative_label, feature_kind in features:
        if relative_label in order:
            order[relative_label].append((relative_label, feature_kind))
    return order


def feature_rows(
    model,
    tokenizer,
    examples,
    *,
    labels_by_sentence,
    seed: int,
    progress: float,
    depth_rows,
    role: str,
    features_by_sentence: dict[str, dict] | None = None,
    sentence_batch_size: int = 8,
    maximum_batch_size: int | None = None,
    state_provider=None,
) -> pd.DataFrame:
    timestep = round(progress * 63)
    materialized = []
    for example in examples:
        state = (
            state_provider(example, seed, timestep)
            if state_provider is not None
            else state_at_time(model, tokenizer, example.text, timestep, 64, seed, True)
        )
        materialized.append((example, state))
    precomputed = dict(features_by_sentence or {})
    rows_by_index: dict[int, list[dict]] = {}

    def build_rows(example, state, features):
        current_rows = []
        labels = labels_by_sentence[example.sentence_id]
        eligible = _eligible_words(example, state, labels)
        reduced = _span_means(features, [span for _index, span, _label in eligible])
        per_depth = _depth_feature_order(features, depth_rows)
        for position, (word_index, _span, label) in enumerate(eligible):
            for depth in depth_rows:
                for key in per_depth[str(depth["relative_label"])]:
                    current_rows.append(
                        {
                            "sentence_id": example.sentence_id,
                            "role": role,
                            "seed": seed,
                            "timestep": timestep,
                            "normalized_progress": progress,
                            **depth,
                            "feature_kind": key[1],
                            "word_index": word_index,
                            "form": example.tokens[word_index],
                            "label": label,
                            "feature": reduced[key][position],
                        }
                    )
        return current_rows

    buckets: dict[int, list[tuple[int, Any]]] = {}
    for index, (example, state) in enumerate(materialized):
        if example.sentence_id in precomputed:
            rows_by_index[index] = build_rows(
                example, state, precomputed[example.sentence_id]
            )
        else:
            buckets.setdefault(state.input_ids.shape[1], []).append((index, state))
    for bucket in buckets.values():
        def forward(current):
            return _forward_feature_batch(model, [state for _index, state in current], depth_rows)

        for _start, current, batch_features in adaptive_forward_batches(
            bucket,
            forward,
            initial_size=sentence_batch_size,
            maximum_size=maximum_batch_size,
        ):
            for (index, _state), features in zip(current, batch_features, strict=True):
                example, state = materialized[index]
                rows_by_index[index] = build_rows(example, state, features)
    rows = [row for index in range(len(materialized)) for row in rows_by_index[index]]
    return pd.DataFrame(rows)


class _FitCheckpointStore:
    """Atomic, identity-checked cache of one fitted-and-evaluated logical probe.

    The logical unit is (seed, progress, relative_label, feature_kind): the
    main classifier plus its shuffled-label and random-feature controls,
    fit and evaluated together since they share one training call. Restarting
    one incomplete unit is acceptable; restarting thousands of completed ones
    is not, so this never serializes partially trained optimizer state --
    only the finished evidence frame and metrics.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        scientific_config_hash: str | None,
        manifest_hashes: dict,
        regularization: float,
        label_inventory: list[str],
    ):
        self.directory = run_dir / "fit_checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scientific_config_hash = scientific_config_hash
        self.manifest_hashes = manifest_hashes
        self.regularization = regularization
        self.label_inventory = label_inventory

    def _paths(self, seed: int, progress: float, relative_label: str, feature_kind: str):
        slug = f"seed-{seed}__p-{progress:.6f}__{relative_label}__{feature_kind}"
        path = self.directory / f"{slug}.parquet"
        return path, path.with_suffix(".meta.json")

    def _expected(self, seed: int, progress: float, relative_label: str, feature_kind: str) -> dict:
        return {
            "schema_version": FIT_CHECKPOINT_SCHEMA,
            "scientific_config_hash": self.scientific_config_hash,
            "manifest_hashes": self.manifest_hashes,
            "seed": seed,
            "normalized_progress": progress,
            "relative_label": relative_label,
            "feature_kind": feature_kind,
            "fixed_regularization_c": self.regularization,
            "label_inventory": self.label_inventory,
        }

    def load(self, seed: int, progress: float, relative_label: str, feature_kind: str):
        path, meta_path = self._paths(seed, progress, relative_label, feature_kind)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        expected = self._expected(seed, progress, relative_label, feature_kind)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        try:
            evidence = pd.read_parquet(path)
        except (OSError, ValueError):
            return None
        if metadata.get("row_count") != len(evidence):
            return None
        return evidence, metadata["metrics"]

    def store(
        self,
        seed: int,
        progress: float,
        relative_label: str,
        feature_kind: str,
        evidence: pd.DataFrame,
        metrics: dict,
    ) -> None:
        path, meta_path = self._paths(seed, progress, relative_label, feature_kind)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        evidence.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        atomic_json(
            meta_path,
            {
                **self._expected(seed, progress, relative_label, feature_kind),
                "row_count": len(evidence),
                "metrics": metrics,
            },
        )


def _reuse_t0_from_seed42(
    store: SentenceCheckpointStore, examples, *, stage: str, target_seed: int
) -> pd.DataFrame:
    """Materialize seed 43/44's t=0 feature checkpoints from seed 42's.

    `state_at_time` draws no randomness at timestep 0 (its reveal loop is
    `for progress in range(diffusion_time)`), so the t=0 state -- and every
    feature computed from it -- is bitwise identical across seeds. Existing
    seed 43/44 checkpoints are loaded normally; anything missing is derived
    from the validated seed-42 chunk covering the same sentence range, with
    only its `seed` column rewritten, through the store's normal atomic
    chunk-write path, never a live model forward.
    """
    target_identity = CheckpointIdentity(
        stage=stage, seed=target_seed, normalized_progress=0.0, timestep=0
    )
    source_identity = CheckpointIdentity(stage=stage, seed=42, normalized_progress=0.0, timestep=0)
    sentence_ids = [str(example.sentence_id) for example in examples]
    frames = []
    for start in range(0, len(examples), store.chunk_size):
        end = min(start + store.chunk_size, len(examples))
        chunk_ids = sentence_ids[start:end]
        target_path = store.directory / target_identity.filename(start, end)
        target_expected = store._expected_metadata(examples, target_identity, start, end)
        existing = store._load_chunk(target_path, target_expected, chunk_ids)
        if existing is not None:
            frames.append(existing)
            continue
        source_path = store.directory / source_identity.filename(start, end)
        source_expected = store._expected_metadata(examples, source_identity, start, end)
        source = store._load_chunk(source_path, source_expected, chunk_ids)
        if source is None:
            raise ArtifactError(
                f"seed 42 t=0 checkpoint required to derive seed {target_seed} is missing: "
                f"{source_path.name}"
            )
        derived = source.copy()
        derived["seed"] = target_seed
        store._write_chunk(target_path, derived, target_expected)
        frames.append(derived)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fit(frame: pd.DataFrame, *, seed: int, regularization: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.stack(frame["feature"].map(np.asarray))
    y = frame["label"].to_numpy()
    if len(set(y)) < 2:
        raise ValueError("POS selection features contain fewer than two classes")
    scaler = StandardScaler().fit(x)
    scaled_x = scaler.transform(x)
    classifier = LogisticRegression(
        C=regularization, max_iter=2000, random_state=seed
    ).fit(scaled_x, y)
    return scaler, classifier, x, scaled_x, y


def _evaluate(fitted, train: pd.DataFrame, test: pd.DataFrame, *, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    scaler, classifier, train_x, scaled_train_x, train_y = fitted
    test_x = np.stack(test["feature"].map(np.asarray))
    test_y = test["label"].to_numpy()
    scaled_test_x = scaler.transform(test_x)
    prediction = classifier.predict(scaled_test_x)
    majority = Counter(train_y).most_common(1)[0][0]
    rng = np.random.default_rng(seed)
    shuffled_y = train_y.copy()
    rng.shuffle(shuffled_y)
    shuffled = LogisticRegression(
        C=classifier.C, max_iter=2000, random_state=seed
    ).fit(scaled_train_x, shuffled_y).predict(scaled_test_x)
    random_train = rng.normal(size=train_x.shape)
    random_test = rng.normal(size=test_x.shape)
    random_feature = LogisticRegression(
        C=classifier.C, max_iter=2000, random_state=seed
    ).fit(random_train, train_y).predict(random_test)
    evidence = test.drop(columns="feature").copy()
    evidence["prediction"] = prediction
    evidence["shuffled_prediction"] = shuffled
    evidence["random_feature_prediction"] = random_feature
    evidence["majority_prediction"] = majority
    metrics = {
        "accuracy": accuracy_score(test_y, prediction),
        "macro_f1": f1_score(test_y, prediction, average="macro", zero_division=0),
        "majority_accuracy": accuracy_score(test_y, np.repeat(majority, len(test_y))),
        "shuffled_accuracy": accuracy_score(test_y, shuffled),
        "random_feature_accuracy": accuracy_score(test_y, random_feature),
        "n_positions": len(test_y),
        "class_counts": dict(Counter(test_y)),
    }
    return evidence, metrics


def _evaluate_main_only(fitted, train: pd.DataFrame, test: pd.DataFrame, *, seed: int):
    """Evaluate only the scientific probe used for ranking.

    This is numerically identical to the main-classifier portion of ``_evaluate``.
    Reduced protocols use it for screening and held-out ranking, then run the
    ordinary full logical probe (including both controls) for the final causal
    heads. It never writes a canonical fit checkpoint.
    """
    from sklearn.metrics import accuracy_score, f1_score

    scaler, classifier, _train_x, _scaled_train_x, train_y = fitted
    test_x = np.stack(test["feature"].map(np.asarray))
    test_y = test["label"].to_numpy()
    prediction = classifier.predict(scaler.transform(test_x))
    majority = Counter(train_y).most_common(1)[0][0]
    evidence = test.drop(columns="feature").copy()
    evidence["prediction"] = prediction
    evidence["majority_prediction"] = majority
    metrics = {
        "accuracy": accuracy_score(test_y, prediction),
        "macro_f1": f1_score(test_y, prediction, average="macro", zero_division=0),
        "majority_accuracy": accuracy_score(test_y, np.repeat(majority, len(test_y))),
        "n_positions": len(test_y),
        "class_counts": dict(Counter(test_y)),
    }
    return evidence, metrics


def _fit_evaluate_probe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
    regularization: float,
):
    """Fit one scientifically independent probe without touching shared state."""
    fitted = _fit(train, seed=seed, regularization=regularization)
    return _evaluate(fitted, train, test, seed=seed)


def _fit_evaluate_main_only(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
    regularization: float,
):
    fitted = _fit(train, seed=seed, regularization=regularization)
    return _evaluate_main_only(fitted, train, test, seed=seed)


def _condition_prefix(stage: str, seed: int, progress: float, timestep: int) -> str:
    stage_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stage).strip("-")
    return (
        f"{stage_slug}__seed-{seed}__p-{progress:.6f}__"
        f"t-{timestep}__heads-all__sentences-"
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy one read-only staging file without exposing a partial destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _feature_store_for_condition(
    run_dir: Path,
    *,
    stage: str,
    seed: int,
    progress: float,
    timestep: int,
    cache_root: str | Path | None,
) -> SentenceCheckpointStore:
    """Return the Drive store or a validated local read-only staging store.

    Only the requested condition is copied. Fit checkpoints and final artifacts
    are never redirected: callers continue to write those atomically to
    ``run_dir``. The ordinary checkpoint validator remains authoritative and
    fails closed on a corrupt same-sized local copy.
    """
    source_store = SentenceCheckpointStore(run_dir)
    if cache_root is None:
        return source_store

    scientific_slug = re.sub(
        r"[^A-Za-z0-9_-]+", "-", str(source_store.scientific_config_hash)
    ).strip("-")[:16]
    local_run = Path(cache_root) / f"dlmrel-pos-features-{scientific_slug}"
    local_checkpoints = local_run / "checkpoints"
    prefix = _condition_prefix(stage, seed, progress, timestep)
    source_paths = sorted(source_store.directory.glob(f"{prefix}*"))
    parquet_paths = [path for path in source_paths if path.suffix == ".parquet"]
    if not parquet_paths:
        raise ArtifactError(
            f"no extracted checkpoints for stage={stage!r} seed={seed} progress={progress}"
        )
    required_paths = []
    for parquet in parquet_paths:
        metadata = parquet.with_suffix(".meta.json")
        if not metadata.is_file():
            raise ArtifactError(f"checkpoint chunk missing metadata: {parquet.name}")
        required_paths.extend((parquet, metadata))

    missing_bytes = sum(
        source.stat().st_size
        for source in required_paths
        if not (local_checkpoints / source.name).is_file()
        or (local_checkpoints / source.name).stat().st_size != source.stat().st_size
    )
    local_run.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(local_run).free
    if missing_bytes and free_bytes < int(missing_bytes * 1.15):
        raise ArtifactError(
            "insufficient local disk to stage the active POS feature condition: "
            f"need {missing_bytes * 1.15 / 2**30:.2f} GiB including safety margin, "
            f"have {free_bytes / 2**30:.2f} GiB"
        )

    for name in ("run_metadata.json", "manifest_refs.json"):
        source = run_dir / name
        destination = local_run / name
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            _atomic_copy(source, destination)
    for source in required_paths:
        destination = local_checkpoints / source.name
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            _atomic_copy(source, destination)
    return SentenceCheckpointStore(local_run)


def _load_condition_frames(
    run_dir: Path,
    *,
    seed: int,
    progress: float,
    cache_root: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestep = round(progress * 63)
    frames = []
    for role in ("select", "test"):
        stage = ROLE_STAGE[role]
        store = _feature_store_for_condition(
            run_dir,
            stage=stage,
            seed=seed,
            progress=progress,
            timestep=timestep,
            cache_root=cache_root,
        )
        frames.append(store.require_stage(stage, seed, progress, timestep))
    return frames[0], frames[1]


def _extract(model, tokenizer, cfg: RunConfig, run_dir: Path) -> None:
    """Phase 1 (GPU): compute and checkpoint features for every role/seed/progress.

    Exits without fitting or evaluating anything -- no sklearn classifier is
    ever constructed here, so this can run to completion on a GPU worker that
    then hands off to a CPU-only fit stage.
    """
    settings = cfg.experiment.settings
    jar_path, tagger_model_path = _stanford_paths(settings)
    tagger_identity = stanford_pos_identity(jar_path, tagger_model_path)
    atomic_json(run_dir / "tagger_identity.json", tagger_identity)

    selection, selection_exclusions = load_manifest_examples(cfg, tokenizer, "select")
    if not selection:
        raise ValueError("POS probes have no valid selection examples")
    selection_labels = stanford_labels(selection, settings)
    test, test_exclusions = load_manifest_examples(cfg, tokenizer, "test")
    test_labels = stanford_labels(test, settings)

    depth_mapping_path = run_dir / "relative_depth_mapping.csv"
    if depth_mapping_path.is_file():
        # A prior extract call already ran this probe forward; re-running
        # extract to resume the remaining chunks must not repeat it.
        depths = pd.read_csv(depth_mapping_path).to_dict("records")
    else:
        probe = state_at_time(model, tokenizer, selection[0].text, 0, 64, 42, True)
        _logits, attentions, _hidden = model.forward_attentions(
            probe.input_ids, output_hidden_states=True
        )
        depths = map_relative_depths(len(attentions), settings["relative_depths"])
        pd.DataFrame(depths).to_csv(depth_mapping_path, index=False)

    store = SentenceCheckpointStore(run_dir)
    exclusions = pd.concat([selection_exclusions, test_exclusions], ignore_index=True)
    exclusions.to_parquet(run_dir / "extract_exclusions.parquet", index=False)

    roles = (("select", selection, selection_labels), ("test", test, test_labels))
    for role, examples, labels in roles:
        state_cache = TrajectoryStateCache(
            model,
            tokenizer,
            [round(progress * 63) for progress in cfg.experiment.normalized_progress],
        )

        def cached_state(example, requested_seed, requested_timestep, cache=state_cache):
            return cache.get(
                example.sentence_id,
                example.text,
                requested_seed,
                requested_timestep,
            )

        stage = ROLE_STAGE[role]
        for seed in cfg.experiment.seeds:
            for progress in cfg.experiment.normalized_progress:
                if progress == 0.0 and seed != 42:
                    _reuse_t0_from_seed42(store, examples, stage=stage, target_seed=seed)
                    continue
                identity = CheckpointIdentity(
                    stage=stage,
                    seed=seed,
                    normalized_progress=progress,
                    timestep=round(progress * 63),
                )
                store.run(
                    examples,
                    identity,
                    lambda chunk, _start, current_seed=seed, current_progress=progress, current_labels=labels, current_role=role, current_state_provider=cached_state: feature_rows(  # noqa: E501
                        model,
                        tokenizer,
                        chunk,
                        labels_by_sentence=current_labels,
                        seed=current_seed,
                        progress=current_progress,
                        depth_rows=depths,
                        role=current_role,
                        sentence_batch_size=cfg.runtime.sentence_batch_size,
                        maximum_batch_size=cfg.runtime.adaptive_batch_max_size,
                        state_provider=current_state_provider,
                    ),
                )


def _fit_and_evaluate(cfg: RunConfig, run_dir: Path, *, manifest_hashes: dict) -> dict[str, Any]:
    """Phase 2 (CPU): fit and evaluate every probe from already-extracted features.

    Loads only checkpointed feature parquet files -- no model, no tokenizer,
    no GPU. Fails closed if a required feature checkpoint is missing.
    """
    settings = cfg.experiment.settings
    depths = pd.read_csv(run_dir / "relative_depth_mapping.csv").to_dict("records")
    tagger_identity = json.loads((run_dir / "tagger_identity.json").read_text(encoding="utf-8"))
    scientific_config_hash = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    ).get("scientific_config_hash")
    regularization = float(settings["fixed_regularization_c"])
    label_inventory = list(LABELS)
    fit_store = _FitCheckpointStore(
        run_dir,
        scientific_config_hash=scientific_config_hash,
        manifest_hashes=manifest_hashes,
        regularization=regularization,
        label_inventory=label_inventory,
    )
    evidence_frames, metric_rows = [], []
    selection_sentences: set[str] = set()
    test_sentences: set[str] = set()
    executor = (
        ThreadPoolExecutor(max_workers=cfg.runtime.pos_fit_workers)
        if cfg.runtime.pos_fit_workers > 1
        else None
    )
    try:
        for seed in cfg.experiment.seeds:
            for progress in cfg.experiment.normalized_progress:
                # Load each large feature table exactly once, use it for all 99
                # residual/head fits in this condition, then release it before
                # moving to the next seed/progress pair. The prior eager layout
                # retained all 12 selection tables simultaneously.
                selection_frame, test_frame = _load_condition_frames(
                    run_dir,
                    seed=seed,
                    progress=progress,
                    cache_root=cfg.runtime.pos_feature_cache,
                )
                selection_sentences.update(selection_frame["sentence_id"].astype(str))
                test_sentences.update(test_frame["sentence_id"].astype(str))
                selection_groups = {
                    identity: group
                    for identity, group in selection_frame.groupby(
                        ["relative_label", "feature_kind"], observed=True, sort=True
                    )
                }
                groups = list(
                    test_frame.groupby(
                        ["relative_label", "feature_kind"], observed=True, sort=True
                    )
                )
                resolved: dict = {}
                pending = []
                for identity_values, group in groups:
                    relative_label, feature_kind = identity_values
                    key = (seed, progress, relative_label, feature_kind)
                    cached = fit_store.load(seed, progress, relative_label, feature_kind)
                    if cached is not None:
                        resolved[key] = cached
                        continue
                    arguments = {
                        "seed": seed,
                        "regularization": regularization,
                    }
                    if executor is None:
                        resolved[key] = _fit_evaluate_probe(
                            selection_groups[identity_values], group, **arguments
                        )
                        fit_store.store(*key, *resolved[key])
                    else:
                        pending.append(
                            (
                                key,
                                executor.submit(
                                    _fit_evaluate_probe,
                                    selection_groups[identity_values],
                                    group,
                                    **arguments,
                                ),
                            )
                        )
                # Resolve in canonical group order. Only the parent thread writes
                # checkpoints, so atomic resume behavior is unchanged.
                for key, future in pending:
                    resolved[key] = future.result()
                    fit_store.store(*key, *resolved[key])
                for identity_values, _group in groups:
                    relative_label, feature_kind = identity_values
                    key = (seed, progress, relative_label, feature_kind)
                    evidence, metrics = resolved[key]
                    evidence_frames.append(evidence)
                    metric_rows.append(
                        {
                            "seed": seed,
                            "normalized_progress": progress,
                            "mask_ratio": 1.0 - progress,
                            "relative_label": relative_label,
                            "feature_kind": feature_kind,
                            **metrics,
                        }
                    )
                # Make condition-at-a-time memory release explicit before the
                # next multi-gigabyte table pair is opened.
                del selection_groups, groups, selection_frame, test_frame, resolved
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    raw = pd.concat(evidence_frames, ignore_index=True)
    per_seed = pd.DataFrame(metric_rows)
    exclusions_path = run_dir / "extract_exclusions.parquet"
    exclusions = (
        pd.read_parquet(exclusions_path) if exclusions_path.exists() else pd.DataFrame()
    )
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    group_keys = ["normalized_progress", "mask_ratio", "relative_label", "feature_kind"]
    metrics = per_seed.groupby(group_keys, as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        majority_accuracy=("majority_accuracy", "mean"),
        shuffled_accuracy=("shuffled_accuracy", "mean"),
        random_feature_accuracy=("random_feature_accuracy", "mean"),
        raw_denominator=("n_positions", "sum"),
        n_seeds=("seed", "nunique"),
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    rankings = metrics[metrics["feature_kind"].str.startswith("head_")].sort_values(
        ["relative_label", "normalized_progress", "accuracy_mean"],
        ascending=[True, True, False],
    )
    rankings.to_csv(run_dir / "pos_head_rankings.csv", index=False)
    per_seed[[*group_keys, "seed", "class_counts"]].to_json(
        run_dir / "class_counts.json", orient="records", indent=2
    )
    return {
        "development_used": False,
        "test_tuning_used": False,
        "tagger_backend": "stanford_loglinear_external",
        "tagger_dependency": tagger_identity,
        "tagger_auto_provision_supported": True,
        "historical_release_recovered": False,
        "ud_upos_substituted": False,
        "label_inventory": label_inventory,
        "relative_depths": depths,
        "mask_ratios": settings["mask_ratios"],
        "fixed_regularization_c": regularization,
        "head_level_probes": True,
        "selection_sentences": len(selection_sentences),
        "test_sentences": len(test_sentences),
    }


def _run_fit_stage(cfg: RunConfig, run_dir: Path, *, manifest_hashes: dict) -> dict[str, Any]:
    """Run fitting with one native thread per independently scheduled probe."""
    if cfg.runtime.pos_fit_workers <= 1:
        return _fit_and_evaluate(cfg, run_dir, manifest_hashes=manifest_hashes)
    from threadpoolctl import threadpool_limits

    # sklearn/scipy wheels normally expose an OpenMP pool and one or more BLAS
    # pools. Without this guard, W Python workers each request all C logical
    # CPUs (W*C runnable threads), which is the observed Colab bottleneck.
    with threadpool_limits(limits=1):
        return _fit_and_evaluate(cfg, run_dir, manifest_hashes=manifest_hashes)


def run(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    pos_stage: str = "all",
    manifest_hashes: dict | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Dispatch to the requested POS stage.

    'extract' (GPU) computes and checkpoints features, then returns without
    fitting anything. 'fit' (CPU, no model) loads only those checkpoints and
    fits/evaluates every probe. 'all' runs both in sequence and is exactly
    equivalent to 'fit' immediately following 'extract' -- it is implemented
    as that same sequence, not a separate code path, so the two can never
    silently diverge.
    """
    manifest_hashes = manifest_hashes or {}
    if pos_stage == "extract":
        _extract(model, tokenizer, cfg, run_dir)
        return {"pos_stage": "extract", "extract_complete": True}
    if pos_stage == "fit":
        return {"pos_stage": "fit", **_run_fit_stage(cfg, run_dir, manifest_hashes=manifest_hashes)}
    _extract(model, tokenizer, cfg, run_dir)
    return {"pos_stage": "all", **_run_fit_stage(cfg, run_dir, manifest_hashes=manifest_hashes)}

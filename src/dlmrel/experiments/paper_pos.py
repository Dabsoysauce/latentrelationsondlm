"""Old multi-depth/multi-mask POS/token-class linear probes without development tuning."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..artifacts import ArtifactError, atomic_json
from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import state_at_time
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


def _forward_features(model, state, depth_rows):
    layers = [int(row["actual_layer_index"]) for row in depth_rows]
    with capture_projection_inputs(model, layers) as (captures, metadata):
        if hasattr(model, "forward_features"):
            _attentions, hidden_states = model.forward_features(state.input_ids)
        else:
            _logits, _attentions, hidden_states = model.forward_attentions(
                state.input_ids, output_hidden_states=True
            )
    output = {}
    for depth in depth_rows:
        layer = int(depth["actual_layer_index"])
        hidden_index = min(layer + 1, len(hidden_states) - 1)
        output[(depth["relative_label"], "residual")] = hidden_states[hidden_index][0]
        values = captures[layer]
        if len(values) != 1:
            raise RuntimeError("attention output projection did not execute exactly once")
        concatenated = values[0][0]
        heads = metadata[layer].number_of_heads
        if concatenated.shape[-1] % heads:
            raise RuntimeError("captured attention width is not divisible into heads")
        width = concatenated.shape[-1] // heads
        for head in range(heads):
            output[(depth["relative_label"], f"head_{head}")] = concatenated[
                :, head * width : (head + 1) * width
            ]
    return output


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
) -> pd.DataFrame:
    rows = []
    timestep = round(progress * 63)
    for example in examples:
        state = state_at_time(
            model, tokenizer, example.text, timestep, 64, seed, True
        )
        features = _forward_features(model, state, depth_rows)
        labels = labels_by_sentence[example.sentence_id]
        eligible = _eligible_words(example, state, labels)
        reduced = _span_means(features, [span for _index, span, _label in eligible])
        per_depth = _depth_feature_order(features, depth_rows)
        for position, (word_index, _span, label) in enumerate(eligible):
            for depth in depth_rows:
                for key in per_depth[str(depth["relative_label"])]:
                    rows.append(
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
        mirror_directory: str | Path | None = None,
    ):
        self.directory = run_dir / "fit_checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mirror_directory = Path(mirror_directory) if mirror_directory else None
        if self.mirror_directory is not None:
            self.mirror_directory.mkdir(parents=True, exist_ok=True)
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
        cached = self._load_paths(
            path, meta_path, seed, progress, relative_label, feature_kind
        )
        if cached is not None or self.mirror_directory is None:
            return cached
        mirror_path = self.mirror_directory / path.name
        return self._load_paths(
            mirror_path,
            mirror_path.with_suffix(".meta.json"),
            seed,
            progress,
            relative_label,
            feature_kind,
        )

    def _load_paths(
        self,
        path: Path,
        meta_path: Path,
        seed: int,
        progress: float,
        relative_label: str,
        feature_kind: str,
    ):
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
        if self.mirror_directory is not None:
            mirror_path = self.mirror_directory / path.name
            mirror_temporary = mirror_path.with_suffix(
                mirror_path.suffix + f".tmp-{os.getpid()}"
            )
            mirror_temporary.unlink(missing_ok=True)
            shutil.copyfile(path, mirror_temporary)
            os.replace(mirror_temporary, mirror_path)
            atomic_json(
                mirror_path.with_suffix(".meta.json"),
                json.loads(meta_path.read_text(encoding="utf-8")),
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
    scaled_test_x = scaler.transform(test_x)
    test_y = test["label"].to_numpy()
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
                    lambda chunk, _start, current_seed=seed, current_progress=progress, current_labels=labels, current_role=role: feature_rows(  # noqa: E501
                        model,
                        tokenizer,
                        chunk,
                        labels_by_sentence=current_labels,
                        seed=current_seed,
                        progress=current_progress,
                        depth_rows=depths,
                        role=current_role,
                    ),
                )


def _fit_store(
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict,
    mirror_directory: str | Path | None = None,
) -> tuple[_FitCheckpointStore, float, list[str]]:
    settings = cfg.experiment.settings
    scientific_config_hash = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    ).get("scientific_config_hash")
    regularization = float(settings["fixed_regularization_c"])
    label_inventory = list(LABELS)
    return (
        _FitCheckpointStore(
            run_dir,
            scientific_config_hash=scientific_config_hash,
            manifest_hashes=manifest_hashes,
            regularization=regularization,
            label_inventory=label_inventory,
            mirror_directory=mirror_directory,
        ),
        regularization,
        label_inventory,
    )


def _probe_keys(cfg: RunConfig, checkpoint_store: SentenceCheckpointStore) -> list[tuple]:
    """Return all logical probes in a stable order shared by every worker."""
    keys = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            frame = checkpoint_store.require_stage(
                ROLE_STAGE["test"],
                seed,
                progress,
                round(progress * 63),
                columns=["relative_label", "feature_kind"],
            )
            identities = frame[["relative_label", "feature_kind"]].drop_duplicates()
            keys.extend(
                (seed, progress, str(row.relative_label), str(row.feature_kind))
                for row in identities.itertuples(index=False)
            )
    return sorted(keys)


def _assigned_probe_keys(
    keys: list[tuple], *, shard_count: int, shard_index: int
) -> list[tuple]:
    """Balance logical probes while keeping equivalent t=0 main fits together."""
    units: dict[tuple, list[tuple]] = {}
    for key in keys:
        seed, progress, relative_label, feature_kind = key
        unit = (
            ("shared-t0", relative_label, feature_kind)
            if progress == 0.0
            else ("seeded", seed, progress, relative_label, feature_kind)
        )
        units.setdefault(unit, []).append(key)
    buckets: list[list[tuple]] = [[] for _ in range(shard_count)]
    for unit in sorted(units):
        target = min(range(shard_count), key=lambda index: (len(buckets[index]), index))
        buckets[target].extend(sorted(units[unit]))
    return sorted(buckets[shard_index])


def _fit_shard(
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict,
    shard_count: int,
    shard_index: int,
    mirror_directory: str | Path | None,
) -> dict[str, Any]:
    """Fit one deterministic subset while keeping only one feature pair in memory."""
    fit_store, regularization, _label_inventory = _fit_store(
        cfg,
        run_dir,
        manifest_hashes=manifest_hashes,
        mirror_directory=mirror_directory,
    )
    checkpoint_store = SentenceCheckpointStore(run_dir)
    all_keys = _probe_keys(cfg, checkpoint_store)
    assigned_keys = _assigned_probe_keys(
        all_keys, shard_count=shard_count, shard_index=shard_index
    )
    completed = reused = fitted_now = reused_main_fits = 0
    main_fit_cache: dict[tuple, Any] = {}

    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            pair_keys = [key for key in assigned_keys if key[:2] == (seed, progress)]
            if not pair_keys:
                continue
            filters = [
                [
                    ("relative_label", "==", relative_label),
                    ("feature_kind", "==", feature_kind),
                ]
                for _seed, _progress, relative_label, feature_kind in pair_keys
            ]
            selection = checkpoint_store.require_stage(
                ROLE_STAGE["select"],
                seed,
                progress,
                round(progress * 63),
                filters=filters,
            )
            test = checkpoint_store.require_stage(
                ROLE_STAGE["test"],
                seed,
                progress,
                round(progress * 63),
                filters=filters,
            )
            selection_groups = {
                tuple(map(str, identity)): group
                for identity, group in selection.groupby(
                    ["relative_label", "feature_kind"], observed=True
                )
            }
            test_groups = {
                tuple(map(str, identity)): group
                for identity, group in test.groupby(
                    ["relative_label", "feature_kind"], observed=True
                )
            }
            for key in pair_keys:
                relative_label, feature_kind = key[2:]
                identity = (relative_label, feature_kind)
                if identity not in selection_groups or identity not in test_groups:
                    raise ArtifactError(f"feature checkpoints do not contain logical probe {key}")
                cached = fit_store.load(*key)
                if cached is None:
                    main_key = (progress, relative_label, feature_kind)
                    fitted = main_fit_cache.get(main_key) if progress == 0.0 else None
                    if fitted is None:
                        fitted = _fit(
                            selection_groups[identity], seed=seed, regularization=regularization
                        )
                        if progress == 0.0 and fitted[1].solver == "lbfgs":
                            main_fit_cache[main_key] = fitted
                    else:
                        reused_main_fits += 1
                    evidence, metrics = _evaluate(
                        fitted, selection_groups[identity], test_groups[identity], seed=seed
                    )
                    fit_store.store(*key, evidence, metrics)
                    fitted_now += 1
                else:
                    reused += 1
                completed += 1

    if completed != len(assigned_keys):
        raise ArtifactError(
            f"POS fit shard {shard_index}/{shard_count} completed {completed} of "
            f"{len(assigned_keys)} assigned probes"
        )
    status = {
        "schema_version": "dlmrel-pos-fit-shard-v1",
        "pos_fit_partial": shard_count > 1,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "total_logical_probes": len(all_keys),
        "assigned_logical_probes": len(assigned_keys),
        "completed_logical_probes": completed,
        "reused_logical_probes": reused,
        "fitted_logical_probes": fitted_now,
        "reused_t0_main_fits": reused_main_fits,
    }
    atomic_json(
        run_dir / f"pos_fit_shard_status-{shard_index:03d}-of-{shard_count:03d}.json",
        status,
    )
    return status


def _aggregate_fit_checkpoints(
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict,
    mirror_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every logical probe checkpoint, then write final artifacts."""
    settings = cfg.experiment.settings
    depths = pd.read_csv(run_dir / "relative_depth_mapping.csv").to_dict("records")
    tagger_identity = json.loads((run_dir / "tagger_identity.json").read_text(encoding="utf-8"))
    fit_store, regularization, label_inventory = _fit_store(
        cfg,
        run_dir,
        manifest_hashes=manifest_hashes,
        mirror_directory=mirror_directory,
    )
    checkpoint_store = SentenceCheckpointStore(run_dir)
    keys = _probe_keys(cfg, checkpoint_store)
    evidence_frames, metric_rows, missing = [], [], []
    for seed, progress, relative_label, feature_kind in keys:
        cached = fit_store.load(seed, progress, relative_label, feature_kind)
        if cached is None:
            missing.append((seed, progress, relative_label, feature_kind))
            continue
        evidence, metrics = cached
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
    if missing:
        preview = ", ".join(map(str, missing[:3]))
        raise ArtifactError(
            f"cannot aggregate POS fit: {len(missing)} of {len(keys)} logical-probe "
            f"checkpoints are missing or invalid; first missing: {preview}"
        )

    selection_sentences: set[str] = set()
    test_sentences: set[str] = set()
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            selection = checkpoint_store.require_stage(
                ROLE_STAGE["select"],
                seed,
                progress,
                round(progress * 63),
                columns=["sentence_id"],
            )
            test = checkpoint_store.require_stage(
                ROLE_STAGE["test"],
                seed,
                progress,
                round(progress * 63),
                columns=["sentence_id"],
            )
            selection_sentences.update(selection["sentence_id"].astype(str))
            test_sentences.update(test["sentence_id"].astype(str))
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
        "logical_probes": len(keys),
    }


def _fit_and_evaluate(
    cfg: RunConfig,
    run_dir: Path,
    *,
    manifest_hashes: dict,
    shard_count: int,
    shard_index: int,
    aggregate_only: bool,
    mirror_directory: str | Path | None,
) -> dict[str, Any]:
    """Phase 2 (CPU): fit a shard or aggregate completed probe checkpoints."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("fit shard index must be within a positive shard count")
    if aggregate_only:
        return _aggregate_fit_checkpoints(
            cfg,
            run_dir,
            manifest_hashes=manifest_hashes,
            mirror_directory=mirror_directory,
        )
    status = _fit_shard(
        cfg,
        run_dir,
        manifest_hashes=manifest_hashes,
        shard_count=shard_count,
        shard_index=shard_index,
        mirror_directory=mirror_directory,
    )
    if shard_count > 1:
        return status
    return {
        **status,
        **_aggregate_fit_checkpoints(
            cfg,
            run_dir,
            manifest_hashes=manifest_hashes,
            mirror_directory=mirror_directory,
        ),
    }


def run(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    pos_stage: str = "all",
    fit_shard_count: int = 1,
    fit_shard_index: int = 0,
    fit_aggregate_only: bool = False,
    fit_checkpoint_mirror: str | Path | None = None,
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
        return {
            "pos_stage": "fit",
            **_fit_and_evaluate(
                cfg,
                run_dir,
                manifest_hashes=manifest_hashes,
                shard_count=fit_shard_count,
                shard_index=fit_shard_index,
                aggregate_only=fit_aggregate_only,
                mirror_directory=fit_checkpoint_mirror,
            ),
        }
    _extract(model, tokenizer, cfg, run_dir)
    return {
        "pos_stage": "all",
        **_fit_and_evaluate(
            cfg,
            run_dir,
            manifest_hashes=manifest_hashes,
            shard_count=1,
            shard_index=0,
            aggregate_only=False,
            mirror_directory=None,
        ),
    }

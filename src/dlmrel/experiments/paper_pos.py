"""Old multi-depth/multi-mask POS/token-class linear probes without development tuning."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import state_at_time
from ..models.decomposition import capture_projection_inputs
from ..paper_protocol import map_relative_depths
from .shared import write_frames

LABELS = ("NOUN", "VERB", "ADJ", "ADV", "PREP", "DET", "PRON", "CONJ")


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
    if not jar or not model:
        raise RuntimeError(
            "Exact POS replication is blocked: set STANFORD_POS_TAGGER_JAR and "
            "STANFORD_POS_TAGGER_MODEL. UD UPOS is intentionally not substituted."
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
) -> pd.DataFrame:
    rows = []
    timestep = round(progress * 63)
    for example in examples:
        state = state_at_time(
            model, tokenizer, example.text, timestep, 64, seed, True
        )
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


def _fit(frame: pd.DataFrame, *, seed: int, regularization: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.stack(frame["feature"].map(np.asarray))
    y = frame["label"].to_numpy()
    if len(set(y)) < 2:
        raise ValueError("POS selection features contain fewer than two classes")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(
        C=regularization, max_iter=2000, random_state=seed
    ).fit(scaler.transform(x), y)
    return scaler, classifier, x, y


def _evaluate(fitted, train: pd.DataFrame, test: pd.DataFrame, *, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    scaler, classifier, train_x, train_y = fitted
    test_x = np.stack(test["feature"].map(np.asarray))
    test_y = test["label"].to_numpy()
    prediction = classifier.predict(scaler.transform(test_x))
    majority = Counter(train_y).most_common(1)[0][0]
    rng = np.random.default_rng(seed)
    shuffled_y = train_y.copy()
    rng.shuffle(shuffled_y)
    shuffled = LogisticRegression(
        C=classifier.C, max_iter=2000, random_state=seed
    ).fit(scaler.transform(train_x), shuffled_y).predict(scaler.transform(test_x))
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


def run(model, tokenizer, cfg: RunConfig, run_dir: Path, **_unused: Any) -> dict[str, Any]:
    settings = cfg.experiment.settings
    _stanford_paths(settings)  # fail before touching any held-out data
    selection, selection_exclusions = load_manifest_examples(cfg, tokenizer, "select")
    selection_labels = stanford_labels(selection, settings)
    if not selection:
        raise ValueError("POS probes have no valid selection examples")
    probe = state_at_time(model, tokenizer, selection[0].text, 0, 64, 42, True)
    _logits, attentions, _hidden = model.forward_attentions(
        probe.input_ids, output_hidden_states=True
    )
    depths = map_relative_depths(len(attentions), settings["relative_depths"])
    pd.DataFrame(depths).to_csv(run_dir / "relative_depth_mapping.csv", index=False)
    store = SentenceCheckpointStore(run_dir)
    frozen = {}
    selection_frames = {}
    regularization = float(settings["fixed_regularization_c"])
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            identity = CheckpointIdentity(
                stage="paper-pos-selection-features",
                seed=seed,
                normalized_progress=progress,
                timestep=round(progress * 63),
            )
            frame = store.run(
                selection,
                identity,
                lambda chunk, _start, current_seed=seed, current_progress=progress: feature_rows(
                    model,
                    tokenizer,
                    chunk,
                    labels_by_sentence=selection_labels,
                    seed=current_seed,
                    progress=current_progress,
                    depth_rows=depths,
                    role="select",
                ),
            )
            for identity_values, group in frame.groupby(
                ["relative_label", "feature_kind"], observed=True
            ):
                key = (seed, progress, *identity_values)
                selection_frames[key] = group
                frozen[key] = _fit(group, seed=seed, regularization=regularization)

    # No test manifest, labels, or features are opened before every probe is frozen.
    test, test_exclusions = load_manifest_examples(cfg, tokenizer, "test")
    test_labels = stanford_labels(test, settings)
    evidence_frames, metric_rows = [], []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            identity = CheckpointIdentity(
                stage="paper-pos-test-features",
                seed=seed,
                normalized_progress=progress,
                timestep=round(progress * 63),
            )
            frame = store.run(
                test,
                identity,
                lambda chunk, _start, current_seed=seed, current_progress=progress: feature_rows(
                    model,
                    tokenizer,
                    chunk,
                    labels_by_sentence=test_labels,
                    seed=current_seed,
                    progress=current_progress,
                    depth_rows=depths,
                    role="test",
                ),
            )
            for identity_values, group in frame.groupby(
                ["relative_label", "feature_kind"], observed=True
            ):
                key = (seed, progress, *identity_values)
                evidence, metrics = _evaluate(
                    frozen[key], selection_frames[key], group, seed=seed
                )
                evidence_frames.append(evidence)
                metric_rows.append(
                    {
                        "seed": seed,
                        "normalized_progress": progress,
                        "mask_ratio": 1.0 - progress,
                        "relative_label": identity_values[0],
                        "feature_kind": identity_values[1],
                        **metrics,
                    }
                )
    raw = pd.concat(evidence_frames, ignore_index=True)
    per_seed = pd.DataFrame(metric_rows)
    exclusions = pd.concat([selection_exclusions, test_exclusions], ignore_index=True)
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
        "ud_upos_substituted": False,
        "label_inventory": list(LABELS),
        "relative_depths": depths,
        "mask_ratios": settings["mask_ratios"],
        "fixed_regularization_c": regularization,
        "head_level_probes": True,
        "selection_sentences": len(selection),
        "test_sentences": len(test),
    }

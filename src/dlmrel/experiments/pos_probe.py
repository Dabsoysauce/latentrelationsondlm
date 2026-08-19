"""Masked POS probe with select fit, dev hyperparameter choice, and frozen test."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import states_at_time
from ..relations import Example
from .shared import write_frames

REGULARIZATION_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class FittedProbe:
    scaler: Any
    classifier: Any
    selected_c: float
    train_x: np.ndarray
    train_y: np.ndarray
    train_forms: np.ndarray


def run(model, tokenizer, cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    exclusions = []
    checkpoint_store = SentenceCheckpointStore(run_dir)
    examples_by_role = {}
    for role in ("select", "dev"):
        examples, dropped = load_manifest_examples(cfg, tokenizer, role)
        exclusions.append(dropped)
        examples_by_role[role] = examples

    fitted = {}
    for seed in cfg.experiment.seeds:
        select = masked_features(
            model,
            tokenizer,
            examples_by_role["select"],
            cfg,
            seed=seed,
            role="select",
            checkpoint_store=checkpoint_store,
        )
        dev = masked_features(
            model,
            tokenizer,
            examples_by_role["dev"],
            cfg,
            seed=seed,
            role="dev",
            checkpoint_store=checkpoint_store,
        )
        fitted[seed] = fit_probe(select, dev, seed)

    # No test manifest, features, labels, or forms are read until every seed's
    # regularization choice has been frozen from select/dev alone.
    test_examples, test_dropped = load_manifest_examples(cfg, tokenizer, "test")
    exclusions.append(test_dropped)
    raw_frames = []
    seed_metrics = []
    for seed in cfg.experiment.seeds:
        test = masked_features(
            model,
            tokenizer,
            test_examples,
            cfg,
            seed=seed,
            role="test",
            checkpoint_store=checkpoint_store,
        )
        raw, metrics = evaluate_fitted_probe(fitted[seed], test, seed)
        raw_frames.append(raw)
        seed_metrics.append(metrics)

    raw = pd.concat(raw_frames, ignore_index=True)
    per_seed = pd.DataFrame(seed_metrics)
    write_frames(run_dir, raw=raw, exclusions=pd.concat(exclusions, ignore_index=True))
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    aggregate_probe_metrics(per_seed).to_csv(run_dir / "metrics.csv", index=False)
    return {
        "selected_c_by_seed": {
            str(int(row.seed)): float(row.selected_c) for row in per_seed.itertuples(index=False)
        },
        "n_test_positions": int(per_seed["n_test_positions"].sum()),
        "n_seeds": int(per_seed["seed"].nunique()),
    }


def fit_probe(select, dev, seed: int) -> FittedProbe:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    train_x, train_y, _, train_forms, _ = _validated_features(select, "select")
    dev_x, dev_y, _, _, _ = _validated_features(dev, "dev")
    if len(np.unique(train_y)) < 2:
        raise ValueError("POS select features must contain at least two classes")
    if train_x.shape[1] != dev_x.shape[1]:
        raise ValueError("POS select and dev feature dimensions differ")
    scaler = StandardScaler().fit(train_x)

    candidates = []
    for regularization in REGULARIZATION_GRID:
        classifier = LogisticRegression(C=regularization, max_iter=2000, random_state=seed)
        classifier.fit(scaler.transform(train_x), train_y)
        accuracy = accuracy_score(dev_y, classifier.predict(scaler.transform(dev_x)))
        candidates.append((accuracy, regularization))
    _, selected_c = max(candidates, key=lambda item: (item[0], -item[1]))

    classifier = LogisticRegression(C=selected_c, max_iter=2000, random_state=seed)
    classifier.fit(scaler.transform(train_x), train_y)
    return FittedProbe(
        scaler=scaler,
        classifier=classifier,
        selected_c=float(selected_c),
        train_x=train_x.copy(),
        train_y=train_y.copy(),
        train_forms=train_forms.copy(),
    )


def evaluate_fitted_probe(
    fitted: FittedProbe, test, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    test_x, test_y, test_groups, test_forms, test_word_indices = _validated_features(
        test, "test"
    )
    if test_x.shape[1] != fitted.scaler.n_features_in_:
        raise ValueError("POS test feature dimension differs from select")
    train_y = fitted.train_y
    train_x = fitted.train_x
    train_forms = fitted.train_forms
    selected_c = fitted.selected_c
    scaler = fitted.scaler
    classifier = fitted.classifier
    prediction = classifier.predict(scaler.transform(test_x))
    majority = Counter(train_y).most_common(1)[0][0]
    lexical = lexical_predictions(train_forms, train_y, test_forms, majority)

    rng = np.random.default_rng(seed)
    shuffled_y = train_y.copy()
    rng.shuffle(shuffled_y)
    shuffled = (
        LogisticRegression(C=selected_c, max_iter=2000, random_state=seed)
        .fit(scaler.transform(train_x), shuffled_y)
        .predict(scaler.transform(test_x))
    )
    random_feature = (
        LogisticRegression(C=selected_c, max_iter=2000, random_state=seed)
        .fit(rng.normal(size=(len(train_y), test_x.shape[1])), train_y)
        .predict(rng.normal(size=test_x.shape))
    )
    raw = pd.DataFrame(
        {
            "seed": seed,
            "sentence_id": test_groups,
            "word_index": test_word_indices,
            "form": test_forms,
            "gold_upos": test_y,
            "prediction": prediction,
            "lexical_prediction": lexical,
            "shuffled_prediction": shuffled,
            "random_feature_prediction": random_feature,
        }
    )
    metrics = {
        "seed": seed,
        "selected_c": selected_c,
        "accuracy": accuracy_score(test_y, prediction),
        "macro_f1": f1_score(test_y, prediction, average="macro", zero_division=0),
        "majority_accuracy": accuracy_score(test_y, np.repeat(majority, len(test_y))),
        "lexical_accuracy": accuracy_score(test_y, lexical),
        "shuffled_accuracy": accuracy_score(test_y, shuffled),
        "random_feature_accuracy": accuracy_score(test_y, random_feature),
        "n_test_positions": len(test_y),
        "n_test_sentences": len(set(test_groups)),
    }
    return raw, metrics


def evaluate_seed(collected, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compatibility wrapper for direct select/dev/test unit evaluation."""
    fitted = fit_probe(collected["select"], collected["dev"], seed)
    return evaluate_fitted_probe(fitted, collected["test"], seed)


def _validated_features(features, role: str):
    try:
        x, y, groups, forms, word_indices = features
    except (TypeError, ValueError) as error:
        raise ValueError(f"POS {role} features must be a five-array tuple") from error
    x = np.asarray(x)
    y = np.asarray(y)
    groups = np.asarray(groups)
    forms = np.asarray(forms)
    word_indices = np.asarray(word_indices)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"POS {role} features must be a nonempty 2D matrix")
    if not (len(x) == len(y) == len(groups) == len(forms) == len(word_indices)):
        raise ValueError(f"POS {role} feature arrays have inconsistent lengths")
    if not np.isfinite(x.astype(float)).all():
        raise ValueError(f"POS {role} features contain non-finite values")
    return x, y, groups, forms, word_indices


def aggregate_probe_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        "accuracy",
        "macro_f1",
        "majority_accuracy",
        "lexical_accuracy",
        "shuffled_accuracy",
        "random_feature_accuracy",
    ]
    summary: dict[str, Any] = {"n_seeds": int(per_seed["seed"].nunique())}
    for column in score_columns:
        summary[column] = float(per_seed[column].mean())
        summary[f"{column}_seed_std"] = float(per_seed[column].std())
    summary["n_test_positions"] = int(per_seed["n_test_positions"].sum())
    summary["n_test_sentences"] = int(per_seed["n_test_sentences"].max())
    return pd.DataFrame([summary])


def masked_features(
    model,
    tokenizer,
    examples: list[Example],
    cfg: RunConfig,
    *,
    seed: int,
    role: str | None = None,
    checkpoint_store: SentenceCheckpointStore | None = None,
):
    timestep = round(cfg.experiment.normalized_progress[0] * (cfg.experiment.steps - 1))
    if checkpoint_store is None:
        frame = masked_feature_rows(model, tokenizer, examples, cfg, seed=seed)
    else:
        if role is None:
            raise ValueError("checkpointed POS features require a dataset role")
        identity = CheckpointIdentity(
            stage=f"pos-probe-{role}-features",
            seed=seed,
            normalized_progress=cfg.experiment.normalized_progress[0],
            timestep=timestep,
        )
        frame = checkpoint_store.run(
            examples,
            identity,
            lambda chunk, _start: masked_feature_rows(
                model, tokenizer, chunk, cfg, seed=seed
            ),
        )
    if frame.empty:
        return (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
        )
    return (
        np.stack(frame["feature"].map(np.asarray)),
        frame["label"].to_numpy(),
        frame["sentence_id"].to_numpy(),
        frame["form"].to_numpy(),
        frame["word_index"].to_numpy(),
    )


def masked_feature_rows(model, tokenizer, examples, cfg: RunConfig, *, seed: int):
    rows = []
    timestep = round(cfg.experiment.normalized_progress[0] * (cfg.experiment.steps - 1))
    for example in examples:
        _, hidden_states, state = states_at_time(
            model,
            tokenizer,
            example.text,
            timestep,
            cfg.experiment.steps,
            seed,
            True,
        )
        hidden = hidden_states[len(hidden_states) // 2][0].float().cpu().numpy()
        for word_index, span in example.word_to_tokens.items():
            if not span or max(span) >= len(state.is_visible):
                continue
            if any(state.is_visible[position] for position in span):
                continue
            rows.append(
                {
                    "sentence_id": example.sentence_id,
                    "word_index": word_index,
                    "form": example.tokens[word_index],
                    "label": example.upos[word_index],
                    "feature": hidden[span].mean(axis=0).tolist(),
                }
            )
    return pd.DataFrame(rows)


def lexical_predictions(train_forms, train_labels, test_forms, fallback):
    counts: dict[str, Counter] = defaultdict(Counter)
    for form, label in zip(train_forms, train_labels, strict=True):
        counts[str(form)][str(label)] += 1
    table = {form: values.most_common(1)[0][0] for form, values in counts.items()}
    return np.asarray([table.get(str(form), fallback) for form in test_forms])

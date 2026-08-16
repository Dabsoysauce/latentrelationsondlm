"""Masked POS probe with select fit, dev hyperparameter choice, and frozen test."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import states_at_time
from ..relations import Example
from .shared import write_frames


def run(model, tokenizer, cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    collected = {}
    exclusions = []
    for role in ("select", "dev", "test"):
        examples, dropped = load_manifest_examples(cfg, tokenizer, role)
        exclusions.append(dropped)
        collected[role] = masked_features(model, tokenizer, examples, cfg)
    train_x, train_y, _, train_forms = collected["select"]
    dev_x, dev_y, _, _ = collected["dev"]
    test_x, test_y, test_groups, test_forms = collected["test"]
    scaler = StandardScaler().fit(train_x)

    candidates = []
    for regularization in (0.01, 0.1, 1.0, 10.0):
        classifier = LogisticRegression(C=regularization, max_iter=2000, random_state=42)
        classifier.fit(scaler.transform(train_x), train_y)
        accuracy = accuracy_score(dev_y, classifier.predict(scaler.transform(dev_x)))
        candidates.append((accuracy, regularization))
    _, selected_c = max(candidates, key=lambda item: (item[0], -item[1]))

    classifier = LogisticRegression(C=selected_c, max_iter=2000, random_state=42)
    classifier.fit(scaler.transform(train_x), train_y)
    prediction = classifier.predict(scaler.transform(test_x))
    majority = Counter(train_y).most_common(1)[0][0]
    lexical = lexical_predictions(train_forms, train_y, test_forms, majority)

    rng = np.random.default_rng(42)
    shuffled_y = train_y.copy()
    rng.shuffle(shuffled_y)
    shuffled = (
        LogisticRegression(C=selected_c, max_iter=2000, random_state=42)
        .fit(scaler.transform(train_x), shuffled_y)
        .predict(scaler.transform(test_x))
    )
    random_feature = (
        LogisticRegression(C=selected_c, max_iter=2000, random_state=42)
        .fit(rng.normal(size=train_x.shape), train_y)
        .predict(rng.normal(size=test_x.shape))
    )
    raw = pd.DataFrame(
        {
            "sentence_id": test_groups,
            "gold_upos": test_y,
            "prediction": prediction,
            "lexical_prediction": lexical,
            "shuffled_prediction": shuffled,
            "random_feature_prediction": random_feature,
        }
    )
    write_frames(run_dir, raw=raw, exclusions=pd.concat(exclusions, ignore_index=True))
    metrics = pd.DataFrame(
        [
            {
                "selected_c": selected_c,
                "accuracy": accuracy_score(test_y, prediction),
                "macro_f1": f1_score(test_y, prediction, average="macro"),
                "majority_accuracy": accuracy_score(test_y, np.repeat(majority, len(test_y))),
                "lexical_accuracy": accuracy_score(test_y, lexical),
                "shuffled_accuracy": accuracy_score(test_y, shuffled),
                "random_feature_accuracy": accuracy_score(test_y, random_feature),
                "n_test_positions": len(test_y),
                "n_test_sentences": len(set(test_groups)),
            }
        ]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    metrics.assign(seed=cfg.experiment.seeds[0]).to_csv(run_dir / "per_seed_metrics.csv", index=False)
    return {"selected_c": selected_c, "n_test_positions": len(test_y)}


def masked_features(model, tokenizer, examples: list[Example], cfg: RunConfig):
    features, labels, groups, forms = [], [], [], []
    timestep = round(cfg.experiment.normalized_progress[0] * (cfg.experiment.steps - 1))
    for example in examples:
        _, hidden_states, state = states_at_time(
            model,
            tokenizer,
            example.text,
            timestep,
            cfg.experiment.steps,
            cfg.experiment.seeds[0],
            True,
        )
        hidden = hidden_states[len(hidden_states) // 2][0].float().cpu().numpy()
        for word_index, span in example.word_to_tokens.items():
            if not span or max(span) >= len(state.is_visible):
                continue
            if any(state.is_visible[position] for position in span):
                continue
            features.append(hidden[span].mean(axis=0))
            labels.append(example.upos[word_index])
            groups.append(example.sentence_id)
            forms.append(example.tokens[word_index])
    return np.asarray(features), np.asarray(labels), np.asarray(groups), np.asarray(forms)


def lexical_predictions(train_forms, train_labels, test_forms, fallback):
    counts: dict[str, Counter] = defaultdict(Counter)
    for form, label in zip(train_forms, train_labels, strict=True):
        counts[str(form)][str(label)] += 1
    table = {form: values.most_common(1)[0][0] for form, values in counts.items()}
    return np.asarray([table.get(str(form), fallback) for form in test_forms])

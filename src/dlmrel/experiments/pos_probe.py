from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, DiffusionConfig
from ..data import examples_for_split
from ..diffusion import states_at_time
from ..relations import Example


def _token_upos(example: Example) -> dict[int, str]:
    labels = {}
    for word_idx, token_idx in example.word_to_tokens.items():
        if word_idx < len(example.upos) and token_idx:
            tag = example.upos[word_idx]
            if tag:
                labels[token_idx[-1]] = tag
    return labels


def collect_probe_features(
    model,
    tokenizer,
    examples: list[Example],
    cfg: DiffusionConfig,
    layers: list[int],
    diffusion_time: int | None = None,
    log_every: int = 100,
):
    t = cfg.steps - 1 if diffusion_time is None else diffusion_time
    feats: dict[int, list[np.ndarray]] = {d: [] for d in layers}
    labels: list[str] = []
    forms: list[str] = []

    for i, example in enumerate(examples):
        _, hidden_states, state = states_at_time(
            model,
            tokenizer,
            example.text,
            diffusion_time=t,
            steps=cfg.steps,
            seed=cfg.seed,
            include_bos=cfg.include_bos,
        )
        seq_len = len(state.tokens)
        tagged = {p: tag for p, tag in _token_upos(example).items() if p < seq_len}
        if not tagged:
            continue

        positions = sorted(tagged)
        for depth in layers:
            h = hidden_states[depth][0].float().cpu().numpy()
            feats[depth].append(h[positions])
        labels.extend(tagged[p] for p in positions)
        forms.extend(state.tokens[p] for p in positions)

        if log_every and (i + 1) % log_every == 0:
            print(f"[pos-probe] {i + 1}/{len(examples)} sentences", flush=True)

    stacked = {d: np.concatenate(v, axis=0) for d, v in feats.items() if v}
    return stacked, np.array(labels), np.array(forms)


def most_frequent_tag_baseline(
    train_forms: np.ndarray,
    train_labels: np.ndarray,
    test_forms: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    table: dict[str, Counter] = defaultdict(Counter)
    for form, tag in zip(train_forms, train_labels, strict=False):
        table[form][tag] += 1
    fallback = Counter(train_labels).most_common(1)[0][0]
    lookup = {f: c.most_common(1)[0][0] for f, c in table.items()}
    pred = [lookup.get(f, fallback) for f in test_forms]
    return float(np.mean(np.array(pred) == test_labels))


def pos_probe(
    train_feats: dict[int, np.ndarray],
    train_labels: np.ndarray,
    test_feats: dict[int, np.ndarray],
    test_labels: np.ndarray,
    seed: int = 42,
    max_iter: int = 2000,
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = []
    for depth in sorted(train_feats):
        scaler = StandardScaler().fit(train_feats[depth])
        clf = LogisticRegression(max_iter=max_iter, random_state=seed)
        clf.fit(scaler.transform(train_feats[depth]), train_labels)
        acc = clf.score(scaler.transform(test_feats[depth]), test_labels)
        rows.append(
            {
                "depth": depth,
                "accuracy": float(acc),
                "n_train": len(train_labels),
                "n_test": len(test_labels),
            }
        )
        print(f"[pos-probe] depth {depth}: {acc:.4f}", flush=True)
    return pd.DataFrame(rows)


def run(model, tokenizer, cfg: Config, out: Path, meta: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n_layers = (meta or {}).get("n_layers")
    if n_layers is None:
        n_layers = model.backbone.config.num_hidden_layers
    stride = max(1, cfg.diffusion.probe_layer_stride)
    layers = list(range(0, n_layers + 1, stride))

    def _examples(split):
        ex = examples_for_split(cfg, tokenizer, split)
        if cfg.diffusion.n_probe_sentences is not None:
            ex = ex[: cfg.diffusion.n_probe_sentences]
        return ex

    train_x, train_y, train_f = collect_probe_features(
        model, tokenizer, _examples("select"), cfg.diffusion, layers
    )
    test_x, test_y, test_f = collect_probe_features(
        model, tokenizer, _examples("test"), cfg.diffusion, layers
    )

    baseline = most_frequent_tag_baseline(train_f, train_y, test_f, test_y)
    majority = float((test_y == pd.Series(train_y).mode()[0]).mean())

    table = pos_probe(train_x, train_y, test_x, test_y, seed=cfg.treebank.seed)
    table["lexical_baseline"] = baseline
    table["majority_baseline"] = majority
    table["delta_vs_lexical"] = table["accuracy"] - baseline
    table.to_csv(out / "pos_probe.csv", index=False)
    print(table.to_string(index=False))

"""Layerwise diagnostics: attention entropy, logit lens, POS probe.

All three read per-layer state from the same forward pass shape as the head
search, so they are scored on the same splits and the same denoising schedule.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from .config import DiffusionConfig
from .diffusion import states_at_time, tokenize
from .model import final_norm_module, lm_head_module
from .relations import Example


def _row_entropy(rows: torch.Tensor) -> torch.Tensor:
    p = rows.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=-1)


def attention_entropy(
    model,
    tokenizer,
    examples: list[Example],
    cfg: DiffusionConfig,
    diffusion_time: int | None = None,
    log_every: int = 100,
) -> pd.DataFrame:
    """Mean attention entropy per (layer, head), with and without the sink.

    Entropy is reported in nats and also normalised by log(seq_len), because
    the maximum achievable entropy grows with sequence length and the splits
    contain sentences of very different lengths.
    """
    t = cfg.steps - 1 if diffusion_time is None else diffusion_time
    totals: dict[str, np.ndarray] = {}
    n_rows = 0

    for i, example in enumerate(examples):
        attentions, _, state = states_at_time(
            model,
            tokenizer,
            example.text,
            diffusion_time=t,
            steps=cfg.steps,
            seed=cfg.seed,
            include_bos=cfg.include_bos,
        )
        seq_len = len(state.tokens)
        if seq_len < 3:
            continue

        stacked = torch.stack([a[0].float() for a in attentions])
        raw = _row_entropy(stacked)

        no_sink = stacked.clone()
        no_sink[:, :, :, 0] = 0.0
        trimmed = _row_entropy(no_sink)

        if not totals:
            shape = (stacked.shape[0], stacked.shape[1])
            totals = {
                "entropy": np.zeros(shape),
                "entropy_norm": np.zeros(shape),
                "entropy_no_sink": np.zeros(shape),
                "sink_mass": np.zeros(shape),
            }

        denom = float(np.log(seq_len))
        totals["entropy"] += raw.mean(dim=-1).cpu().numpy()
        totals["entropy_norm"] += (raw / denom).mean(dim=-1).cpu().numpy()
        totals["entropy_no_sink"] += trimmed.mean(dim=-1).cpu().numpy()
        totals["sink_mass"] += stacked[:, :, :, 0].mean(dim=-1).cpu().numpy()
        n_rows += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"[entropy] {i + 1}/{len(examples)} sentences", flush=True)

    if not n_rows:
        return pd.DataFrame()

    rows = []
    n_layers, n_heads = totals["entropy"].shape
    for layer in range(n_layers):
        for head in range(n_heads):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "entropy": totals["entropy"][layer, head] / n_rows,
                    "entropy_norm": totals["entropy_norm"][layer, head] / n_rows,
                    "entropy_no_sink": totals["entropy_no_sink"][layer, head] / n_rows,
                    "sink_mass": totals["sink_mass"][layer, head] / n_rows,
                    "n_sentences": n_rows,
                    "diffusion_time": t,
                }
            )
    return pd.DataFrame(rows)


def _depth_hidden(hidden_states, depth: int, norm):
    """Hidden state at `depth`, normalised the way the lm_head expects.

    `output_hidden_states` emits the embedding output first and each block's
    input after it, so every entry except the last is pre-final-norm. The last
    one has already been normalised by the backbone and must not be normalised
    twice.
    """
    last = len(hidden_states) - 1
    h = hidden_states[depth]
    return h if depth == last else norm(h)


def logit_lens(
    model,
    tokenizer,
    examples: list[Example],
    cfg: DiffusionConfig,
    diffusion_times: list[int] | None = None,
    log_every: int = 50,
) -> pd.DataFrame:
    """Top-1 accuracy of the true token per layer, at masked vs visible slots.

    Run at partially-denoised frames rather than the final one: the last frame
    is forced fully visible, so there would be no masked position to measure
    and every layer would simply be copying its input.
    """
    times = diffusion_times or [8, 16, 32]
    norm = final_norm_module(model)
    head = lm_head_module(model)

    correct: dict[tuple[int, int, bool], int] = defaultdict(int)
    seen: dict[tuple[int, int, bool], int] = defaultdict(int)
    n_depths = 0

    for t in times:
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
            true_ids, _ = tokenize(
                tokenizer, example.text, state.input_ids.device, cfg.include_bos
            )
            if true_ids.shape[1] != state.input_ids.shape[1]:
                continue

            n_depths = len(hidden_states)
            visible = state.is_visible

            for depth in range(n_depths):
                h = _depth_hidden(hidden_states, depth, norm)
                pred = head(h)[0].argmax(dim=-1)
                hit = (pred == true_ids[0]).cpu().numpy()
                for pos in range(len(visible)):
                    key = (t, depth, bool(visible[pos]))
                    seen[key] += 1
                    correct[key] += int(hit[pos])

            if log_every and (i + 1) % log_every == 0:
                print(
                    f"[logit-lens] t={t}: {i + 1}/{len(examples)} sentences",
                    flush=True,
                )

    rows = []
    for t in times:
        for depth in range(n_depths):
            for vis in (True, False):
                n = seen[(t, depth, vis)]
                if not n:
                    continue
                rows.append(
                    {
                        "diffusion_time": t,
                        "depth": depth,
                        "position_state": "visible" if vis else "masked",
                        "accuracy": correct[(t, depth, vis)] / n,
                        "n_positions": n,
                    }
                )
    return pd.DataFrame(rows)


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
    """Hidden states and UPOS labels at the last sub-token of each word."""
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
    """Per-token-type majority tag, backing off to the global majority.

    This is the baseline a POS probe has to beat. English POS is close to
    lexically determined, so a lookup table is already strong and an accuracy
    reported without it says nothing about the representation.
    """
    table: dict[str, Counter] = defaultdict(Counter)
    for form, tag in zip(train_forms, train_labels):
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

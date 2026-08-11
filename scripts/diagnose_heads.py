"""Find out why a head search returned noise.

Run this when `dlmrel search` produces near-chance accuracies. It answers three
questions in order, and the first one that fails localizes the bug:

  1. Does the model have positional heads at all?
     Every GPT-2-derived model has previous-token and next-token heads with
     attention mass near 0.9. If the best head here is at 0.1, we are not
     reading real attention weights -- the problem is model loading or the
     forward pass, not the relation analysis.

  2. Is every head predicting the same position?
     If the argmax is pinned to one column (typically the attention sink at
     index 0 or 1), the exclusion logic is removing the wrong column.

  3. Are predictions off by a constant offset?
     If object->verb accuracy peaks at `pred == receiver + d` for some d != 0,
     the word-to-token alignment is shifted -- most likely a BOS off-by-one.

Usage:
    python scripts/diagnose_heads.py --config configs/diffugpt-s.yaml [--n 200]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlmrel.config import Config
from dlmrel.diffusion import attentions_at_time
from dlmrel.model import load_model
from dlmrel.splits import examples_for_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/diffugpt-s.yaml")
    ap.add_argument("--n", type=int, default=200, help="sentences to inspect")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    model, tokenizer, meta = load_model(cfg.model)
    n_layers, n_heads = meta["n_layers"], meta["n_heads"]
    examples = examples_for_split(cfg, tokenizer, args.split)[: args.n]
    final_t = cfg.diffusion.steps - 1

    # (1) positional-head census, over raw attention with nothing excluded.
    prev_hits = np.zeros((n_layers, n_heads))
    next_hits = np.zeros((n_layers, n_heads))
    self_hits = np.zeros((n_layers, n_heads))
    bos_mass = np.zeros((n_layers, n_heads))
    peak_mass = np.zeros((n_layers, n_heads))
    n_rows = 0

    # (2) where the argmax lands once BOS and self are excluded.
    argmax_cols: Counter[int] = Counter()

    # (3) offset profile for object->verb.
    offsets = list(range(-4, 5))
    off_hits = {d: np.zeros((n_layers, n_heads)) for d in offsets}
    off_total = 0

    for ex in examples:
        attentions, state = attentions_at_time(
            model, tokenizer, ex.text, final_t,
            steps=cfg.diffusion.steps, seed=cfg.diffusion.seed,
            include_bos=cfg.diffusion.include_bos,
        )
        seq = len(state.tokens)
        if seq < 4:
            continue

        # Stack to [layers, heads, seq, seq] once per sentence.
        att = np.stack(
            [a[0].detach().float().cpu().numpy() for a in attentions], axis=0
        )
        q = np.arange(1, seq)  # skip the BOS row itself
        raw = att[:, :, q, :]                       # [L, H, |q|, seq]
        pred_raw = raw.argmax(axis=-1)              # [L, H, |q|]

        prev_hits += (pred_raw == (q - 1)).sum(axis=-1)
        next_hits += (pred_raw == (q + 1)).sum(axis=-1)
        self_hits += (pred_raw == q).sum(axis=-1)
        bos_mass += raw[:, :, :, 0].sum(axis=-1)
        peak_mass += raw.max(axis=-1).sum(axis=-1)
        n_rows += len(q)

        for inst in ex.relations:
            a_span = [t for t in inst.attender_span if t < seq]
            r_span = [t for t in inst.receiver_span if t < seq]
            if not a_span or not r_span:
                continue
            row = att[:, :, a_span[-1], :].copy()    # [L, H, seq]
            row[:, :, 0] = -np.inf                   # exclude BOS
            for c in a_span:
                row[:, :, c] = -np.inf               # exclude self
            pred = row.argmax(axis=-1)               # [L, H]
            argmax_cols.update(pred.reshape(-1).tolist())

            if inst.relation == "object_to_verb":
                off_total += 1
                for d in offsets:
                    tgt = {t + d for t in r_span}
                    off_hits[d] += np.isin(pred, list(tgt))

    print(f"\n=== (1) positional heads over {n_rows} query rows ===")
    print("A healthy GPT-2 shows prev- or next-token heads well above 0.5.")
    for name, arr in (
        ("previous-token", prev_hits), ("next-token", next_hits), ("self", self_hits)
    ):
        frac = arr / max(n_rows, 1)
        li, hi = np.unravel_index(frac.argmax(), frac.shape)
        print(f"  best {name:15s} head L{li} H{hi}: {frac[li, hi]:.3f}")
    print(f"  mean attention mass on BOS   : {(bos_mass / n_rows).mean():.3f}")
    print(f"  mean peak attention per row  : {(peak_mass / n_rows).mean():.3f}")
    print("  (a peak near 1/seq_len means the rows are ~uniform, i.e. not real)")

    print("\n=== (2) argmax column distribution after exclusions ===")
    tot = sum(argmax_cols.values())
    for col, cnt in argmax_cols.most_common(6):
        print(f"  column {col:4d}: {cnt / max(tot, 1):6.1%}")
    print("  (one column dominating means the exclusion logic is wrong)")

    print(f"\n=== (3) object->verb accuracy vs target offset, n={off_total} ===")
    print("A peak at a non-zero offset means the alignment is shifted.")
    for d in offsets:
        frac = off_hits[d] / max(off_total, 1)
        li, hi = np.unravel_index(frac.argmax(), frac.shape)
        star = "  <-- peak" if d != 0 and frac.max() > off_hits[0].max() / max(
            off_total, 1
        ) else ""
        print(f"  offset {d:+d}: best L{li} H{hi} = {frac[li, hi]:.3f}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

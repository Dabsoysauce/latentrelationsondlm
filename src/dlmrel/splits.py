"""Deterministic reconstruction of a split's Example objects.

The GPU stages run in separate processes from `dlmrel data`, and Example
objects hold token spans that are tokenizer-dependent, so they are rebuilt
rather than serialised. Given the same config -- same treebanks, same seed,
same filters -- `examples_for_split` reproduces exactly the sentences that
`dlmrel data` wrote to `sentences_<split>.csv`, and verifies that it did.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .relations import Example, build_examples
from .treebank import load_treebanks, split_sentences


def build_all_splits(cfg: Config, tokenizer) -> dict[str, list[Example]]:
    sentences = load_treebanks(cfg.treebank.treebanks, cfg.treebank.cache_dir)
    usable = build_examples(
        sentences,
        tokenizer,
        cfg.treebank,
        include_bos=cfg.diffusion.include_bos,
        tag="pool",
    )
    return split_sentences(
        usable,
        cfg.treebank.n_select,
        cfg.treebank.n_dev,
        cfg.treebank.n_test,
        cfg.treebank.seed,
        cfg.treebank.shuffle,
    )


def examples_for_split(cfg: Config, tokenizer, split: str) -> list[Example]:
    """Rebuild one split, asserting it matches what `dlmrel data` recorded."""
    examples = build_all_splits(cfg, tokenizer)[split]

    manifest = Path(cfg.out_dir) / f"sentences_{split}.csv"
    if manifest.exists():
        expected = pd.read_csv(manifest)["sentence"].tolist()
        actual = [e.text for e in examples]
        if expected != actual:
            raise RuntimeError(
                f"split {split!r} does not match {manifest}: "
                f"{len(expected)} recorded vs {len(actual)} rebuilt. "
                "The config changed since `dlmrel data` ran -- rerun it."
            )
    return examples

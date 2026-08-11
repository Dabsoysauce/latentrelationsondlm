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


def load_tokenizer(name: str):
    """AutoTokenizer, retrying with remote code for models that require it."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[splits] {name}: plain tokenizer load failed "
            f"({type(exc).__name__}); retrying with trust_remote_code=True"
        )
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def restrict_to_texts(examples: list[Example], texts: set[str]) -> list[Example]:
    """Keep only examples whose sentence is in `texts`, preserving pool order.

    Order matters more than it looks. Splits are carved by index from a shuffled
    pool, so if two models admit even slightly different sentence sets the index
    alignment shifts and the splits diverge far more than the pool difference
    suggests -- measured at 73% test-split overlap for a ~1% pool difference.
    Filtering both models to a common pool *before* shuffling makes the two
    sequences identical, and therefore the splits identical.
    """
    return [e for e in examples if e.text in texts]


def dedupe_by_text(examples: list[Example]) -> list[Example]:
    """Keep the first example per distinct sentence, preserving pool order.

    Without this a sentence repeated in the corpus can be drawn into two
    different splits, so the head search would be selecting on sentences it is
    also reporting on.
    """
    seen: set[str] = set()
    kept: list[Example] = []
    for example in examples:
        if example.text not in seen:
            seen.add(example.text)
            kept.append(example)
    return kept


def common_pool_texts(cfg: Config, sentences) -> set[str]:
    """Sentences every model in `common_pool_models` can align and admit.

    Each model is tokenized once and the results cached, keyed by the model list
    and the filters that affect admission, because `dlmrel data` runs once per
    model and would otherwise redo this work for each.
    """
    import hashlib
    import json

    tc = cfg.treebank
    fingerprint = "|".join(
        sorted(tc.common_pool_models)
        + [
            str(tc.max_seq_len),
            str(tc.min_seq_len),
            str(tc.skip_multiword),
            str(tc.require_full_alignment),
            str(cfg.diffusion.include_bos),
            ",".join(tc.treebanks),
        ]
    )
    key = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
    cache = Path(tc.cache_dir) / f"common_pool_{key}.json"
    if cache.exists():
        texts = set(json.loads(cache.read_text()))
        print(f"[splits] common pool: {len(texts)} sentences (cached)")
        return texts

    texts: set[str] | None = None
    for name in sorted(tc.common_pool_models):
        examples = build_examples(
            sentences,
            load_tokenizer(name),
            tc,
            include_bos=cfg.diffusion.include_bos,
            tag=f"pool[{name}]",
        )
        admitted = {e.text for e in examples}
        texts = admitted if texts is None else (texts & admitted)
    texts = texts or set()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted(texts)))
    print(f"[splits] common pool: {len(texts)} sentences -> {cache}")
    return texts


def build_all_splits(cfg: Config, tokenizer) -> dict[str, list[Example]]:
    sentences = load_treebanks(cfg.treebank.treebanks, cfg.treebank.cache_dir)
    usable = build_examples(
        sentences,
        tokenizer,
        cfg.treebank,
        include_bos=cfg.diffusion.include_bos,
        tag="pool",
    )
    if cfg.treebank.common_pool_models:
        before = len(usable)
        usable = restrict_to_texts(usable, common_pool_texts(cfg, sentences))
        print(f"[splits] restricted pool {before} -> {len(usable)} sentences")
    if cfg.treebank.dedupe_by_text:
        before = len(usable)
        usable = dedupe_by_text(usable)
        print(f"[splits] deduplicated {before} -> {len(usable)} sentences")
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

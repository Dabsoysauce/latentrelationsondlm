"""Shared data preparation and loading.

Loads Universal Dependencies treebanks and builds the three disjoint splits
every model and experiment is scored on. Two defects in the previous pipeline
are fixed here:

1. Sentences were taken by walking the treebank from the top until a quota was
   filled. UD-EWT is ordered by genre, so the "1000 training sentences" were a
   contiguous block of one source. Sampling is now random under a fixed seed.

2. There were two splits, so head *selection* and head *evaluation* shared no
   sentences but the selection itself was never held out. There are now three:
   `select` (search all heads), `dev` (tune anything else), `test` (report).

Splits are carved from one shuffled pool restricted to the sentences every
model's tokenizer admits, so adding a model can only shrink the pool, never
shift which sentences a given model sees relative to another.
"""

from __future__ import annotations

import random
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError, URLError

import pandas as pd
from conllu import parse_incr
from conllu.models import TokenList

from .config import Config
from .relations import Example, build_examples

UD_RAW_BASE = "https://raw.githubusercontent.com/UniversalDependencies/{repo}/master"

# UD file naming is `<langcode>_<treebank>-ud-<split>.conllu`; the stem differs
# per repository, so it is resolved from the repo name.
_STEMS = {
    "UD_English-EWT": "en_ewt",
    "UD_English-GUM": "en_gum",
    "UD_English-LinES": "en_lines",
    "UD_English-ParTUT": "en_partut",
    "UD_English-Atis": "en_atis",
    "UD_English-ESL": "en_esl",
}


def treebank_stem(repo: str) -> str:
    if repo in _STEMS:
        return _STEMS[repo]
    lang, _, name = repo.removeprefix("UD_").partition("-")
    return f"{lang[:2].lower()}_{name.lower()}"


def download_conllu(repo: str, split: str, cache_dir: str | Path) -> Path:
    """Fetch one CoNLL-U file, caching it under `cache_dir`."""
    stem = treebank_stem(repo)
    fname = f"{stem}-ud-{split}.conllu"
    dest = Path(cache_dir) / repo / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        url = f"{UD_RAW_BASE.format(repo=repo)}/{fname}"
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def iter_sentences(path: Path) -> Iterator[TokenList]:
    with open(path, "r", encoding="utf-8") as fh:
        yield from parse_incr(fh)


def load_treebanks(
    repos: list[str],
    cache_dir: str | Path,
    splits: tuple[str, ...] = ("train", "dev", "test"),
) -> list[TokenList]:
    """Load every requested treebank and concatenate all CoNLL-U splits.

    The UD-provided train/dev/test boundaries are deliberately discarded: this
    study's splits are over *sentences sampled for probing*, not over the
    original parser-training partition, and merging first means the three
    probing splits are drawn from one homogeneous pool.
    """
    sentences: list[TokenList] = []
    for repo in repos:
        for split in splits:
            try:
                path = download_conllu(repo, split, cache_dir)
            except (URLError, HTTPError, OSError) as exc:
                print(f"[data] skipping {repo}/{split}: {exc}")
                continue
            n_before = len(sentences)
            for sent in iter_sentences(path):
                sent.metadata["source_treebank"] = repo
                sent.metadata["source_split"] = split
                sentences.append(sent)
            print(f"[data] {repo}/{split}: {len(sentences) - n_before} sentences")
    return sentences


def split_sentences(
    sentences: list,
    n_select: int | None,
    n_dev: int | None,
    n_test: int | None,
    seed: int,
    shuffle: bool = True,
) -> dict[str, list]:
    """Partition into three disjoint splits.

    Splits are carved from a single shuffled pool, so a sentence can never
    appear in more than one. A `None` budget takes everything left over after
    the fixed-size splits are filled.
    """
    pool = list(sentences)
    if shuffle:
        random.Random(seed).shuffle(pool)

    cursor = 0
    out: dict[str, list] = {}
    for name, n in (("select", n_select), ("dev", n_dev), ("test", n_test)):
        if n is None:
            out[name] = pool[cursor:]
            cursor = len(pool)
        else:
            out[name] = pool[cursor : cursor + n]
            cursor += n

    for name, chunk in out.items():
        print(f"[data] split {name}: {len(chunk)} sentences")
    if cursor > len(pool):
        raise ValueError(
            f"requested {cursor} sentences but only {len(pool)} passed filtering; "
            "lower the split sizes or add treebanks"
        )
    return out


def load_tokenizer(name: str):
    """AutoTokenizer, retrying with remote code for models that require it."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[data] {name}: plain tokenizer load failed "
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
    and the filters that affect admission, because data prep runs once per model
    and would otherwise redo this work for each.
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
        print(f"[data] common pool: {len(texts)} sentences (cached)")
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
    print(f"[data] common pool: {len(texts)} sentences -> {cache}")
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
        print(f"[data] restricted pool {before} -> {len(usable)} sentences")
    if cfg.treebank.dedupe_by_text:
        before = len(usable)
        usable = dedupe_by_text(usable)
        print(f"[data] deduplicated {before} -> {len(usable)} sentences")
    return split_sentences(
        usable,
        cfg.treebank.n_select,
        cfg.treebank.n_dev,
        cfg.treebank.n_test,
        cfg.treebank.seed,
        cfg.treebank.shuffle,
    )


def examples_for_split(cfg: Config, tokenizer, split: str) -> list[Example]:
    """Rebuild one split, asserting it matches what data prep recorded."""
    examples = build_all_splits(cfg, tokenizer)[split]

    manifest = Path(cfg.out_dir) / f"sentences_{split}.csv"
    if manifest.exists():
        expected = pd.read_csv(manifest)["sentence"].tolist()
        actual = [e.text for e in examples]
        if expected != actual:
            raise RuntimeError(
                f"split {split!r} does not match {manifest}: "
                f"{len(expected)} recorded vs {len(actual)} rebuilt. "
                "The config changed since data prep ran -- rerun it."
            )
    return examples

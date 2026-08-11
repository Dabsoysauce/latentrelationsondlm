"""Loading Universal Dependencies treebanks and building disjoint splits.

Two defects in the previous pipeline are fixed here:

1. Sentences were taken by walking the treebank from the top until a quota was
   filled. UD-EWT is ordered by genre, so the "1000 training sentences" were a
   contiguous block of one source. Sampling is now random under a fixed seed.

2. There were two splits, so head *selection* and head *evaluation* shared no
   sentences but the selection itself was never held out. There are now three:
   `select` (search all heads), `dev` (tune anything else), `test` (report).
"""

from __future__ import annotations

import random
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError, URLError

from conllu import parse_incr
from conllu.models import TokenList

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
    # Fall back to the UD naming convention: UD_English-Foo -> en_foo.
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
        print(f"[treebank] downloading {url}")
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
                # Not every treebank ships every split; a missing file is a
                # 404 and is expected, so it must not abort the whole pool.
                print(f"[treebank] skipping {repo}/{split}: {exc}")
                continue
            n_before = len(sentences)
            for sent in iter_sentences(path):
                sent.metadata["source_treebank"] = repo
                sent.metadata["source_split"] = split
                sentences.append(sent)
            print(f"[treebank] {repo}/{split}: {len(sentences) - n_before} sentences")
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
        print(f"[treebank] split {name}: {len(chunk)} sentences")
    if cursor > len(pool):
        raise ValueError(
            f"requested {cursor} sentences but only {len(pool)} passed filtering; "
            "lower the split sizes or add treebanks"
        )
    return out

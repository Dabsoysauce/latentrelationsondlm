"""Mapping UD word indices onto model sub-token indices.

UD annotates whole words; the models operate on sub-tokens. Alignment goes
through character offsets rather than through tokenizer-specific word markers,
so the same code path serves GPT-2 byte-level BPE and LLaMA SentencePiece.
This matters because the previous DiffuLLaMA port special-cased the "▁" marker
and therefore diverged from the DiffuGPT path it was meant to replicate.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from conllu.models import TokenList


@dataclass(frozen=True)
class AlignmentDiagnostics:
    method: str
    success: bool
    reason: str


def is_syntactic_token(token) -> bool:
    """True for ordinary tokens; False for multi-word ranges and empty nodes.

    CoNLL-U represents "can't" as a range id (1-2) plus two syntactic pieces.
    Range ids are tuples, empty-node ids are decimals.
    """
    return isinstance(token.get("id"), int)


def has_multiword_tokens(sentence: TokenList) -> bool:
    return any(not is_syntactic_token(tok) for tok in sentence)


def syntactic_tokens(sentence: TokenList) -> tuple[list, dict[int, int]]:
    """Return ordinary tokens plus a map from UD id to 0-based index."""
    toks = [tok for tok in sentence if is_syntactic_token(tok)]
    return toks, {tok["id"]: i for i, tok in enumerate(toks)}


def find_char_spans(text: str, forms: list[str]) -> list[tuple[int, int]] | None:
    """Locate each UD token form in the raw text, scanning left to right.

    Returns None if any form cannot be found in order, which happens when the
    treebank's `text` metadata has been normalised differently from the token
    forms. Such sentences are dropped rather than guessed at.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for form in forms:
        if form is None:
            return None
        idx = text.find(form, cursor)
        if idx < 0:
            return None
        spans.append((idx, idx + len(form)))
        cursor = idx + len(form)
    return spans


def manual_token_offsets(tokenizer, ids: list[int], text: str) -> list[tuple[int, int]]:
    """Fail-closed cumulative decode fallback without substring guessing.

    Cumulative decoding preserves byte-level whitespace and Unicode composition
    better than decoding one token and stripping it. Every decoded prefix must
    match the normalized text prefix exactly; otherwise remaining tokens receive
    zero-width spans and the sentence is excluded by full-alignment checks.
    """
    offsets: list[tuple[int, int]] = []
    normalized_text = unicodedata.normalize("NFC", text)
    previous = ""
    failed = False
    for end in range(1, len(ids) + 1):
        if failed:
            offsets.append((len(previous), len(previous)))
            continue
        try:
            decoded = tokenizer.decode(ids[:end], clean_up_tokenization_spaces=False)
        except TypeError:
            decoded = tokenizer.decode(ids[:end])
        decoded = unicodedata.normalize("NFC", decoded)
        if not decoded.startswith(previous) or not normalized_text.startswith(decoded):
            failed = True
            offsets.append((len(previous), len(previous)))
            continue
        start, stop = len(previous), len(decoded)
        while start < stop and normalized_text[start].isspace():
            start += 1
        while stop > start and normalized_text[stop - 1].isspace():
            stop -= 1
        offsets.append((start, stop))
        previous = decoded
    return offsets


def token_offsets_with_diagnostics(
    tokenizer, text: str
) -> tuple[list[tuple[int, int]], AlignmentDiagnostics]:
    """Return offsets and a structured audit reason."""
    fast = getattr(tokenizer, "_dlmrel_fast_offsets", None)
    if fast is None:
        # A slow tokenizer does not necessarily raise on return_offsets_mapping.
        # Some silently drop the argument and return an encoding without the
        # key, so probing for an exception alone records a false positive and
        # the next real call dies with KeyError: 'offset_mapping'. Confirm the
        # key actually came back.
        try:
            probe = tokenizer("probe", return_offsets_mapping=True)
            fast = "offset_mapping" in probe
        except Exception:  # noqa: BLE001
            fast = False
        tokenizer._dlmrel_fast_offsets = fast
    if fast:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
        return offsets, AlignmentDiagnostics("offset_mapping", True, "ok")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    offsets = manual_token_offsets(tokenizer, ids, text)
    success = all(stop > start for start, stop in offsets)
    reason = "ok" if success else "cumulative_decode_prefix_mismatch"
    return offsets, AlignmentDiagnostics("cumulative_decode", success, reason)


def token_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    """Per-token character offsets, falling back to a manual scan."""
    return token_offsets_with_diagnostics(tokenizer, text)[0]


def align_words_to_tokens(
    text: str,
    char_spans: list[tuple[int, int]],
    tokenizer,
    include_bos: bool = True,
) -> dict[int, list[int]]:
    """Map each UD word index to the model token indices covering it.

    Token indices are shifted by +1 when `include_bos`, because the experiment
    prepends BOS to every sequence.
    """
    offsets = token_offsets(tokenizer, text)
    shift = 1 if include_bos else 0

    word_to_tokens: dict[int, list[int]] = {}
    for wi, (start, end) in enumerate(char_spans):
        covered = [
            ti + shift
            for ti, (a, b) in enumerate(offsets)
            # Any character overlap counts; a sub-token may straddle a word
            # boundary in byte-level BPE.
            if a < end and b > start
        ]
        if covered:
            word_to_tokens[wi] = covered
    return word_to_tokens

from __future__ import annotations

import pytest

from dlmrel.alignment import (
    align_words_to_tokens,
    find_char_spans,
    manual_token_offsets,
    token_offsets,
)


class _SlowTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise ValueError("offset mapping not available")
        return {"input_ids": list(range(len(self.pieces)))}

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_manual_offsets_locate_each_piece():
    tok = _SlowTokenizer(["The", " chef", " cooked"])
    offsets = manual_token_offsets(tok, [0, 1, 2], "The chef cooked")
    assert offsets == [(0, 3), (4, 8), (9, 15)]


def test_unplaceable_piece_gets_a_zero_width_span():
    tok = _SlowTokenizer(["The", " zzz", " cooked"])
    offsets = manual_token_offsets(tok, [0, 1, 2], "The chef cooked")
    assert offsets[1][0] == offsets[1][1]


def test_token_offsets_falls_back_when_mapping_unavailable():
    tok = _SlowTokenizer(["The", " chef"])
    assert token_offsets(tok, "The chef") == [(0, 3), (4, 8)]


def test_alignment_maps_words_to_covering_tokens():
    tok = _SlowTokenizer(["The", " chef", " cooked"])
    spans = find_char_spans("The chef cooked", ["The", "chef", "cooked"])
    aligned = align_words_to_tokens("The chef cooked", spans, tok, include_bos=True)
    assert aligned == {0: [1], 1: [2], 2: [3]}


def test_alignment_shift_zero_without_bos():
    tok = _SlowTokenizer(["The", " chef"])
    spans = find_char_spans("The chef", ["The", "chef"])
    aligned = align_words_to_tokens("The chef", spans, tok, include_bos=False)
    assert aligned == {0: [0], 1: [1]}


@pytest.mark.parametrize("include_bos,expected", [(True, [1]), (False, [0])])
def test_bos_shift(include_bos, expected):
    tok = _SlowTokenizer(["word"])
    spans = find_char_spans("word", ["word"])
    aligned = align_words_to_tokens("word", spans, tok, include_bos=include_bos)
    assert aligned[0] == expected

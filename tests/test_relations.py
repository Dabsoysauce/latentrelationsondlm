"""Tests for gold relation extraction and word/sub-token alignment."""

from __future__ import annotations

import conllu
import pytest

from dlmrel.alignment import find_char_spans, has_multiword_tokens, syntactic_tokens
from dlmrel.relations import extract_relations

# "The old man saw a red dog ." with gold annotations:
#   subject det->noun (The->man), subject adj->noun (old->man),
#   subject->verb (man->saw), object det->noun (a->dog),
#   object adj->noun (red->dog), object->verb (dog->saw)
CONLLU = """\
# text = The old man saw a red dog .
1\tThe\tthe\tDET\tDT\t_\t3\tdet\t_\t_
2\told\told\tADJ\tJJ\t_\t3\tamod\t_\t_
3\tman\tman\tNOUN\tNN\t_\t4\tnsubj\t_\t_
4\tsaw\tsee\tVERB\tVBD\t_\t0\troot\t_\t_
5\ta\ta\tDET\tDT\t_\t7\tdet\t_\t_
6\tred\tred\tADJ\tJJ\t_\t7\tamod\t_\t_
7\tdog\tdog\tNOUN\tNN\t_\t4\tobj\t_\t_
8\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_

"""

MULTIWORD = """\
# text = He can't go .
1\tHe\the\tPRON\tPRP\t_\t4\tnsubj\t_\t_
2-3\tcan't\t_\t_\t_\t_\t_\t_\t_\t_
2\tca\tcan\tAUX\tMD\t_\t4\taux\t_\t_
3\tn't\tnot\tPART\tRB\t_\t4\tadvmod\t_\t_
4\tgo\tgo\tVERB\tVB\t_\t0\troot\t_\t_
5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_

"""


@pytest.fixture
def sentence():
    return conllu.parse(CONLLU)[0]


def relations_of(sentence):
    tokens, id_to_idx = syntactic_tokens(sentence)
    # Identity alignment: one word per token index.
    word_to_tokens = {i: [i] for i in range(len(tokens))}
    return extract_relations(tokens, id_to_idx, word_to_tokens)


def test_all_six_relations_are_extracted(sentence):
    found = {r.relation for r in relations_of(sentence)}
    assert found == {
        "subject_to_verb",
        "object_to_verb",
        "subject_adj_to_noun",
        "object_adj_to_noun",
        "subject_det_to_noun",
        "object_det_to_noun",
    }


def test_adjective_to_noun_is_adjacent(sentence):
    # This adjacency is precisely why a +1 offset head scores so highly on
    # adjective->noun, and why the null model is necessary.
    for rel in relations_of(sentence):
        if rel.relation.endswith("_adj_to_noun"):
            assert rel.word_distance == 1


def test_determiner_to_noun_is_not_always_adjacent(sentence):
    # "The old man": an intervening adjective pushes the determiner to
    # distance 2. This partial spread is why the fixed-offset null reaches
    # only ~0.55-0.60 on determiner->noun instead of saturating -- and it is
    # still enough to match DiffuGPT-S's best determiner head.
    distances = {
        rel.word_distance
        for rel in relations_of(sentence)
        if rel.relation.endswith("_det_to_noun")
    }
    assert distances == {2}


def test_object_to_verb_points_backwards(sentence):
    obj = next(r for r in relations_of(sentence) if r.relation == "object_to_verb")
    assert obj.attender_text == "dog"
    assert obj.receiver_text == "saw"
    assert obj.word_distance == -3


def test_punctuation_and_root_are_not_relations(sentence):
    texts = {(r.attender_text, r.receiver_text) for r in relations_of(sentence)}
    assert not any(a == "." for a, _ in texts)
    # The root verb has head 0 and must not produce an instance.
    assert not any(a == "saw" for a, _ in texts)


def test_multiword_tokens_are_detected():
    sentence = conllu.parse(MULTIWORD)[0]
    assert has_multiword_tokens(sentence)
    tokens, _ = syntactic_tokens(sentence)
    # The 2-3 range row is excluded; its two pieces remain.
    assert [t["form"] for t in tokens] == ["He", "ca", "n't", "go", "."]


def test_char_spans_are_found_in_order():
    spans = find_char_spans("The old man saw a red dog .", ["The", "old", "man"])
    assert spans == [(0, 3), (4, 7), (8, 11)]


def test_char_spans_fail_closed_on_mismatch():
    # A form absent from the text must yield None so the sentence is dropped
    # rather than silently misaligned.
    assert find_char_spans("The old man", ["The", "young"]) is None


def test_repeated_forms_advance_the_cursor():
    spans = find_char_spans("the cat the dog", ["the", "cat", "the", "dog"])
    assert spans == [(0, 3), (4, 7), (8, 11), (12, 15)]

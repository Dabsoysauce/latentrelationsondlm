from __future__ import annotations

import conllu
import pytest

from dlmrel import relations
from dlmrel.alignment import find_char_spans, has_multiword_tokens, syntactic_tokens
from dlmrel.config import DatasetConfig
from dlmrel.relations import build_example, extract_relations

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
    for rel in relations_of(sentence):
        if rel.relation.endswith("_adj_to_noun"):
            assert rel.word_distance == 1


def test_determiner_to_noun_is_not_always_adjacent(sentence):
    distances = {rel.word_distance for rel in relations_of(sentence) if rel.relation.endswith("_det_to_noun")}
    assert distances == {2}


def test_object_to_verb_points_backwards(sentence):
    obj = next(r for r in relations_of(sentence) if r.relation == "object_to_verb")
    assert obj.attender_text == "dog"
    assert obj.receiver_text == "saw"
    assert obj.word_distance == -3


def test_punctuation_and_root_are_not_relations(sentence):
    texts = {(r.attender_text, r.receiver_text) for r in relations_of(sentence)}
    assert not any(a == "." for a, _ in texts)
    assert not any(a == "saw" for a, _ in texts)


def test_multiword_tokens_are_detected():
    sentence = conllu.parse(MULTIWORD)[0]
    assert has_multiword_tokens(sentence)
    tokens, _ = syntactic_tokens(sentence)
    assert [t["form"] for t in tokens] == ["He", "ca", "n't", "go", "."]


def test_empty_nodes_are_excluded_from_syntactic_tokens():
    sentence = conllu.parse(
        """# text = I saw him today .
1\tI\tI\tPRON\t_\t_\t2\tnsubj\t_\t_
2\tsaw\tsee\tVERB\t_\t_\t0\troot\t_\t_
3\thim\the\tPRON\t_\t_\t2\tobj\t_\t_
3.1\tghost\tghost\tNOUN\t_\t_\t_\t_\t_\t_
4\ttoday\ttoday\tADV\t_\t_\t2\tadvmod\t_\t_
5\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

"""
    )[0]
    tokens, mapping = syntactic_tokens(sentence)
    assert [token["id"] for token in tokens] == [1, 2, 3, 4, 5]
    assert set(mapping) == {1, 2, 3, 4, 5}


def test_char_spans_are_found_in_order():
    spans = find_char_spans("The old man saw a red dog .", ["The", "old", "man"])
    assert spans == [(0, 3), (4, 7), (8, 11)]


def test_char_spans_fail_closed_on_mismatch():
    assert find_char_spans("The old man", ["The", "young"]) is None


def test_repeated_forms_advance_the_cursor():
    spans = find_char_spans("the cat the dog", ["the", "cat", "the", "dog"])
    assert spans == [(0, 3), (4, 7), (8, 11), (12, 15)]


def test_dependency_subtypes_csubj_iobj_aux_and_propn_are_supported():
    tokens = [
        {"id": 1, "form": "Leaving", "upos": "NOUN", "head": 2, "deprel": "csubj:outer"},
        {"id": 2, "form": "is", "upos": "AUX", "head": 0, "deprel": "root"},
        {"id": 3, "form": "Ada", "upos": "PROPN", "head": 2, "deprel": "iobj:agent"},
        {"id": 4, "form": "the", "upos": "DET", "head": 3, "deprel": "det:def"},
        {"id": 5, "form": "wise", "upos": "ADJ", "head": 3, "deprel": "amod:emph"},
    ]
    relations = extract_relations(
        tokens,
        {item["id"]: i for i, item in enumerate(tokens)},
        {i: [i] for i in range(5)},
    )

    assert {relation.relation for relation in relations} == {
        "subject_to_verb",
        "object_to_verb",
        "object_det_to_noun",
        "object_adj_to_noun",
    }


def test_wrong_pos_roots_and_missing_heads_do_not_create_relations():
    tokens = [
        {"id": 1, "form": "x", "upos": "NOUN", "head": 2, "deprel": "obj"},
        {"id": 2, "form": "noun", "upos": "NOUN", "head": 0, "deprel": "root"},
        {"id": 3, "form": "adj", "upos": "ADJ", "head": 99, "deprel": "amod"},
        {"id": 4, "form": "root", "upos": "VERB", "head": 0, "deprel": "obj"},
    ]
    assert extract_relations(
        tokens, {item["id"]: i for i, item in enumerate(tokens)}, {i: [i] for i in range(4)}
    ) == []


def test_structural_metadata_and_instance_ids_are_stable():
    tokens = [
        {"id": 1, "form": "work", "upos": "NOUN", "head": 3, "deprel": "nsubj:pass"},
        {"id": 2, "form": ",", "upos": "PUNCT", "head": 3, "deprel": "punct"},
        {"id": 3, "form": "done", "upos": "VERB", "head": 5, "deprel": "acl:relcl"},
        {"id": 4, "form": "was", "upos": "AUX", "head": 3, "deprel": "aux:pass"},
        {"id": 5, "form": "continued", "upos": "VERB", "head": 7, "deprel": "conj"},
        {"id": 6, "form": "tasks", "upos": "NOUN", "head": 5, "deprel": "obj"},
        {"id": 7, "form": "began", "upos": "VERB", "head": 0, "deprel": "root"},
    ]
    mapping = {item["id"]: i for i, item in enumerate(tokens)}
    spans = {i: [i] for i in range(len(tokens))}
    first = extract_relations(tokens, mapping, spans)
    second = extract_relations(tokens, mapping, spans)
    passive = next(relation for relation in first if relation.attender_text == "work")

    assert passive.passive_voice is True
    assert passive.punctuation_between is True
    assert passive.relative_clause is True
    assert passive.embedded_clause is True
    assert passive.word_distance == 2
    assert next(relation for relation in first if relation.attender_text == "tasks").coordinated
    assert [relation.instance_id for relation in first] == [relation.instance_id for relation in second]


def test_128_subtoken_limit_counts_the_prepended_bos(monkeypatch, sentence):
    class LengthTokenizer:
        def __init__(self, length):
            self.length = length

        def __call__(self, _text, **_kwargs):
            return {"input_ids": list(range(self.length))}

    mapping = {index: [index + 1] for index in range(8)}
    monkeypatch.setattr(relations, "align_words_to_tokens", lambda *_args, **_kwargs: mapping)
    cfg = DatasetConfig(revision="abc")

    accepted = build_example(sentence, LengthTokenizer(127), cfg, include_bos=True)
    rejected = build_example(sentence, LengthTokenizer(128), cfg, include_bos=True)

    assert accepted is not None and accepted.seq_len == 128
    assert rejected is None

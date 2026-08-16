"""Extracting gold dependency relations and turning treebank sentences into
model-ready examples.

Each relation instance records the *word* distance between endpoints as well as
their token spans. That distance is what the fixed-offset null model consumes,
and what the stratified analysis bins over, so it is stored at extraction time
rather than recovered later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from conllu.models import TokenList

from .alignment import (
    align_words_to_tokens,
    find_char_spans,
    syntactic_tokens,
)
from .config import NOUN_UPOS, OBJECT_DEPS, SUBJECT_DEPS, VERB_UPOS, DatasetConfig


@dataclass
class RelationInstance:
    relation: str
    attender_span: list[int]
    receiver_span: list[int]
    attender_text: str
    receiver_text: str
    attender_word_idx: int
    receiver_word_idx: int
    dep: str
    instance_id: str = ""
    attender_upos: str = ""
    receiver_upos: str = ""
    punctuation_between: bool = False
    clause_depth: int = 0
    embedded_clause: bool = False
    coordinated: bool = False
    relative_clause: bool = False
    passive_voice: bool = False
    intervening_verbs: int = 0
    intervening_nouns: int = 0

    @property
    def word_distance(self) -> int:
        """Signed receiver-minus-attender distance in words.

        Positive means the receiver follows the attender. This is the quantity a
        fixed-offset head would have to guess correctly.
        """
        return self.receiver_word_idx - self.attender_word_idx


@dataclass
class Example:
    text: str
    tokens: list[str]
    upos: list[str]
    deprel: list[str]
    head: list[int]
    word_to_tokens: dict[int, list[int]]
    relations: list[RelationInstance]
    seq_len: int
    source: str = ""
    sentence_id: str = ""
    language: str = ""
    original_split: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _base_dep(dep: str | None) -> str:
    """UD labels carry subtypes (`nsubj:pass`); group by the base relation."""
    return "" if dep is None else str(dep).split(":", 1)[0]


def _noun_role(head_token) -> str | None:
    """Classify a noun as subject-side or object-side by its own dependency."""
    base = _base_dep(head_token.get("deprel"))
    if base in SUBJECT_DEPS:
        return "subject"
    if base in OBJECT_DEPS:
        return "object"
    return None


def extract_relations(
    tokens: list,
    id_to_idx: dict[int, int],
    word_to_tokens: dict[int, list[int]],
) -> list[RelationInstance]:
    """Pull the six studied relations out of one gold-annotated sentence."""
    instances: list[RelationInstance] = []

    for i, tok in enumerate(tokens):
        if i not in word_to_tokens:
            continue

        head_id = tok.get("head")
        if not isinstance(head_id, int) or head_id == 0 or head_id not in id_to_idx:
            continue
        head_i = id_to_idx[head_id]
        if head_i not in word_to_tokens:
            continue

        head_tok = tokens[head_i]
        dep = tok.get("deprel")
        base = _base_dep(dep)
        head_upos = head_tok.get("upos")

        relation: str | None = None
        if base in SUBJECT_DEPS and head_upos in VERB_UPOS:
            relation = "subject_to_verb"
        elif base in OBJECT_DEPS and head_upos in VERB_UPOS:
            relation = "object_to_verb"
        elif base in ("amod", "det") and head_upos in NOUN_UPOS:
            role = _noun_role(head_tok)
            if role is not None:
                kind = "adj" if base == "amod" else "det"
                relation = f"{role}_{kind}_to_noun"

        if relation is None:
            continue

        between = tokens[min(i, head_i) + 1 : max(i, head_i)]
        path_depth = _ancestor_depth(tokens, id_to_idx, i)
        passive = str(dep or "").startswith("nsubj:pass") or any(
            _base_dep(item.get("deprel")) == "aux" and str(item.get("deprel")).endswith(":pass")
            for item in tokens
            if item.get("head") == head_id
        )
        relation_id = hashlib.sha256(
            f"{relation}|{i}|{head_i}|{tok.get('form')}|{head_tok.get('form')}".encode()
        ).hexdigest()[:20]
        instances.append(
            RelationInstance(
                relation=relation,
                attender_span=word_to_tokens[i],
                receiver_span=word_to_tokens[head_i],
                attender_text=tok.get("form"),
                receiver_text=head_tok.get("form"),
                attender_word_idx=i,
                receiver_word_idx=head_i,
                dep=dep,
                instance_id=relation_id,
                attender_upos=str(tok.get("upos") or ""),
                receiver_upos=str(head_tok.get("upos") or ""),
                punctuation_between=any(item.get("upos") == "PUNCT" for item in between),
                clause_depth=path_depth,
                embedded_clause=path_depth > 1
                or _base_dep(head_tok.get("deprel")) in {"ccomp", "xcomp", "advcl", "acl"},
                coordinated=_base_dep(tok.get("deprel")) == "conj"
                or _base_dep(head_tok.get("deprel")) == "conj",
                relative_clause=str(head_tok.get("deprel") or "").startswith("acl:relcl"),
                passive_voice=passive,
                intervening_verbs=sum(item.get("upos") in VERB_UPOS for item in between),
                intervening_nouns=sum(item.get("upos") in NOUN_UPOS for item in between),
            )
        )

    return instances


def _ancestor_depth(tokens: list, id_to_idx: dict[int, int], start: int) -> int:
    depth, current, seen = 0, start, set()
    while current not in seen:
        seen.add(current)
        head_id = tokens[current].get("head")
        if not isinstance(head_id, int) or head_id == 0 or head_id not in id_to_idx:
            break
        current = id_to_idx[head_id]
        depth += 1
    return depth


def build_example(
    sentence: TokenList,
    tokenizer,
    cfg: DatasetConfig,
    include_bos: bool = True,
    max_subtokens: int = 128,
) -> Example | None:
    """Convert one CoNLL-U sentence into an Example, or None if unusable."""
    text = sentence.metadata.get("text")
    if not text:
        return None

    tokens, id_to_idx = syntactic_tokens(sentence)
    forms = [tok.get("form") for tok in tokens]
    if len(forms) < cfg.min_words:
        return None
    if cfg.max_words is not None and len(forms) > cfg.max_words:
        return None

    char_spans = find_char_spans(text, forms)
    if char_spans is None:
        return None

    seq_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    seq_len += 1 if include_bos else 0
    if seq_len > max_subtokens:
        return None

    word_to_tokens = align_words_to_tokens(text, char_spans, tokenizer, include_bos)
    if len(word_to_tokens) < len(tokens):
        return None

    relations = extract_relations(tokens, id_to_idx, word_to_tokens)
    if not relations:
        return None

    sent_id = str(sentence.metadata.get("sent_id") or "")
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    stable_sentence_id = sent_id or f"text-{text_hash}"
    for relation in relations:
        relation.instance_id = f"{stable_sentence_id}:{relation.instance_id}"

    return Example(
        text=text,
        tokens=forms,
        upos=[tok.get("upos") for tok in tokens],
        deprel=[tok.get("deprel") for tok in tokens],
        head=[tok.get("head") for tok in tokens],
        word_to_tokens=word_to_tokens,
        relations=relations,
        seq_len=seq_len,
        source=sentence.metadata.get("source_treebank", ""),
        sentence_id=stable_sentence_id,
        original_split=str(sentence.metadata.get("source_split") or ""),
    )

"""Extracting gold dependency relations and turning treebank sentences into
model-ready examples.

Each relation instance records the *word* distance between endpoints as well as
their token spans. That distance is what the fixed-offset null model consumes,
and what the stratified analysis bins over, so it is stored at extraction time
rather than recovered later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from conllu.models import TokenList

from .alignment import (
    align_words_to_tokens,
    find_char_spans,
    has_multiword_tokens,
    syntactic_tokens,
)
from .config import NOUN_UPOS, OBJECT_DEPS, SUBJECT_DEPS, VERB_UPOS, TreebankConfig


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
            )
        )

    return instances


def build_example(
    sentence: TokenList,
    tokenizer,
    cfg: TreebankConfig,
    include_bos: bool = True,
) -> Example | None:
    """Convert one CoNLL-U sentence into an Example, or None if unusable."""
    if cfg.skip_multiword and has_multiword_tokens(sentence):
        return None

    text = sentence.metadata.get("text")
    if not text:
        return None

    tokens, id_to_idx = syntactic_tokens(sentence)
    forms = [tok.get("form") for tok in tokens]
    if not forms:
        return None

    char_spans = find_char_spans(text, forms)
    if char_spans is None:
        return None

    seq_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    seq_len += 1 if include_bos else 0
    if not (cfg.min_seq_len <= seq_len <= cfg.max_seq_len):
        return None

    word_to_tokens = align_words_to_tokens(text, char_spans, tokenizer, include_bos)
    if cfg.require_full_alignment and len(word_to_tokens) < len(tokens):
        return None

    relations = extract_relations(tokens, id_to_idx, word_to_tokens)
    if not relations:
        return None

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
    )


def build_examples(
    sentences: list[TokenList],
    tokenizer,
    cfg: TreebankConfig,
    include_bos: bool = True,
    limit: int | None = None,
    tag: str = "",
) -> list[Example]:
    """Filter and convert a list of sentences, reporting why sentences drop out."""
    examples: list[Example] = []
    dropped = 0
    for sentence in sentences:
        example = build_example(sentence, tokenizer, cfg, include_bos)
        if example is None:
            dropped += 1
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break

    counts = Counter(inst.relation for ex in examples for inst in ex.relations)
    print(f"[relations] {tag}: {len(examples)} usable, {dropped} dropped")
    print(f"[relations] {tag}: instances {dict(counts)}")
    return examples


def relations_to_records(examples: list[Example], split: str) -> list[dict]:
    """Flatten to rows for the audit CSV that every downstream analysis reads."""
    rows = []
    for si, ex in enumerate(examples):
        for inst in ex.relations:
            rows.append(
                {
                    "split": split,
                    "sentence_idx": si,
                    "sentence": ex.text,
                    "source": ex.source,
                    "relation": inst.relation,
                    "attender_text": inst.attender_text,
                    "receiver_text": inst.receiver_text,
                    "dep": inst.dep,
                    "attender_span": inst.attender_span,
                    "receiver_span": inst.receiver_span,
                    "attender_word_idx": inst.attender_word_idx,
                    "receiver_word_idx": inst.receiver_word_idx,
                    "word_distance": inst.word_distance,
                }
            )
    return rows

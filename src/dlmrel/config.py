"""Experiment configuration.

Every knob that changes a reported number lives here so a run is reproducible
from a single YAML file. Defaults encode the corrections identified in the
2026-08-01 audit of the previous pipeline:

  * random sentence sampling instead of taking the first N from the treebank
  * a longer sequence cap, so long-distance dependencies survive filtering
  * three disjoint splits, so "held out" is held out over heads as well as
    sentences
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Relation direction is always attender -> receiver.
RELATION_NAMES = (
    "object_to_verb",
    "subject_to_verb",
    "object_adj_to_noun",
    "subject_adj_to_noun",
    "object_det_to_noun",
    "subject_det_to_noun",
)

SUBJECT_DEPS = frozenset({"nsubj", "csubj"})
OBJECT_DEPS = frozenset({"obj", "iobj"})
NOUN_UPOS = frozenset({"NOUN", "PROPN"})
VERB_UPOS = frozenset({"VERB", "AUX"})


@dataclass
class TreebankConfig:
    """Which UD data to use and how to sample it."""

    # Treebank repository names under github.com/UniversalDependencies.
    treebanks: list[str] = field(default_factory=lambda: ["UD_English-EWT"])
    cache_dir: str = "data/ud"

    # Sentence budget per split. None means "everything that passes filters".
    n_select: int | None = 4000
    n_dev: int | None = 1000
    n_test: int | None = 1000

    # The previous pipeline capped at 64 BPE tokens, which preferentially
    # deleted the long-distance dependencies that distinguish a relation head
    # from a fixed-offset head. 128 keeps most of EWT.
    max_seq_len: int = 128
    min_seq_len: int = 4

    # Sample randomly rather than walking the treebank from the top. EWT is
    # ordered by genre, so sequential sampling yields a single-source block.
    shuffle: bool = True
    seed: int = 42

    # Drop sentences containing multi-word tokens ("can't" -> "ca" + "n't").
    # These break the direct form-to-character-span match.
    skip_multiword: bool = True
    # Drop a sentence if any UD word fails to align to a model token.
    require_full_alignment: bool = True

    # Models whose tokenizers must all admit a sentence for it to enter the
    # pool. Splits are carved by index from a shuffled pool, so two models that
    # admit slightly different sentences end up with badly misaligned splits --
    # a ~1% pool difference produced only 73% test-split overlap. Listing both
    # models here makes the pool, and therefore the splits, identical.
    common_pool_models: list[str] = field(default_factory=list)

    # UD-EWT is web text and repeats boilerplate verbatim ("Attached are the gas
    # settlement and support for May."). Keeping every copy puts the same
    # sentence in two splits -- measured at 27 test sentences also present in
    # the selection split -- so head selection would partly see its own test set.
    dedupe_by_text: bool = True


@dataclass
class ModelConfig:
    name: str = "diffusionfamily/diffullama"
    family: str = "diffullama"  # "diffullama" | "diffugpt" | "dream"
    dtype: str = "bfloat16"
    device: str = "cuda"
    # Only eager attention returns attention weights; sdpa and
    # flash_attention_2 silently drop them and the whole analysis becomes zeros.
    attn_implementation: str = "eager"


@dataclass
class DiffusionConfig:
    steps: int = 64
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    include_bos: bool = True
    # Attention-sink column and the attender's own positions are masked before
    # the argmax, otherwise every head "predicts" BOS or itself.
    exclude_bos: bool = True
    exclude_self: bool = True
    # Which sub-token of a multi-token word carries the attention row.
    attender_token: str = "last"  # "last" | "first"
    # Minimum number of still-masked positions for a frame to count toward the
    # masked-state statistic.
    min_masked_positions: int = 25


@dataclass
class AnalysisConfig:
    # Offset range searched for the fixed-offset null model.
    offset_range: tuple[int, int] = (-15, 15)
    n_bootstrap: int = 10_000
    ci: float = 0.95
    # Dependency-distance bins, in words, for the stratified analysis that
    # separates positional heads from relational ones.
    distance_bins: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 100])


@dataclass
class Config:
    treebank: TreebankConfig = field(default_factory=TreebankConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    out_dir: str = "results/run"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            treebank=TreebankConfig(**raw.get("treebank", {})),
            model=ModelConfig(**raw.get("model", {})),
            diffusion=DiffusionConfig(**raw.get("diffusion", {})),
            analysis=AnalysisConfig(**raw.get("analysis", {})),
            out_dir=raw.get("out_dir", "results/run"),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

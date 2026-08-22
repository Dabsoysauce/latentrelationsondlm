"""One strict configuration schema for every active experiment."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

SCHEMA_VERSION = "dlmrel-config-v2"
TRACKS = (
    "confirmatory_ewt",
    "external_treebank_transfer",
    "exploratory_extensions",
)
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
PROTOCOL_SEEDS = [42, 43, 44]
PROTOCOL_STEPS = 64
PAPER_EXPERIMENT_TYPES = (
    "relation_head_receiver_prediction",
    "relation_head_receiver_prediction_over_diffusion_time",
    "attention_entropy",
    "pos_token_class_linear_probes",
    "final_token_prediction_by_layer",
    "prediction_before_unmasking_timing_analysis",
    "direct_logit_attribution",
    "matched_relation_head_ablation",
    "attention_heatmaps_and_trajectories",
    "multilingual_relation_head_transfer",
)
_ALL_64_PROGRESS = [step / 63 for step in range(64)]
PROTOCOL_PROGRESS = {
    "head_search": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    "time_curve": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    "attention_entropy": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    "logit_lens": [0.0, 0.25, 0.5, 0.75, 1.0],
    "pos_probe": [0.5],
    "relation_head_receiver_prediction": [1.0],
    "relation_head_receiver_prediction_over_diffusion_time": _ALL_64_PROGRESS,
    "attention_entropy_paper": _ALL_64_PROGRESS,
    "pos_token_class_linear_probes": [0.0, 0.25, 0.5, 0.75],
    "final_token_prediction_by_layer": _ALL_64_PROGRESS,
    "prediction_before_unmasking_timing_analysis": _ALL_64_PROGRESS,
    "direct_logit_attribution": [20 / 63, 30 / 63, 40 / 63],
    "matched_relation_head_ablation": [20 / 63, 30 / 63, 40 / 63],
    "attention_heatmaps_and_trajectories": _ALL_64_PROGRESS,
    "multilingual_relation_head_transfer": _ALL_64_PROGRESS,
}
_SHA256 = re.compile(r"sha256:[0-9a-fA-F]{64}\Z")


def is_paper_experiment(experiment: ExperimentConfig) -> bool:
    """Whether a resolved experiment belongs to the corrected active protocol."""
    return experiment.type in PAPER_EXPERIMENT_TYPES or (
        experiment.type == "attention_entropy" and experiment.id == "attention_entropy"
    )


class ConfigError(ValueError):
    """A config is incomplete, contradictory, or contains an unknown field."""


@dataclass(frozen=True)
class DatasetConfig:
    id: str = "ewt"
    treebank: str = "UD_English-EWT"
    language: str = "en"
    release: str = "2.15"
    repository: str = "https://github.com/UniversalDependencies/UD_English-EWT"
    revision: str = "unknown"
    checksums: dict[str, str] = field(default_factory=dict)
    cache_dir: str = "data/ud"
    select_from: str = "train"
    dev_from: str = "dev"
    test_from: str = "test"
    n_select: int | None = 4000
    n_dev: int | None = 1000
    n_test: int | None = 1000
    seed: int = 42
    dedupe_normalized_text: bool = True
    min_words: int = 4
    max_words: int | None = None

    def validate(self) -> None:
        if not self.id or not self.treebank or not self.revision:
            raise ConfigError("dataset id, treebank, and revision are required")
        if (self.select_from, self.dev_from, self.test_from) != ("train", "dev", "test"):
            raise ConfigError("rigorous datasets must map select/dev/test to official train/dev/test")
        if set(self.checksums) != {"train", "dev", "test"}:
            raise ConfigError("dataset checksums must contain exactly train, dev, and test")
        if any(
            not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            for digest in self.checksums.values()
        ):
            raise ConfigError("dataset checksums must be sha256:<64 hex characters>")


@dataclass(frozen=True)
class CapabilityConfig:
    logits: bool = False
    hidden_states: bool = False
    attentions: bool = False


@dataclass(frozen=True)
class ModelConfig:
    id: str = "fake"
    name: str = "fake"
    family: str = "fake"
    revision: str = "local-v1"
    tokenizer_revision: str = "local-v1"
    remote_code_revision: str | None = None
    dtype: str = "float32"
    device: str = "cpu"
    attn_implementation: str = "eager"
    capabilities: CapabilityConfig = field(default_factory=CapabilityConfig)

    def validate(self) -> None:
        if not self.revision or self.revision in {"main", "master"}:
            raise ConfigError("model revision must be immutable, not main/master")
        if self.remote_code_revision in {"main", "master"}:
            raise ConfigError("remote-code revision must be immutable")
        if not self.tokenizer_revision or self.tokenizer_revision in {"main", "master"}:
            raise ConfigError("tokenizer revision must be immutable, not main/master")


@dataclass(frozen=True)
class ScoringConfig:
    attender_rows: str = "mean"
    receiver_span: str = "sum"
    top_k: int = 5
    primary_relation: str = "object_to_verb"
    primary_visibility: str = "both_masked"

    def validate(self) -> None:
        if self.attender_rows not in {"mean", "first", "last"}:
            raise ConfigError("attender_rows must be mean, first, or last")
        if self.receiver_span not in {"sum", "mean", "max", "source_argmax"}:
            raise ConfigError("receiver_span must be sum, mean, max, or source_argmax")
        if self.top_k < 1:
            raise ConfigError("top_k must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    id: str = "head_search"
    type: str = "head_search"
    steps: int = 64
    normalized_progress: list[float] = field(
        default_factory=lambda: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    settings: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.seeds != PROTOCOL_SEEDS:
            raise ConfigError("experiment seeds must be exactly [42, 43, 44]")
        if self.steps != PROTOCOL_STEPS:
            raise ConfigError("experiment steps must be exactly 64")
        if any(x < 0 or x > 1 for x in self.normalized_progress):
            raise ConfigError("normalized progress must lie in [0, 1]")
        progress_key = "attention_entropy_paper" if (
            self.type == "attention_entropy" and self.id == "attention_entropy"
        ) else self.type
        expected_progress = PROTOCOL_PROGRESS.get(progress_key)
        if expected_progress is not None and self.normalized_progress != expected_progress:
            raise ConfigError(f"normalized progress does not match the frozen {self.type} protocol")
        self.scoring.validate()
        if is_paper_experiment(self):
            self._validate_paper_protocol()
        elif self.scoring != ScoringConfig():
            raise ConfigError("scoring settings do not match the frozen protocol")

    def _validate_paper_protocol(self) -> None:
        if self.id != self.type and self.id != "attention_entropy":
            raise ConfigError("corrected paper config id and type must be identical")
        frozen = ScoringConfig(
            attender_rows="last",
            receiver_span="source_argmax",
            top_k=1,
            primary_relation="object_to_verb",
            primary_visibility="both_visible",
        )
        if self.scoring != frozen:
            raise ConfigError("corrected paper scoring must use the old single-source argmax")
        if self.settings.get("relative_depths") not in (
            None,
            {"early": 0.2, "middle": 0.5, "late": 0.9},
        ):
            raise ConfigError("relative depths must be early=.20, middle=.50, late=.90")
        forbidden = {"dev", "development", "permutation", "permutations", "holm"}
        serialized = repr(self.settings).lower()
        if any(name in serialized for name in forbidden):
            raise ConfigError("corrected paper settings cannot reference dev, permutations, or Holm")
        if self.type in {
            "relation_head_receiver_prediction_over_diffusion_time",
            "final_token_prediction_by_layer",
            "prediction_before_unmasking_timing_analysis",
            "attention_heatmaps_and_trajectories",
            "multilingual_relation_head_transfer",
        } and len(self.normalized_progress) != 64:
            raise ConfigError("corrected time-resolved experiments must contain all 64 steps")
        if self.id == "attention_entropy" and len(self.normalized_progress) != 64:
            raise ConfigError("Attention Entropy must contain all 64 steps")


@dataclass(frozen=True)
class RuntimeConfig:
    results_root: str = "results"
    run_id: str | None = None
    resume: bool = False
    dry_run: bool = False
    selection_lock: str | None = None


@dataclass(frozen=True)
class RunConfig:
    schema_version: str = SCHEMA_VERSION
    track: str = "confirmatory_ewt"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(f"expected schema_version {SCHEMA_VERSION}")
        if self.track not in TRACKS:
            raise ConfigError(f"track must be one of {TRACKS}")
        self.model.validate()
        self.dataset.validate()
        self.experiment.validate()
        if self.track == "confirmatory_ewt" and self.dataset.id != "ewt":
            raise ConfigError("confirmatory_ewt track requires the EWT dataset")
        if self.track == "external_treebank_transfer" and self.dataset.id == "ewt":
            raise ConfigError("external transfer requires a non-EWT dataset")
        required = {
            "head_search": "attentions",
            "time_curve": "attentions",
            "attention_entropy": "attentions",
            "pos_probe": "hidden_states",
            "logit_lens": "logits",
            "relation_head_receiver_prediction": "attentions",
            "relation_head_receiver_prediction_over_diffusion_time": "attentions",
            "pos_token_class_linear_probes": "hidden_states",
            "final_token_prediction_by_layer": "hidden_states",
            "prediction_before_unmasking_timing_analysis": "logits",
            "direct_logit_attribution": "hidden_states",
            "matched_relation_head_ablation": "logits",
            "attention_heatmaps_and_trajectories": "attentions",
            "multilingual_relation_head_transfer": "attentions",
        }
        if self.experiment.type not in required:
            raise ConfigError(f"unsupported experiment type: {self.experiment.type!r}")
        capability = required.get(self.experiment.type)
        if capability and not getattr(self.model.capabilities, capability):
            raise ConfigError(f"experiment {self.experiment.type!r} requires model capability {capability!r}")
        if self.experiment.type == "multilingual_relation_head_transfer" and self.track != (
            "external_treebank_transfer"
        ):
            raise ConfigError("multilingual transfer requires the external_treebank_transfer track")

    @classmethod
    def load_files(
        cls,
        model: str | Path,
        dataset: str | Path,
        experiment: str | Path,
        *,
        track: str | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> RunConfig:
        model_raw = _read_yaml(model)
        dataset_raw = _read_yaml(dataset)
        experiment_raw = _read_yaml(experiment)
        selected_track = track or experiment_raw.pop("track", "confirmatory_ewt")
        cfg = cls(
            track=selected_track,
            model=_strict_dataclass(ModelConfig, model_raw, "model"),
            dataset=_strict_dataclass(DatasetConfig, dataset_raw, "dataset"),
            experiment=_strict_dataclass(ExperimentConfig, experiment_raw, "experiment"),
            runtime=runtime or RuntimeConfig(),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        """Load a resolved configuration with the same strict schema checks."""
        cfg = _strict_dataclass(cls, raw, "config")
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        atomic_yaml(path, self.to_dict())


T = TypeVar("T")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return raw


def _reject_unknown(cls: type, raw: dict[str, Any], where: str) -> None:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown {where} field(s): {', '.join(unknown)}")


def _nested_dataclass(field_type: Any) -> type | None:
    if isinstance(field_type, type) and is_dataclass(field_type):
        return field_type
    origin = get_origin(field_type)
    if origin is not None:
        for arg in get_args(field_type):
            if isinstance(arg, type) and is_dataclass(arg):
                return arg
    return None


def _strict_dataclass(cls: type[T], raw: dict[str, Any], where: str) -> T:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")
    _reject_unknown(cls, raw, where)
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in raw:
            raise ConfigError(f"missing required {where} field: {item.name}")
        value = raw[item.name]
        nested = _nested_dataclass(hints.get(item.name, item.type))
        values[item.name] = _strict_dataclass(nested, value, f"{where}.{item.name}") if nested else value
    return cls(**values)


def atomic_yaml(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)

"""Strict, serializable configuration for rigorous and legacy tracks."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

SCHEMA_VERSION = "dlmrel-config-v2"
TRACKS = (
    "legacy_reproduction",
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
        for split, digest in self.checksums.items():
            if split not in {"train", "dev", "test"} or not digest.startswith("sha256:"):
                raise ConfigError("dataset checksums must be sha256:<hex> for train/dev/test")


@dataclass
class TreebankConfig:
    """Compatibility config used only by the explicitly legacy loader."""

    treebanks: list[str] = field(default_factory=lambda: ["UD_English-EWT"])
    cache_dir: str = "data/ud"
    n_select: int | None = 4000
    n_dev: int | None = 1000
    n_test: int | None = 1000
    max_seq_len: int = 128
    min_seq_len: int = 4
    shuffle: bool = True
    seed: int = 42
    skip_multiword: bool = True
    require_full_alignment: bool = True
    common_pool_models: list[str] = field(default_factory=list)
    dedupe_by_text: bool = True


@dataclass(frozen=True)
class CapabilityConfig:
    logits: bool = False
    hidden_states: bool = False
    attentions: bool = False
    native_timestep: bool = False
    head_residuals: bool = False
    head_ablation: bool = False
    native_generation: bool = False


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


@dataclass(frozen=True)
class ScoringConfig:
    attender_rows: str = "mean"
    receiver_span: str = "sum"
    exclude_special: bool = True
    exclude_self: bool = True
    top_k: int = 5
    tie_break: str = "layer_then_head"
    primary_relation: str = "object_to_verb"
    primary_visibility: str = "both_masked"

    def validate(self) -> None:
        if self.attender_rows not in {"mean", "first", "last"}:
            raise ConfigError("attender_rows must be mean, first, or last")
        if self.receiver_span not in {"sum", "mean", "max"}:
            raise ConfigError("receiver_span must be sum, mean, or max")
        if self.top_k < 1:
            raise ConfigError("top_k must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    id: str = "head_search"
    type: str = "head_search"
    trajectory: str = "teacher_forced_gold"
    steps: int = 64
    normalized_progress: list[float] = field(
        default_factory=lambda: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    shard_size: int = 100

    def validate(self) -> None:
        if self.trajectory not in {"teacher_forced_gold", "native_generated", "static_final"}:
            raise ConfigError("unknown trajectory")
        if not self.seeds or self.steps < 1 or self.shard_size < 1:
            raise ConfigError("experiment seeds, steps, and shard_size must be positive")
        if any(x < 0 or x > 1 for x in self.normalized_progress):
            raise ConfigError("normalized progress must lie in [0, 1]")
        self.scoring.validate()


@dataclass(frozen=True)
class RuntimeConfig:
    results_root: str = "results"
    run_id: str | None = None
    resume: bool = False
    dry_run: bool = False
    workers: int = 1
    selection_lock: str | None = None


@dataclass
class DiffusionConfig:
    """Compatibility surface for the legacy experiment implementations."""

    steps: int = 64
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    include_bos: bool = True
    exclude_bos: bool = True
    exclude_self: bool = True
    attender_token: str = "last"
    min_masked_positions: int = 25
    timestep_stride: int = 1
    n_curve_sentences: int | None = None
    timesteps: list[int] | None = None
    n_probe_sentences: int | None = 400
    probe_layer_stride: int = 4


@dataclass
class AnalysisConfig:
    offset_range: tuple[int, int] = (-15, 15)
    n_bootstrap: int = 10_000
    ci: float = 0.95
    distance_bins: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 100])


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
        if self.model.family == "gpt2" and self.experiment.trajectory != "static_final":
            raise ConfigError("GPT-2 supports static_final experiments only")
        required = {
            "head_search": "attentions",
            "time_curve": "attentions",
            "attention_entropy": "attentions",
            "pos_probe": "hidden_states",
            "logit_lens": "logits",
            "dla": "head_residuals",
            "ablation": "head_ablation",
        }
        capability = required.get(self.experiment.type)
        if capability and not getattr(self.model.capabilities, capability):
            raise ConfigError(f"experiment {self.experiment.type!r} requires model capability {capability!r}")

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        atomic_yaml(path, self.to_dict())


@dataclass
class Config:
    """Legacy aggregate retained for historical reproduction only."""

    treebank: TreebankConfig = field(default_factory=TreebankConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    out_dir: str = "results/run"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = _read_yaml(path)
        _reject_unknown(cls, raw, "legacy config")
        return cls(
            treebank=_strict_dataclass(TreebankConfig, raw.get("treebank", {}), "treebank"),
            model=_strict_dataclass(ModelConfig, raw.get("model", {}), "model"),
            diffusion=_strict_dataclass(DiffusionConfig, raw.get("diffusion", {}), "diffusion"),
            analysis=_strict_dataclass(AnalysisConfig, raw.get("analysis", {}), "analysis"),
            out_dir=raw.get("out_dir", "results/run"),
        )

    def save(self, path: str | Path) -> None:
        atomic_yaml(path, asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
            if item.default is MISSING and item.default_factory is MISSING:
                raise ConfigError(f"missing required {where} field: {item.name}")
            continue
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

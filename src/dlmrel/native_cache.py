"""Shared native-trajectory cache for the paper's final-token and timing experiments.

`final_token_prediction_by_layer` and `prediction_before_unmasking_timing_analysis`
use identical generation settings -- same prompt manifest, seeds, steps,
generation length, temperature, top-p, and reveal policy -- but each has always
generated its own trajectories inside its own run directory, so the same 64-step
native generation happens twice. Timing analysis needs no model at all once a
trajectory exists.

This cache lives outside any single experiment's run directory, keyed by a
scientific identity hash, with one file per (prompt, seed) so an interrupted
generation resumes only the missing pairs. Any identity mismatch -- model,
tokenizer, remote-code revision, prompt manifest, or a sampler setting -- is a
cache miss, never a silent reuse.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import atomic_json, canonical_hash

CACHE_SCHEMA = "dlmrel-native-trajectory-cache-v1"


@dataclass(frozen=True)
class NativeCacheIdentity:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    remote_code_revision: str | None
    prompt_manifest_hash: str
    steps: int
    generation_length: int
    temperature: float
    top_p: float
    reveal_policy: str
    prediction_offset: int

    def key_hash(self) -> str:
        return canonical_hash(asdict(self))


class NativeTrajectoryCache:
    """One shared directory per scientific identity; one file per (prompt, seed)."""

    def __init__(self, cache_root: str | Path, identity: NativeCacheIdentity):
        self.identity = identity
        self.directory = Path(cache_root) / identity.key_hash()
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.directory / "identity.json",
            {"schema_version": CACHE_SCHEMA, **asdict(identity)},
        )

    def _paths(self, prompt_id: str, seed: int) -> tuple[Path, Path]:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", str(prompt_id)).strip("-")
        path = self.directory / f"{safe}__seed-{seed}.parquet"
        return path, path.with_suffix(".meta.json")

    def _expected_metadata(self, prompt_id: str, seed: int) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA,
            "identity_hash": self.identity.key_hash(),
            "prompt_id": str(prompt_id),
            "seed": seed,
        }

    def load(self, prompt_id: str, seed: int) -> pd.DataFrame | None:
        path, meta_path = self._paths(prompt_id, seed)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = self._expected_metadata(prompt_id, seed)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            return None
        if metadata.get("row_count") != len(frame):
            return None
        return frame

    def store(self, prompt_id: str, seed: int, frame: pd.DataFrame) -> None:
        path, meta_path = self._paths(prompt_id, seed)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        atomic_json(
            meta_path,
            {**self._expected_metadata(prompt_id, seed), "row_count": len(frame)},
        )

    def get_or_generate(
        self,
        examples: list,
        seeds: list[int],
        generate_one: Callable[[Any, int], pd.DataFrame],
    ) -> pd.DataFrame:
        """Return one concatenated frame, generating only the (prompt, seed) pairs missing."""
        frames = []
        for seed in seeds:
            for example in examples:
                cached = self.load(example.sentence_id, seed)
                if cached is not None:
                    frames.append(cached)
                    continue
                frame = generate_one(example, seed)
                self.store(example.sentence_id, seed, frame)
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_or_generate_batched(
        self,
        examples: list,
        seeds: list[int],
        generate_many: Callable[[list[Any], int], pd.DataFrame],
        *,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate missing pairs in prompt batches while retaining pair-level resume files."""
        if batch_size < 1:
            raise ValueError("native prompt batch_size must be positive")
        frames = []
        for seed in seeds:
            missing = []
            for example in examples:
                cached = self.load(example.sentence_id, seed)
                if cached is None:
                    missing.append(example)
                else:
                    frames.append(cached)
            for start in range(0, len(missing), batch_size):
                current = missing[start : start + batch_size]
                generated = generate_many(current, seed)
                if len(generated) != len(current) or "prompt_id" not in generated:
                    raise RuntimeError("batched native generation returned an invalid frame")
                for example in current:
                    frame = generated[
                        generated["prompt_id"].astype(str) == str(example.sentence_id)
                    ].copy()
                    if len(frame) != 1:
                        raise RuntimeError(
                            "batched native generation must return exactly one row per prompt"
                        )
                    self.store(example.sentence_id, seed, frame)
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

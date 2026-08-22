from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from dlmrel.artifacts import atomic_json, initialize_run
from dlmrel.checkpoints import CheckpointIdentity, SentenceCheckpointStore
from dlmrel.config import RunConfig, RuntimeConfig
from dlmrel.experiments.paper_entropy import (
    ATTENTION_CACHE_CHUNK_SIZE,
    ATTENTION_CACHE_STAGE,
    _load_attention_cache,
)
from dlmrel.paper_protocol import map_relative_depths

ROOT = Path(__file__).parents[1]


def _cache_rows(chunk, seed, depths):
    timesteps = range(64) if seed == 42 else range(1, 63)
    rows = []
    for example in chunk:
        for timestep in timesteps:
            for depth in depths:
                rows.append(
                    {
                        "sentence_id": example.sentence_id,
                        "treebank": "ewt",
                        "seed": seed,
                        "timestep": timestep,
                        "normalized_progress": timestep / 63,
                        **depth,
                        "layer": depth["actual_layer_index"],
                        "head": 0,
                        "entropy_normalized": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def test_completed_time_run_entropy_cache_is_validated_and_reused(tmp_path):
    source_cfg = RunConfig.load_files(
        ROOT / "configs/models/fake.yaml",
        ROOT / "configs/datasets/ewt.yaml",
        ROOT / "configs/experiments/relation_head_receiver_prediction_over_diffusion_time.yaml",
        runtime=RuntimeConfig(export_attention_cache=True),
    )
    entropy_cfg = RunConfig.load_files(
        ROOT / "configs/models/fake.yaml",
        ROOT / "configs/datasets/ewt.yaml",
        ROOT / "configs/experiments/attention_entropy.yaml",
        runtime=RuntimeConfig(attention_cache=str(tmp_path / "source")),
    )
    source = tmp_path / "source"
    manifests = {"test": "frozen-test-hash"}
    initialize_run(source, source_cfg.to_dict(), "source", manifests)
    examples = [SimpleNamespace(sentence_id="s1"), SimpleNamespace(sentence_id="s2")]
    depths = map_relative_depths(2)
    store = SentenceCheckpointStore(source, chunk_size=ATTENTION_CACHE_CHUNK_SIZE)
    for seed in source_cfg.experiment.seeds:
        identity = CheckpointIdentity(
            stage=ATTENTION_CACHE_STAGE,
            seed=seed,
            normalized_progress=-1.0,
            timestep=-1,
        )
        store.run(
            examples,
            identity,
            lambda chunk, _start, current_seed=seed: _cache_rows(
                chunk, current_seed, depths
            ),
        )
    atomic_json(
        source / "summary.json",
        {
            "completion_status": "complete",
            "attention_entropy_cache_exported": True,
            "attention_entropy_cache_relative_depths": depths,
        },
    )

    frames = _load_attention_cache(source, examples, entropy_cfg, depths, manifests)

    assert [set(frame["seed"]) for frame in frames] == [{42}, {43}, {44}]
    assert [set(frame["timestep"]) for frame in frames] == [
        set(range(64)),
        set(range(1, 63)),
        set(range(1, 63)),
    ]

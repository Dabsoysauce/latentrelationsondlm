"""Attention concentration over the shared masking schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import attentions_at_time
from .shared import write_frames


def run(model, tokenizer, cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    store = SentenceCheckpointStore(run_dir)
    frames = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            timestep = round(progress * (cfg.experiment.steps - 1))
            identity = CheckpointIdentity(
                stage="attention-entropy-test",
                seed=seed,
                normalized_progress=progress,
                timestep=timestep,
            )
            frames.append(
                store.run(
                    examples,
                    identity,
                    lambda chunk, _start, current_seed=seed, current_progress=progress: (
                        entropy_rows(
                            model,
                            tokenizer,
                            chunk,
                            cfg,
                            seed=current_seed,
                            progress=current_progress,
                        )
                    ),
                )
            )
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    group = ["seed", "treebank", "timestep", "normalized_progress", "layer", "head"]
    per_seed = raw.groupby(group, as_index=False).mean(numeric_only=True)
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics = per_seed.groupby(group[1:], as_index=False).agg(
        entropy_mean=("entropy", "mean"),
        entropy_normalized=("entropy_normalized", "mean"),
        entropy_no_bos=("entropy_no_bos", "mean"),
        bos_sink_mass=("bos_sink_mass", "mean"),
        n_seeds=("seed", "nunique"),
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return {"n_rows": len(raw), "n_sentences": len(examples)}


def entropy_rows(model, tokenizer, examples, cfg: RunConfig, *, seed: int, progress: float):
    timestep = round(progress * (cfg.experiment.steps - 1))
    rows = []
    for example in examples:
        attentions, state = attentions_at_time(
            model, tokenizer, example.text, timestep, cfg.experiment.steps, seed, True
        )
        for layer, attention in enumerate(attentions):
            probability = attention[0].float()
            probability /= probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            no_bos = probability.clone()
            no_bos[:, :, 0] = 0
            no_bos /= no_bos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy_no_bos = -(no_bos * no_bos.clamp_min(1e-12).log()).sum(dim=-1)
            valid_keys = probability.shape[-1]
            normalization = float(np.log(valid_keys)) if valid_keys > 1 else None
            for head in range(probability.shape[0]):
                mean_entropy = float(entropy[head].mean())
                rows.append(
                    {
                        "sentence_id": example.sentence_id,
                        "treebank": example.source,
                        "seed": seed,
                        "timestep": timestep,
                        "normalized_progress": progress,
                        "layer": layer,
                        "head": head,
                        "entropy": mean_entropy,
                        "entropy_normalized": (
                            mean_entropy / normalization if normalization is not None else 0.0
                        ),
                        "entropy_no_bos": float(entropy_no_bos[head].mean()),
                        "bos_sink_mass": float(probability[head, :, 0].mean()),
                        "valid_key_count": valid_keys,
                        "n_masked": state.n_masked,
                    }
                )
    return pd.DataFrame(rows)

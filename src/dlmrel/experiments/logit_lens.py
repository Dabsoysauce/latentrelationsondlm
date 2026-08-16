"""Decode gold tokens from intermediate hidden states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RunConfig
from ..data import load_manifest_examples
from ..diffusion import states_at_time, tokenize
from .shared import write_frames


def run(model, tokenizer, cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    store = SentenceCheckpointStore(run_dir)
    frames = []
    for seed in cfg.experiment.seeds:
        for progress in cfg.experiment.normalized_progress:
            timestep = round(progress * (cfg.experiment.steps - 1))
            identity = CheckpointIdentity(
                stage="logit-lens-test",
                seed=seed,
                normalized_progress=progress,
                timestep=timestep,
            )
            frames.append(
                store.run(
                    examples,
                    identity,
                    lambda chunk, _start, current_seed=seed, current_progress=progress: (
                        logit_lens_rows(
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
    parity_error = (
        float(raw.pop("_final_depth_parity_error").max())
        if "_final_depth_parity_error" in raw
        else float("nan")
    )
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    group = [
        "seed",
        "treebank",
        "timestep",
        "normalized_progress",
        "depth",
        "position_state",
    ]
    per_seed = raw.groupby(group, as_index=False).agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        mrr=("mrr", "mean"),
        mean_rank=("rank", "mean"),
        n_positions=("rank", "size"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    metrics = per_seed.groupby(group[1:], as_index=False).agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        mrr=("mrr", "mean"),
        mean_rank=("mean_rank", "mean"),
        n_positions=("n_positions", "sum"),
        n_seeds=("seed", "nunique"),
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return {
        "n_rows": len(raw),
        "n_sentences": len(examples),
        "final_depth_max_abs_parity_error": parity_error,
    }


def logit_lens_rows(model, tokenizer, examples, cfg: RunConfig, *, seed: int, progress: float):
    timestep = round(progress * (cfg.experiment.steps - 1))
    rows = []
    for example in examples:
        _, hidden_states, state = states_at_time(
            model, tokenizer, example.text, timestep, cfg.experiment.steps, seed, True
        )
        true_ids, _ = tokenize(tokenizer, example.text, state.input_ids.device, True)
        direct = model.get_logits(hidden_states[-1]).float()
        lens = model.get_lm_head()(hidden_states[-1]).float()
        parity_error = float((direct - lens).abs().max())
        for depth, hidden in enumerate(hidden_states):
            transformed = hidden if depth == len(hidden_states) - 1 else model.get_final_norm()(hidden)
            logits = model.get_lm_head()(transformed)[0].float()
            order = logits.argsort(dim=-1, descending=True)
            for position in range(logits.shape[0]):
                rank = int((order[position] == true_ids[0, position]).nonzero()[0]) + 1
                rows.append(
                    {
                        "sentence_id": example.sentence_id,
                        "treebank": example.source,
                        "seed": seed,
                        "timestep": timestep,
                        "normalized_progress": progress,
                        "depth": depth,
                        "position": position,
                        "position_state": "visible" if state.is_visible[position] else "masked",
                        "target_token_id": int(true_ids[0, position]),
                        "top1": int(rank == 1),
                        "top5": int(rank <= 5),
                        "rank": rank,
                        "mrr": 1.0 / rank,
                        "target_logit": float(logits[position, true_ids[0, position]]),
                        "_final_depth_parity_error": parity_error,
                    }
                )
    return pd.DataFrame(rows)

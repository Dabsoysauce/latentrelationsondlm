from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config, DiffusionConfig
from ..data import examples_for_split
from ..diffusion import states_at_time
from ..relations import Example


def _row_entropy(rows: torch.Tensor) -> torch.Tensor:
    p = rows.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=-1)


def attention_entropy(
    model,
    tokenizer,
    examples: list[Example],
    cfg: DiffusionConfig,
    diffusion_time: int | None = None,
    log_every: int = 100,
) -> pd.DataFrame:
    t = cfg.steps - 1 if diffusion_time is None else diffusion_time
    totals: dict[str, np.ndarray] = {}
    n_rows = 0

    for i, example in enumerate(examples):
        attentions, _, state = states_at_time(
            model,
            tokenizer,
            example.text,
            diffusion_time=t,
            steps=cfg.steps,
            seed=cfg.seed,
            include_bos=cfg.include_bos,
        )
        seq_len = len(state.tokens)
        if seq_len < 3:
            continue

        stacked = torch.stack([a[0].float() for a in attentions])
        raw = _row_entropy(stacked)

        no_sink = stacked.clone()
        no_sink[:, :, :, 0] = 0.0
        trimmed = _row_entropy(no_sink)

        if not totals:
            shape = (stacked.shape[0], stacked.shape[1])
            totals = {
                "entropy": np.zeros(shape),
                "entropy_norm": np.zeros(shape),
                "entropy_no_sink": np.zeros(shape),
                "sink_mass": np.zeros(shape),
            }

        denom = float(np.log(seq_len))
        totals["entropy"] += raw.mean(dim=-1).cpu().numpy()
        totals["entropy_norm"] += (raw / denom).mean(dim=-1).cpu().numpy()
        totals["entropy_no_sink"] += trimmed.mean(dim=-1).cpu().numpy()
        totals["sink_mass"] += stacked[:, :, :, 0].mean(dim=-1).cpu().numpy()
        n_rows += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"[entropy] {i + 1}/{len(examples)} sentences", flush=True)

    if not n_rows:
        return pd.DataFrame()

    rows = []
    n_layers, n_heads = totals["entropy"].shape
    for layer in range(n_layers):
        for head in range(n_heads):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "entropy": totals["entropy"][layer, head] / n_rows,
                    "entropy_norm": totals["entropy_norm"][layer, head] / n_rows,
                    "entropy_no_sink": totals["entropy_no_sink"][layer, head] / n_rows,
                    "sink_mass": totals["sink_mass"][layer, head] / n_rows,
                    "n_sentences": n_rows,
                    "diffusion_time": t,
                }
            )
    return pd.DataFrame(rows)


def run(model, tokenizer, cfg: Config, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    examples = examples_for_split(cfg, tokenizer, "test")
    if cfg.diffusion.n_probe_sentences is not None:
        examples = examples[: cfg.diffusion.n_probe_sentences]

    if cfg.diffusion.timesteps:
        timesteps = cfg.diffusion.timesteps
    else:
        timesteps = sorted(
            {
                round(progress * (cfg.diffusion.steps - 1))
                for progress in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
            }
        )
    tables = [
        attention_entropy(model, tokenizer, examples, cfg.diffusion, timestep) for timestep in timesteps
    ]
    table = pd.concat([item for item in tables if not item.empty], ignore_index=True)
    table["normalized_progress"] = table["diffusion_time"] / max(cfg.diffusion.steps - 1, 1)
    table.to_csv(out / "attention_entropy.csv", index=False)
    print(table.groupby("layer")[["entropy_norm", "entropy_no_sink", "sink_mass"]].mean().to_string())

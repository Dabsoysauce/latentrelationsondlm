from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..config import Config, DiffusionConfig
from ..data import examples_for_split
from ..diffusion import states_at_time, tokenize
from ..relations import Example


def _depth_hidden(hidden_states, depth: int, norm):
    last = len(hidden_states) - 1
    h = hidden_states[depth]
    return h if depth == last else norm(h)


def logit_lens(
    model,
    tokenizer,
    examples: list[Example],
    cfg: DiffusionConfig,
    diffusion_times: list[int] | None = None,
    log_every: int = 50,
) -> pd.DataFrame:
    times = diffusion_times or [8, 16, 32]
    norm = model.get_final_norm()
    head = model.get_lm_head()

    correct: dict[tuple[int, int, bool], int] = defaultdict(int)
    seen: dict[tuple[int, int, bool], int] = defaultdict(int)
    n_depths = 0

    for t in times:
        for i, example in enumerate(examples):
            _, hidden_states, state = states_at_time(
                model,
                tokenizer,
                example.text,
                diffusion_time=t,
                steps=cfg.steps,
                seed=cfg.seed,
                include_bos=cfg.include_bos,
            )
            true_ids, _ = tokenize(
                tokenizer, example.text, state.input_ids.device, cfg.include_bos
            )
            if true_ids.shape[1] != state.input_ids.shape[1]:
                continue

            n_depths = len(hidden_states)
            visible = state.is_visible

            for depth in range(n_depths):
                h = _depth_hidden(hidden_states, depth, norm)
                pred = head(h)[0].argmax(dim=-1)
                hit = (pred == true_ids[0]).cpu().numpy()
                for pos in range(len(visible)):
                    key = (t, depth, bool(visible[pos]))
                    seen[key] += 1
                    correct[key] += int(hit[pos])

            if log_every and (i + 1) % log_every == 0:
                print(f"[logit-lens] t={t}: {i + 1}/{len(examples)} sentences", flush=True)

    rows = []
    for t in times:
        for depth in range(n_depths):
            for vis in (True, False):
                n = seen[(t, depth, vis)]
                if not n:
                    continue
                rows.append(
                    {
                        "diffusion_time": t,
                        "depth": depth,
                        "position_state": "visible" if vis else "masked",
                        "accuracy": correct[(t, depth, vis)] / n,
                        "n_positions": n,
                    }
                )
    return pd.DataFrame(rows)


def run(model, tokenizer, cfg: Config, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    examples = examples_for_split(cfg, tokenizer, "test")
    if cfg.diffusion.n_probe_sentences is not None:
        examples = examples[: cfg.diffusion.n_probe_sentences]

    table = logit_lens(model, tokenizer, examples, cfg.diffusion, cfg.diffusion.timesteps)
    table.to_csv(out / "logit_lens.csv", index=False)
    print(
        table.pivot_table(
            index="depth", columns=["diffusion_time", "position_state"], values="accuracy"
        ).to_string()
    )

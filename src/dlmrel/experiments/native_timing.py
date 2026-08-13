"""Exploratory native-generation discovery/reveal timing records."""

from __future__ import annotations

import pandas as pd


def timing_rows(
    trajectories: list[dict],
) -> pd.DataFrame:
    """Normalize adapter-native events without mixing them with UD trajectories.

    Each event must supply prompt, generated token, model-native step, found
    step, reveal step, seed, sampler settings, and optional parser confidence.
    """
    required = {
        "prompt",
        "generated_token",
        "native_step",
        "found_step",
        "reveal_step",
        "seed",
        "sampler_settings",
    }
    rows = []
    for event in trajectories:
        missing = required - set(event)
        if missing:
            raise ValueError(f"native timing event missing: {sorted(missing)}")
        row = dict(event)
        row["trajectory"] = "native_generated"
        row.setdefault("parser_confidence", None)
        rows.append(row)
    return pd.DataFrame(rows)

"""Paired causal head-ablation measurements."""

from __future__ import annotations

import pandas as pd


def paired_ablation_rows(adapter, instances: list[dict], *, layer: int, head: int) -> pd.DataFrame:
    adapter.capabilities.require("head_ablation")
    rows = []
    for instance in instances:
        delta = adapter.ablation_delta(
            instance["input_ids"],
            layer,
            head,
            instance["target_position"],
            instance["target_token"],
        )
        rows.append(
            {
                "sentence_id": instance["sentence_id"],
                "instance_id": instance["instance_id"],
                "layer": layer,
                "head": head,
                "query_position": instance["target_position"],
                "output_token_id": instance["target_token"],
                "logit_delta": delta,
            }
        )
    return pd.DataFrame(rows)

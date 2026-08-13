"""Direct-logit-attribution semantics shared across capable adapters."""

from __future__ import annotations

import torch


def direct_logit_attribution(
    head_output: torch.Tensor,
    unembedding: torch.Tensor,
    *,
    query_position: int,
    output_token_id: int,
) -> float:
    """Score a head write at the query for an explicit output token.

    Receiver/key spans are deliberately not accepted as the output target: the
    caller must name the token whose logit is being explained.
    """
    if head_output.ndim != 2:
        raise ValueError("head_output must have shape [sequence, hidden]")
    if unembedding.ndim != 2:
        raise ValueError("unembedding must have shape [hidden, vocabulary]")
    return float(head_output[query_position] @ unembedding[:, output_token_id])

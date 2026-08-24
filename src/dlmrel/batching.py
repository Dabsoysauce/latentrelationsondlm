"""Execution-only batching helpers with conservative CUDA OOM recovery."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

import torch

T = TypeVar("T")
R = TypeVar("R")


def _is_memory_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def adaptive_forward_batches(
    items: Sequence[T],
    forward: Callable[[Sequence[T]], R],
    *,
    initial_size: int,
    maximum_size: int | None = None,
) -> Iterator[tuple[int, Sequence[T], R]]:
    """Forward ordered slices, grow after success, and retry smaller after OOM.

    No completed slice is replayed. Non-memory errors are never swallowed. The
    yielded start offsets and item order are identical to fixed-size iteration,
    so callers can reconstruct their original scientific row order.
    """
    if initial_size < 1:
        raise ValueError("initial batch size must be positive")
    ceiling = initial_size if maximum_size is None else maximum_size
    if ceiling < initial_size:
        raise ValueError("maximum batch size cannot be smaller than initial batch size")
    start = 0
    size = min(initial_size, ceiling)
    while start < len(items):
        current = items[start : start + size]
        try:
            result = forward(current)
        except RuntimeError as error:
            if not _is_memory_error(error) or len(current) == 1:
                raise
            ceiling = max(1, len(current) // 2)
            size = ceiling
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        yield start, current, result
        start += len(current)
        size = min(ceiling, size * 2)

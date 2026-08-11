"""Do diffusion language models represent grammatical relations, or positions?

The measurement this package exists to get right: an attention head that always
points a fixed number of tokens away solves any dependency whose endpoints sit
at a near-constant distance. Distinguishing such a head from one that tracks
syntax requires a fixed-offset null model, which is what `dlmrel.nulls`
provides and what every reported number here is measured against.
"""

__version__ = "0.1.0"

from .config import RELATION_NAMES, Config  # noqa: F401

"""Dream-7B figures, reusing the publication figure code from make_figures.py.

Generates Dream-only variants of fig1/fig2/fig3, and optionally a side-by-side
with DiffuGPT-S — but only when told the two runs share a sentence pool, since
the 3-model common pool may differ from the 2-model pool the existing
DiffuGPT-S results were computed on. Comparability is a property of the splits,
not the plot, so the flag is explicit rather than assumed.

Usage:
    python scripts/make_dream_figures.py \
        --dream results/dream-7b-ewt --out figures-dream
    python scripts/make_dream_figures.py \
        --dream results/dream-7b-ewt --small results/diffugpt-s-ewt \
        --pools-match --out figures-dream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from make_figures import (
        fig_distance_distributions,
        fig_head_profiles,
        fig_head_vs_null,
        set_style,
    )
except ImportError as exc:  # pragma: no cover - depends on an unmerged PR
    raise SystemExit(
        "scripts/make_figures.py is missing. It belongs to the open figures PR "
        "(branch `figures`, PR #5) and is deliberately not duplicated here. "
        "Merge that PR, or fetch the file with:\n"
        "    git show origin/figures:scripts/make_figures.py > scripts/make_figures.py\n"
        f"(underlying error: {exc})"
    ) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dream", type=Path, default=Path("results/dream-7b-ewt"))
    ap.add_argument("--small", type=Path, default=None)
    ap.add_argument(
        "--pools-match",
        action="store_true",
        help="assert the runs share a sentence pool, enabling side-by-side panels",
    )
    ap.add_argument("--out", type=Path, default=Path("figures-dream"))
    args = ap.parse_args()

    set_style()
    print("dream figures:")

    fig_distance_distributions(args.dream / "relation_instances.csv", args.out)

    vs_null = {"Dream-7B (784 heads)": args.dream / "head_vs_null.csv"}
    profiles = {"Dream-7B": args.dream / "head_profiles.csv"}
    if args.small is not None and args.pools_match:
        vs_null = {
            "DiffuGPT-S (144 heads)": args.small / "head_vs_null.csv",
            **vs_null,
        }
        profiles = {"DiffuGPT-S": args.small / "head_profiles.csv", **profiles}
    elif args.small is not None:
        print("  note: --small given without --pools-match; plotting Dream alone")

    fig_head_vs_null(vs_null, args.out)
    fig_head_profiles(profiles, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

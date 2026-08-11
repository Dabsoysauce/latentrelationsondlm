import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    from make_figures import (
        fig_distance_distributions,
        fig_head_profiles,
        fig_head_vs_null,
        set_style,
    )
except ImportError as exc:
    raise SystemExit(
        "scripts/make_figures.py is missing. It comes from the figures branch:\n"
        "    git show origin/figures:scripts/make_figures.py > scripts/make_figures.py\n"
        f"({exc})"
    ) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dream", type=Path, default=Path("results/dream-7b-ewt"))
    ap.add_argument("--small", type=Path, default=None)
    ap.add_argument("--pools-match", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("figures-dream"))
    args = ap.parse_args()

    set_style()

    fig_distance_distributions(args.dream / "relation_instances.csv", args.out)

    vs_null = {"Dream-7B (784 heads)": args.dream / "head_vs_null.csv"}
    profiles = {"Dream-7B": args.dream / "head_profiles.csv"}

    if args.small is not None and args.pools_match:
        vs_null = {"DiffuGPT-S (144 heads)": args.small / "head_vs_null.csv", **vs_null}
        profiles = {"DiffuGPT-S": args.small / "head_profiles.csv", **profiles}
    elif args.small is not None:
        print("--small given without --pools-match, plotting Dream alone")

    fig_head_vs_null(vs_null, args.out)
    fig_head_profiles(profiles, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

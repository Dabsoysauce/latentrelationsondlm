"""Command-line entry point for the shared experiment pipeline."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="dlmrel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-data", help="Prepare fixed Universal Dependencies splits")

    run_parser = subparsers.add_parser("run", help="Run one experiment on one model")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--experiment", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare completed model runs")
    compare_parser.add_argument("--experiment", required=True)

    args = parser.parse_args()
    raise SystemExit(
        f"The clean repository scaffold is ready, but '{args.command}' still needs the "
        "working implementation migrated from the backup branch."
    )


if __name__ == "__main__":
    main()

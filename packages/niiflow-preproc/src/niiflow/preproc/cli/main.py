"""Command-line entry point for preprocessing workflows.

The CLI parses command-line arguments, loads a preprocessing config file,
and dispatches to :mod:`niiflow.preproc.cli.commands`.

Available actions:

``execute``
    Build a run plan from ``run_inputs`` and execute it. Optionally save the
    generated plan before execution.

``plan``
    Build and save a run plan without executing preprocessing stages.

``execute-plan``
    Load and execute an existing saved run plan.

Pass ``--dry-run`` with any command to print the resolved run plan and exit
without saving or executing.
"""

from __future__ import annotations

__all__ = [
    "build_parser",
    "main",
]

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from niiflow.preproc.config.load import load_preproc_config
from niiflow.preproc.cli import commands


def build_parser() -> argparse.ArgumentParser:
    """Build the preprocessing command-line parser."""
    parser = argparse.ArgumentParser(
        prog="niiflow-preproc",
        description=(
            "Run config-driven preprocessing workflows. "
            "Use `--dry-run` with any command to print the resolved run plan "
            "without saving or executing."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full tracebacks on errors.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dry-run",
        action="store_true",
        help=("Print the resolved run plan and exit without saving or executing."),
    )

    execute_parser = subparsers.add_parser(
        "execute",
        parents=[common],
        help="Build a run plan from run_inputs and execute it.",
        description=(
            "Build a run plan from the config's `run_inputs`, optionally save "
            "the generated plan, and execute the planned preprocessing entries."
        ),
    )
    execute_parser.add_argument(
        "config",
        type=Path,
        metavar="PATH",
        help="Path to a preprocessing JSON or YAML config file.",
    )
    execute_parser.add_argument(
        "--save-plan",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path where the generated run plan should be saved. If omitted, "
            "`artifacts.plan_path` from the config is used when available."
        ),
    )

    plan_parser = subparsers.add_parser(
        "plan",
        parents=[common],
        help="Build and save a run plan without executing preprocessing.",
        description=(
            "Resolve run_inputs, discover active files, stage entries, and save "
            "a run plan without executing preprocessing stages."
        ),
    )
    plan_parser.add_argument(
        "config",
        type=Path,
        metavar="PATH",
        help="Path to a preprocessing JSON or YAML config file.",
    )
    plan_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path where the run plan should be saved. If omitted, "
            "`artifacts.plan_path` from the config is used."
        ),
    )

    execute_plan_parser = subparsers.add_parser(
        "execute-plan",
        parents=[common],
        help="Execute an existing saved run plan.",
        description=(
            "Load a saved run plan and execute it with the workflow configured "
            "by the provided config file. This does not rediscover files or "
            "restage entries."
        ),
    )
    execute_plan_parser.add_argument(
        "config",
        type=Path,
        metavar="PATH",
        help="Path to a preprocessing JSON or YAML config file.",
    )
    execute_plan_parser.add_argument(
        "plan_path",
        type=Path,
        metavar="PATH",
        help="Path to a saved .json or .duckdb run plan.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the preprocessing CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_preproc_config(args.config)

        if args.command == "execute":
            commands.execute(
                config,
                save_plan=args.save_plan,
                dry_run=args.dry_run,
            )
            return

        if args.command == "plan":
            commands.plan(config, output=args.output, dry_run=args.dry_run)
            return

        if args.command == "execute-plan":
            commands.execute_plan(
                config,
                plan_path=args.plan_path,
                dry_run=args.dry_run,
            )
            return

        parser.error(f"Unknown command: {args.command}")

    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

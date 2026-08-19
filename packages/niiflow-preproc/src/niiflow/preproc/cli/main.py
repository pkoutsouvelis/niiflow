"""Command-line entry point for preprocessing orchestration drivers.

The CLI parses a registered command name, loads a config file as that command's
keyword arguments, and dispatches via :func:`niiflow.preproc.cli.commands.run`.

Registered commands (see :data:`~niiflow.preproc.cli.commands.COMMANDS`):

``dynamic_workflow``
    Plan and/or execute a :class:`~niiflow.preproc.workflows.DynamicProcessingWorkflow`
    from driver kwargs in the config (``settings``, ``inputs``, ``from_plan``,
    ``plan_only``, ``dry_run``, save paths, etc.).
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

from niiflow.preproc.cli import commands
from niiflow.preproc.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the preprocessing command-line parser from the command registry."""
    command_names = ", ".join(sorted(commands.COMMANDS))
    parser = argparse.ArgumentParser(
        prog="niiflow-preproc",
        description=(
            "Run a registered preprocessing orchestration command with a "
            f"kwargs config file. Available commands: {command_names}."
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
        metavar="COMMAND",
        help=f"One of: {command_names}.",
    )

    for name in sorted(commands.COMMANDS):
        command_parser = subparsers.add_parser(
            name,
            help=f"Run `{name}` with kwargs from a config file.",
            description=(
                f"Load a JSON/YAML config and call `{name}(**config)`. "
                "All mode and path options belong in the config file."
            ),
        )
        command_parser.add_argument(
            "config",
            type=Path,
            metavar="PATH",
            help="Path to a JSON or YAML kwargs config file.",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the preprocessing CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        commands.run(args.command, config)
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

"""Command-line interface for niiflow-preproc."""

__all__ = [
    "COMMANDS",
    "build_parser",
    "main",
    "run",
]

from .commands import COMMANDS, run
from .main import build_parser, main

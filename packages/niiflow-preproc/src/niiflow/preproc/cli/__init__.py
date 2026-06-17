"""Command-line interface for niiflow-preproc."""

__all__ = [
    "build_parser",
    "execute",
    "execute_plan",
    "plan",
]

from .main import build_parser
from .commands import execute, execute_plan, plan

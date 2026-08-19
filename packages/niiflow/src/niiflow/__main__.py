"""Console entry point; `niiflow` has no `__init__.py` (namespace package)."""

from __future__ import annotations


def main() -> None:
    """Print a placeholder message until CLI subcommands are implemented."""
    print(
        "niiflow: import niiflow.train (training) or niiflow.preproc (offline preprocessing). "
        "CLI subcommands will be added here."
    )

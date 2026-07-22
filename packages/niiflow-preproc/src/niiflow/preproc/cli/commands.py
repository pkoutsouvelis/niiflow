"""Registry of CLI command callables.

Each registered name is a CLI subcommand. The config file for that command is passed as
``**kwargs`` to the callable.
"""

from __future__ import annotations

__all__ = [
    "COMMANDS",
    "run",
]

from collections.abc import Callable, Mapping
from typing import Any

from niiflow.preproc.workflows.dynamic_workflow import dynamic_workflow

COMMANDS: dict[str, Callable[..., Any]] = {
    "dynamic_workflow": dynamic_workflow,
}


def run(name: str, config: Mapping[str, Any]) -> Any:
    """Dispatch ``config`` as kwargs to the registered command ``name``."""
    try:
        fn = COMMANDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown command {name!r}; expected one of {sorted(COMMANDS)}"
        ) from exc
    return fn(**dict(config))

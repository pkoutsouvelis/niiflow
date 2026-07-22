"""Config loading utilities for preprocessing CLI commands."""

from __future__ import annotations

__all__ = [
    "load_config",
]

from pathlib import Path
from typing import Any

from niiflow.preproc.utils.file import read_json, read_yaml, resolve_path


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a JSON or YAML config file as a top-level mapping.

    The returned dict is intended to be passed as keyword arguments to a
    registered CLI command (e.g. :func:`~niiflow.preproc.workflows.dynamic_workflow`).

    Args:
        path: Path to a ``.json``, ``.yaml``, or ``.yml`` config file.

    Returns:
        Loaded config mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported.
        TypeError: If the loaded document is not a mapping.
    """
    resolved = resolve_path(path)

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    suffix = resolved.suffix.lower()

    if suffix == ".json":
        data = read_json(resolved)
    elif suffix in {".yaml", ".yml"}:
        data = read_yaml(resolved)
    else:
        raise ValueError(
            f"Unsupported config extension {suffix!r}; use '.json', '.yaml', "
            "or '.yml'"
        )

    if not isinstance(data, dict):
        raise TypeError(
            f"Config file must contain a mapping at the top level, got "
            f"{type(data).__name__}"
        )

    return data

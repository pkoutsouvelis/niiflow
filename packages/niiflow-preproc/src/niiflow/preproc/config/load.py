"""Config loading utilities for preprocessing."""

from __future__ import annotations

__all__ = [
    "load_preproc_config",
]

from pathlib import Path
from typing import cast

from niiflow.preproc.utils.file import read_json, read_yaml, resolve_path

from .types import PreprocConfig


def load_preproc_config(path: Path | str) -> PreprocConfig:
    """Load a preprocessing config from JSON or YAML.

    Args:
        path: Path to a ``.json``, ``.yaml``, or ``.yml`` config file.

    Returns:
        Loaded preprocessing config.

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

    return cast(PreprocConfig, data)

"""Stager discovery and instantiation for config-driven staging."""

from __future__ import annotations

__all__ = [
    "create_stager",
    "discover_stager_classes",
]

import importlib
import inspect
from typing import Any

from .stager import Stager

_STAGER_MODULES: tuple[str, ...] = ("niiflow.preproc.staging.file_stager",)


def discover_stager_classes() -> dict[str, type[Stager]]:
    """Return concrete :class:`Stager` types keyed by class name.

    Scans the modules listed in :data:`_STAGER_MODULES` and excludes the abstract base
    :class:`Stager`.
    """
    registry: dict[str, type[Stager]] = {}
    for module_name in _STAGER_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Stager:
                continue
            if not issubclass(obj, Stager):
                continue
            registry[name] = obj
    return registry


def _resolve_stager_class(name: str, registry: dict[str, type[Stager]]) -> type[Stager]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"`stager_name` must be a non-empty string, got {name!r}.")
    name = name.strip()
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown stager {name!r}; expected one of {sorted(registry)}."
        ) from exc


def create_stager(
    stager_name: str,
    stager_kwargs: dict[str, Any] | None = None,
    *,
    registry: dict[str, type[Stager]] | None = None,
) -> Stager:
    """Instantiate a :class:`Stager` from a registered class name and constructor
    kwargs.

    ``stager_name`` must match a concrete stager listed by
    :func:`discover_stager_classes`, for example ``"FileStager"``. ``stager_kwargs`` is
    forwarded to that class's constructor.
    """
    if stager_kwargs is not None and not isinstance(stager_kwargs, dict):
        raise TypeError(
            f"`stager_kwargs` must be a dictionary or None, got "
            f"{type(stager_kwargs).__name__}."
        )

    stager_cls = _resolve_stager_class(
        stager_name, registry or discover_stager_classes()
    )
    kwargs = {} if stager_kwargs is None else dict(stager_kwargs)

    try:
        return stager_cls(**kwargs)
    except TypeError as exc:
        raise TypeError(f"Failed to instantiate stager {stager_name!r}: {exc}") from exc

"""Public modifiers for dynamic reference values.

Modifiers run after a reference has been resolved to a Python value. Each
modifier validates that the value type is appropriate before transforming it
and raises :class:`ValueError` on misuse. The dynamic resolver wraps those
errors as :class:`~niiflow.preproc.staging.dynamic_referencing.DynamicReferenceError`.

Callables listed in ``_MODIFIER_NAMES`` are discovered by :func:`get_modifiers`.
The registered name is the modifier name used in ``|name`` /
``|name:arg[,arg...]`` syntax. Positional arguments after ``:`` are
comma-separated.
"""

from __future__ import annotations

__all__ = [
    "lstrip",
    "replace",
    "rstrip",
    "strip",
    "get_modifiers",
]

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from niiflow.preproc.utils.decorators import deprecate

ModifierFn = Callable[..., Any]

_MODIFIER_NAMES = frozenset({"lstrip", "replace", "rstrip", "strip"})


def _require_str_or_path(value: Any, *, modifier: str, ref: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError(
            f"modifier {modifier!r} requires str or Path, got {type(value).__name__} "
            f"in reference {{{ref}}}."
        )
    return str(value)


def rstrip(value: Any, suffix: str, *, ref: str) -> str:
    """Remove ``suffix`` from the end of a path string or :class:`~pathlib.Path`.

    Uses :meth:`str.removesuffix` semantics. For :class:`~pathlib.Path` values, the
    suffix is stripped from the full path string (``str(path)``).
    """
    if suffix == "":
        raise ValueError(f"Empty rstrip suffix in reference {{{ref}}}.")
    return _require_str_or_path(value, modifier="rstrip", ref=ref).removesuffix(suffix)


@deprecate(remove_in="0.5.0", alternative="rstrip")
def strip(value: Any, suffix: str, *, ref: str) -> str:
    """Deprecated alias of :func:`rstrip`.

    Will be removed in v0.5.0.
    """
    return rstrip(value, suffix, ref=ref)


def lstrip(value: Any, prefix: str, *, ref: str) -> str:
    """Remove ``prefix`` from the start of a path string or :class:`~pathlib.Path`.

    Uses :meth:`str.removeprefix` semantics. For :class:`~pathlib.Path` values, the
    prefix is stripped from the full path string (``str(path)``).
    """
    if prefix == "":
        raise ValueError(f"Empty lstrip prefix in reference {{{ref}}}.")
    return _require_str_or_path(value, modifier="lstrip", ref=ref).removeprefix(prefix)


def replace(value: Any, old: str, new: str, *, ref: str) -> str:
    """Replace ``old`` with ``new`` in a path string or :class:`~pathlib.Path`.

    Syntax is ``|replace:old,new``. All non-overlapping occurrences of ``old`` are
    replaced. ``new`` may be empty.
    """
    if old == "":
        raise ValueError(f"Empty replace pattern in reference {{{ref}}}.")
    return _require_str_or_path(value, modifier="replace", ref=ref).replace(old, new)


def get_modifiers() -> dict[str, ModifierFn]:
    """Return registered modifier callables keyed by modifier name."""
    module = sys.modules[__name__]
    modifiers: dict[str, ModifierFn] = {}
    for name in _MODIFIER_NAMES:
        obj = getattr(module, name)
        if callable(obj):
            modifiers[name] = obj
    return modifiers

"""Public modifiers for dynamic reference values.

Modifiers run after a reference has been resolved to a Python value. Each
modifier validates that the value type is appropriate before transforming it
and raises :class:`ValueError` on misuse. The dynamic resolver wraps those
errors as :class:`~niiflow.preproc.staging.dynamic_referencing.DynamicReferenceError`.

Callables listed in ``_MODIFIER_NAMES`` are discovered by :func:`get_modifiers`.
The registered name is the modifier name used in ``|name`` / ``|name:arg``
syntax.
"""

from __future__ import annotations

__all__ = [
    "strip",
    "get_modifiers",
]

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ModifierFn = Callable[..., Any]

_MODIFIER_NAMES = frozenset({"strip"})


def strip(value: Any, *, arg: str | None, ref: str) -> str:
    """Remove ``arg`` from the end of a path string or :class:`~pathlib.Path`.

    Uses :meth:`str.removesuffix` semantics. For :class:`~pathlib.Path` values, the
    suffix is stripped from the full path string (``str(path)``).
    """
    if arg is None or arg == "":
        raise ValueError(f"Empty strip suffix in reference {{{ref}}}.")

    if not isinstance(value, (str, Path)):
        raise ValueError(
            f"modifier 'strip' requires str or Path, got {type(value).__name__} "
            f"in reference {{{ref}}}."
        )

    return str(value).removesuffix(arg)


def get_modifiers() -> dict[str, ModifierFn]:
    """Return registered modifier callables keyed by modifier name."""
    module = sys.modules[__name__]
    modifiers: dict[str, ModifierFn] = {}
    for name in _MODIFIER_NAMES:
        obj = getattr(module, name)
        if callable(obj):
            modifiers[name] = obj
    return modifiers

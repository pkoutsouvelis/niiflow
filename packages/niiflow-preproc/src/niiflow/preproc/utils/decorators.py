"""Decorators for package-wide utilities."""

from __future__ import annotations

__all__ = ["deprecate"]

import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def deprecate(
    message: str | None = None,
    *,
    remove_in: str | None = None,
    alternative: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Emit :class:`DeprecationWarning` each time the wrapped callable is used.

    Args:
        message:
            Custom warning text. When omitted, a message is built from the
            callable name, ``remove_in``, and ``alternative``.
        remove_in:
            Version in which the callable will be removed (e.g. ``"1.0.0"``).
        alternative:
            Name of the replacement callable or API.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        warning = message
        if warning is None:
            warning = f"`{fn.__name__}` is deprecated"
            if remove_in is not None:
                warning += f" and will be removed in v{remove_in}"
            warning += "."
            if alternative is not None:
                warning += f" Use `{alternative}` instead."

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        return wrapper

    return decorator

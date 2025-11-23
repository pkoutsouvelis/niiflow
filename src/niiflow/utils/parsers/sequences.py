"""Helper functions to parse user-provided arguments into sequences."""

from __future__ import annotations

__all__ = [
    "ensure_sequence",
    "ensure_tuple",
]

from typing import Any
from collections.abc import Sequence

from niiflow.utils._validators import (
    validate_type,
    validate_seq_len,
    validate_seq_content_type,
)


def ensure_sequence(
    obj: Any,
    /,
    n: int | None = None,
    allowed_types: type | tuple[type, ...] | None = None,
    *,
    msg_id: str | None = None,
) -> Sequence[Any]:
    """Parse a user-provided object into a sequence."""
    if n is not None:
        validate_type(n, int, msg_id=msg_id)
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes)):
        obj = (obj,) * (n or 1)
    if n is not None:
        validate_seq_len(obj, n, msg_id=msg_id)
    if allowed_types is not None:
        validate_seq_content_type(obj, allowed_types, msg_id=msg_id)
    return obj


def ensure_tuple(
    obj: Any,
    /,
    n: int | None = None,
    allowed_types: type | tuple[type, ...] | None = None,
    *,
    msg_id: str | None = None,
) -> tuple[Any, ...]:
    """Parse a user-provided object into a tuple."""
    obj = ensure_sequence(obj, n, allowed_types, msg_id=msg_id)
    return tuple[Any, ...](obj)

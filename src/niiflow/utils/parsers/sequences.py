"""Helper functions to parse user-provided arguments into sequences."""

from __future__ import annotations

__all__ = [
    "ensure_sequence",
    "ensure_tuple",
]

from typing import Any, TypeVar
from collections.abc import Sequence

from niiflow.utils._validators import (
    validate_type,
    validate_seq_len,
    validate_seq_content_type,
)

T = TypeVar("T")

def ensure_sequence(
    obj: Any,
    /,
    n: int | None = None,
    allowed_types: type[T] | tuple[type[T], ...] | None = None,
    *,
    msg_id: str | None = None,
) -> Sequence[T]:
    """
    Parse a user-provided object into a sequence.

    Args:
        obj: Object to parse.
        n: Target length of the sequence.
        allowed_types: Expected types of the sequence contents.
        msg_id: Message identifier to use in the error message.

    Returns:
        Sequence of type T.

    Raises:
        TypeError: If the target length is not an integer.
        ValueError: If the sequence does not have the specified length.
        TypeError: If the sequence contents are not of the specified type.
    """
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
    allowed_types: type[T] | tuple[type[T], ...] | None = None,
    *,
    msg_id: str | None = None,
) -> tuple[T, ...]:
    """
    Parse a user-provided object into a tuple.

    Args:
        obj: Object to parse.
        n: Target length of the tuple.
        allowed_types: Expected types of the tuple contents.
        msg_id: Message identifier to use in the error message.

    Returns:
        Tuple of type T.

    Raises:
        TypeError: If the target length is not an integer.
        ValueError: If the tuple does not have the specified length.
        TypeError: If the tuple contents are not of the specified type.
    """
    obj = ensure_sequence(obj, n, allowed_types, msg_id=msg_id)
    return tuple(obj)

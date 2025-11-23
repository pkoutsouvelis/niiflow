"""Helpers to validate input sequences."""

from __future__ import annotations

__all__ = [
    "validate_non_str_sequence",
    "validate_seq_len",
    "validate_seq_content_type",
    "validate_non_str_sequence",
]

from typing import Any
from collections.abc import Sequence

from niiflow.utils._validators.basic import validate_type
from niiflow.utils._validators.utils import (
    parse_msg_id,
    parse_type,
)


def validate_non_str_sequence(obj: Any, /, *, msg_id: str | None = None) -> None:
    """
    Check if an object is a non-string sequence.

    Args:
        obj: Object to check.
        msg_id: Message identifier to use in the error message.

    Raises:
        TypeError: If the object is a string or not a sequence.
    """
    msg_id = parse_msg_id(msg_id)
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes)):
        msg = f"{msg_id}Expected a non-string sequence, " f"got {type(obj).__name__!r}."
        raise TypeError(msg)


def validate_seq_len(
    obj: Sequence,
    n: int,
    /,
    *,
    msg_id: str | None = None,
) -> None:
    """
    Check if a sequence has the specified length.

    Args:
        obj: Sequence to check.
        length: Expected length of the sequence.
        msg_id: Message identifier to use in the error message.

    Raises:
        ValueError: If the sequence does not have the specified length.
    """
    validate_type(n, int, msg_id=msg_id)
    msg_id = parse_msg_id(msg_id)
    if len(obj) != n:
        msg = f"{msg_id}Expected a sequence of length {n}, " f"got {len(obj)}."
        raise ValueError(msg)


def validate_seq_content_type(
    obj: Sequence,
    type_: type | tuple[type, ...],
    /,
    *,
    msg_id: str | None = None,
) -> None:
    """
    Check if the contents of a sequence are of the specified type.

    Args:
        obj: Sequence to check.
        type_: Expected type(s) of the sequence contents.
        msg_id: Message identifier to use in the error message.

    Raises:
        TypeError: If the sequence contents are not of the specified type.
    """
    msg_id = parse_msg_id(msg_id)
    type_ = parse_type(type_)
    for i, item in enumerate(obj):
        try:
            validate_type(item, type_, msg_id=msg_id)
        except TypeError as e:
            msg = (
                f"{msg_id}Expected an object of type "
                f"{', '.join([t.__name__ for t in type_])!r}, "
                f"got {type(item).__name__!r} in sequence item {i}."
            )
            raise TypeError(msg) from e

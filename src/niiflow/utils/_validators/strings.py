"""Helpers to validate input strings."""

from __future__ import annotations

__all__ = [
    "validate_literal_str",
]

from typing import Any
from collections.abc import Sequence

from niiflow.utils._validators.utils import parse_msg_id
from niiflow.utils._validators.basic import validate_type


def validate_literal_str(
    obj: Any,
    literals: tuple[str, ...],
    /,
    *,
    msg_id: str | None = None,
) -> None:
    """
    Check if an object is a string that is one of the specified literals.

    Args:
        obj: Object to check.
        literals: Tuple of literals to check against.
        msg_id: Message identifier to use in the error message.

    Raises:
        TypeError: If the object is not a string.
        ValueError: If the object is not one of the specified literals.
    """
    msg_id = parse_msg_id(msg_id)
    validate_type(literals, tuple, msg_id=msg_id)
    for literal in literals:
        validate_type(literal, str, msg_id=msg_id)

    validate_type(obj, str, msg_id=msg_id)
    if obj not in literals:
        msg = f"{msg_id}Expected a string that is one of {literals!r}, " f"got {obj!r}."
        raise ValueError(msg)

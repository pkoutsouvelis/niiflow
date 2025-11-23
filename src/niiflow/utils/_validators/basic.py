"""Fundamental input validators."""

from __future__ import annotations

__all__ = [
    "validate_type",
]

from typing import Any

from niiflow.utils._validators.utils import parse_msg_id, parse_type


def validate_type(
    obj: Any,
    type_: type | tuple[type, ...],
    /,
    *,
    msg_id: str | None = None,
) -> None:
    """
    Check if an object is of the specified type.

    Args:
        obj: Object to check.
        type_: Expected type of the object.
        msg_id: Message identifier to use in the error message.

    Raises:
        TypeError: If the object is not of the specified type.
    """
    msg_id = parse_msg_id(msg_id)
    type_ = parse_type(type_)
    if not isinstance(obj, type_):
        msg = (
            f"{msg_id}Expected an object of type "
            f"{', '.join([t.__name__ for t in type_])!r}, "
            f"got {type(obj).__name__!r}."
        )
        raise TypeError(msg)

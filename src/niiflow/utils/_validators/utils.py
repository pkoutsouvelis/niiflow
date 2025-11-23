"""Helpers to validate input objects."""

from __future__ import annotations

from typing import Any


def parse_msg_id(msg_id: Any) -> str:
    """Format the message identifier."""
    if msg_id is not None and not isinstance(msg_id, str):
        raise TypeError("msg_id must be a string or None.")
    return f"[{msg_id}] " if msg_id is not None else ""


def parse_type(type_: Any) -> tuple[Any, ...]:
    """Parse the input type into a tuple."""
    if not isinstance(type_, (type, tuple)):
        raise TypeError("type must be a type or a tuple of types.")

    if isinstance(type_, tuple):
        for t in type_:
            if not isinstance(t, type):
                raise TypeError("type must be a type or a tuple of types.")
        return type_

    return (type_,)

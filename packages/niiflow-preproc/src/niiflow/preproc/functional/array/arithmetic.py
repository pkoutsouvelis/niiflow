"""Pointwise arithmetic helpers for numpy arrays."""

from __future__ import annotations

__all__ = [
    "pointwise_arithmetic",
]

from collections.abc import Mapping
from typing import Any

import numpy as np

from niiflow.preproc.functional.array.utils import validate_numeric_array

_OP_CALLABLES = {
    "mul": np.multiply,
    "add": np.add,
    "div": np.divide,
    "sub": np.subtract,
}


def pointwise_arithmetic(
    input: np.ndarray,
    *operations: Mapping[str, Any],
) -> np.ndarray:
    """Apply an ordered sequence of pointwise arithmetic operations.

    Operations are applied strictly left-to-right with no combining, reordering,
    or algebraic simplification. Each operation is a single-key mapping:

    * ``{"mul": value}`` -> ``out = out * value``
    * ``{"add": value}`` -> ``out = out + value``
    * ``{"div": value}`` -> ``out = out / value``
    * ``{"sub": value}`` -> ``out = out - value``

    ``value`` may be a scalar or a :class:`numpy.ndarray` compatible with the
    running result under ordinary NumPy broadcasting. Dtype and broadcasting
    follow native NumPy arithmetic semantics.

    Args:
        input:
            Numeric input array.
        *operations:
            One or more single-key operation mappings. At least one is required.

    Returns:
        The transformed array after applying every operation in order.

    Raises:
        ValueError: If `input` is invalid, no operations are provided, an
            operation is not a single-key mapping, an operation key is not one
            of ``"mul"``, ``"add"``, ``"div"``, or ``"sub"``, or an operand
            cannot be broadcast with the running array.

    Examples:
        >>> field = np.array([0.0, 1.0, 2.0])
        >>> mask = np.array([0.0, 1.0, 0.5])
        >>> pointwise_arithmetic(
        ...     field,
        ...     {"sub": 1},
        ...     {"mul": mask},
        ...     {"add": 1},
        ... )
        array([1. , 1. , 1.5])
    """
    validate_numeric_array(input, name="input")
    if not operations:
        raise ValueError("`pointwise_arithmetic` requires at least one operation")

    out: np.ndarray = input
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise ValueError(
                f"operation at index {index} must be a mapping, got "
                f"{type(operation).__name__}"
            )
        if len(operation) != 1:
            raise ValueError(
                f"operation at index {index} must contain exactly one key, "
                f"got {len(operation)}"
            )
        ((key, value),) = operation.items()
        op = _OP_CALLABLES.get(key)
        if op is None:
            supported = ", ".join(sorted(repr(name) for name in _OP_CALLABLES))
            raise ValueError(
                f"unsupported operation {key!r} at index {index}; "
                f"expected one of {{{supported}}}"
            )
        try:
            out = op(out, value)
        except ValueError as e:
            value_shape = getattr(value, "shape", ())
            raise ValueError(
                f"{key!r} operation at index {index} failed: could not "
                f"broadcast running array of shape {out.shape} with operand "
                f"of shape {value_shape}"
            ) from e
    return out

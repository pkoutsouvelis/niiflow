"""Pointwise arithmetic helpers for ANTsImage objects.

Mirrors :mod:`niiflow.preproc.functional.array.arithmetic`, preserving image
metadata while operating on voxel data in numpy.
"""

from __future__ import annotations

__all__ = [
    "pointwise_arithmetic",
]

from collections.abc import Mapping
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.array import arithmetic as _array_arithmetic

from .utils import ants_to_numpy_with_metadata, numpy_to_ants_with_metadata


def _resolve_operation(
    operation: Mapping[str, Any] | Any,
) -> Mapping[str, Any] | Any:
    """Convert ANTsImage operands to numpy; leave other operations unchanged."""
    if not isinstance(operation, Mapping) or len(operation) != 1:
        return operation
    ((key, value),) = operation.items()
    if isinstance(value, ANTsImage):
        return {key: value.numpy()}
    return operation


def pointwise_arithmetic(
    input: ANTsImage,
    *operations: Mapping[str, Any],
) -> ANTsImage:
    """Apply an ordered sequence of pointwise arithmetic operations.

    Wraps :func:`~niiflow.preproc.functional.array.arithmetic.pointwise_arithmetic`,
    preserving origin, spacing, and direction of `input`.

    Operations are applied strictly left-to-right with no combining, reordering,
    or algebraic simplification. Each operation is a single-key mapping:

    * ``{"mul": value}`` → ``out = out * value``
    * ``{"add": value}`` → ``out = out + value``
    * ``{"div": value}`` → ``out = out / value``
    * ``{"sub": value}`` → ``out = out - value``

    ``value`` may be a scalar, a :class:`numpy.ndarray`, or an
    :class:`~ants.core.ANTsImage` compatible with `input` under ordinary NumPy
    broadcasting. Image operands are converted to arrays before the operation;
    scalars and arrays are forwarded as-is.

    Args:
        input:
            Input :class:`~ants.core.ANTsImage`.
        *operations:
            One or more single-key operation mappings. At least one is required.

    Returns:
        A new :class:`~ants.core.ANTsImage` with the same spatial metadata as
        `input`.

    Raises:
        ValueError: If `input` is invalid, no operations are provided, an
            operation is not a single-key mapping, an operation key is not one
            of ``"mul"``, ``"add"``, ``"div"``, or ``"sub"``, an operand cannot
            be broadcast with the running array, or the result shape differs
            from `input`.
    """
    image_array, metadata = ants_to_numpy_with_metadata(input)
    resolved = tuple(_resolve_operation(operation) for operation in operations)
    out_array = _array_arithmetic.pointwise_arithmetic(image_array, *resolved)
    return numpy_to_ants_with_metadata(out_array, metadata)

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

from .metadata import sync_ants_metadata
from .utils import ants_to_numpy_with_metadata, numpy_to_ants_with_metadata


def _resolve_operation(
    operation: Mapping[str, Any] | Any,
    *,
    reference: ANTsImage,
    index: int,
) -> Mapping[str, Any] | Any:
    """Convert ANTsImage operands to numpy; leave other operations unchanged."""
    if not isinstance(operation, Mapping) or len(operation) != 1:
        return operation
    ((key, value),) = operation.items()
    if isinstance(value, ANTsImage):
        try:
            sync_ants_metadata(value, reference)
        except ValueError as e:
            raise ValueError(
                f"ANTsImage operand for {key!r} at operation index {index} "
                f"must share a voxel grid with `input`: {e}"
            ) from e
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
    :class:`~ants.core.ANTsImage`. Image operands must share a voxel grid with
    `input` (shape, origin, spacing, and direction within the tolerances of
    :func:`~niiflow.preproc.functional.image.metadata.sync_ants_metadata`);
    they are converted to arrays before the operation. Scalars and arrays are
    forwarded as-is and must be broadcast-compatible with `input`.

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
            of ``"mul"``, ``"add"``, ``"div"``, or ``"sub"``, an ANTsImage
            operand is not on the same voxel grid as `input`, an operand cannot
            be broadcast with the running array, or the result shape differs
            from `input`.
    """
    image_array, metadata = ants_to_numpy_with_metadata(input)
    resolved = tuple(
        _resolve_operation(operation, reference=input, index=index)
        for index, operation in enumerate(operations)
    )
    out_array = _array_arithmetic.pointwise_arithmetic(image_array, *resolved)
    return numpy_to_ants_with_metadata(out_array, metadata)

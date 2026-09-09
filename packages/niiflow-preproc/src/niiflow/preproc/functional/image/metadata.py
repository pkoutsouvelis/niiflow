"""Spatial metadata helpers for ANTsImage objects."""

from __future__ import annotations

__all__ = [
    "sync_ants_metadata",
]

import ants
import numpy as np
from ants.core import ANTsImage

from niiflow.preproc.utils.misc import numeric_mismatch

from .utils import ensure_ants_image


def sync_ants_metadata(
    image: ANTsImage,
    reference: ANTsImage,
    *,
    origin_atol: float = 1e-5,
    spacing_atol: float = 1e-5,
    direction_atol: float = 1e-5,
    rtol: float = 0.0,
) -> ANTsImage:
    """Copy `reference` spatial metadata onto `image` when they share a voxel grid.

    A tolerance-gated helper that validates two images are effectively on the
    same voxel grid before synchronizing metadata. Shapes must match. Origin,
    spacing, and direction are compared against configurable absolute
    tolerances and a shared relative tolerance. When every difference is
    within tolerance, `reference`'s spatial metadata is copied onto
    `image`'s voxel data with no resampling and no change to voxel values.
    When any difference exceeds tolerance, an error is raised rather than
    masking a genuine spatial mismatch.

    Args:
        image:
            The :class:`~ants.core.ANTsImage` whose voxel values are kept.
        reference:
            The :class:`~ants.core.ANTsImage` whose origin, spacing, and
            direction are copied onto `image` when the grids match.
        origin_atol:
            Absolute tolerance for origin comparison (physical units).
        spacing_atol:
            Absolute tolerance for spacing comparison (physical units).
        direction_atol:
            Absolute tolerance for the direction-cosine matrix.
        rtol:
            Relative tolerance applied to origin, spacing, and direction.
            Comparison uses :func:`numpy.allclose`:
            ``abs(a - b) <= atol + rtol * abs(b)``.

    Returns:
        A new :class:`~ants.core.ANTsImage` with `image`'s voxel data and
        `reference`'s spatial metadata.

    Raises:
        TypeError: If a tolerance is not a number.
        ValueError: If either input is not an ANTsImage, shapes differ, a
            tolerance is invalid, or origin, spacing, or direction differ
            beyond the configured tolerance.
    """
    ensure_ants_image(image)
    ensure_ants_image(reference, name="reference")

    if image.shape != reference.shape:
        raise ValueError(
            "`image` and `reference` must have the same shape, got "
            f"{image.shape} and {reference.shape}"
        )

    mismatches = [
        mismatch
        for mismatch in (
            numeric_mismatch(
                image.origin,
                reference.origin,
                atol=origin_atol,
                rtol=rtol,
                name="origin",
                actual_label="image",
                expected_label="reference",
            ),
            numeric_mismatch(
                image.spacing,
                reference.spacing,
                atol=spacing_atol,
                rtol=rtol,
                name="spacing",
                actual_label="image",
                expected_label="reference",
            ),
            numeric_mismatch(
                image.direction,
                reference.direction,
                atol=direction_atol,
                rtol=rtol,
                name="direction",
                actual_label="image",
                expected_label="reference",
            ),
        )
        if mismatch is not None
    ]
    if mismatches:
        raise ValueError(
            "`image` and `reference` are not on the same voxel grid: "
            + "; ".join(mismatches)
        )

    return ants.from_numpy(
        image.numpy(),
        origin=tuple(float(v) for v in reference.origin),
        spacing=tuple(float(v) for v in reference.spacing),
        direction=np.asarray(reference.direction, dtype=np.float64),
    )

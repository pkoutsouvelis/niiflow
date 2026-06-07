"""Reorientation functions for ANTsImage objects."""

from __future__ import annotations

__all__ = ["ants_reorient"]

from typing import Literal, get_args

import ants
from ants.core import ANTsImage

from niiflow.preproc.functional.image.utils import ensure_ants_image

OrientationConvention = Literal["itk", "nibabel"]

_VALID_AXIS_LABELS = frozenset("RLAPSI")
_NIBABEL_TO_ITK = str.maketrans("SI", "IS")


def _to_itk_orientation(
    orientation: str,
    *,
    convention: OrientationConvention,
) -> str:
    if convention not in get_args(OrientationConvention):
        raise TypeError(
            f"`convention` must be one of {get_args(OrientationConvention)}, "
            f"got {convention!r}"
        )
    if not isinstance(orientation, str):
        raise TypeError(
            f"`orientation` must be a string, got {type(orientation).__name__}"
        )
    code = orientation.strip().upper()
    if len(code) != 3 or any(label not in _VALID_AXIS_LABELS for label in code):
        raise ValueError(
            f"`orientation` must be a three-letter code from R/L/A/P/S/I, "
            f"got {orientation!r}"
        )

    itk_orientation = (
        code.translate(_NIBABEL_TO_ITK) if convention == "nibabel" else code
    )
    possible = ants.get_possible_orientations()
    if itk_orientation not in possible:
        raise ValueError(
            f"Unknown orientation {orientation!r} for convention {convention!r}; "
            f"expected one of {possible}"
        )
    return itk_orientation


def ants_reorient(
    image: ANTsImage,
    orientation: str,
    *,
    convention: OrientationConvention = "itk",
) -> ANTsImage:
    """Reorient an ANTsImage to a target orientation code.

    Wraps :func:`ants.reorient_image2`.

    Args:
        image:
            The :class:`ants.core.ANTsImage` to reorient.
        orientation:
            Three-letter target orientation code.
        convention:
            ``"itk"`` (default) — pass `orientation` directly to ANTs.
            ``"nibabel"`` — convert from a nibabel axcode first (S↔I per axis).

    Returns:
        The reoriented :class:`ants.core.ANTsImage`.
    """
    ensure_ants_image(image)
    return ants.reorient_image2(
        image, _to_itk_orientation(orientation, convention=convention)
    )

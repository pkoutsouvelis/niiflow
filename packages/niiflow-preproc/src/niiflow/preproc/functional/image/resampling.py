"""Resampling functions for ANTsImage objects.

Thin wrappers around the ANTs resampling routines (:func:`ants.resample_image` and
:func:`ants.resample_image_to_target`) that expose a small, validated interface while
leaving the heavy lifting to ANTs. See the ANTsPy documentation for the full behaviour
of the underlying calls.
"""

from __future__ import annotations

__all__ = [
    "ants_resample",
    "ants_resample_to_target",
]

from typing import Any, Literal, Sequence, cast

from ants.core import ANTsImage
from ants.ops import resample_image, resample_image_to_target

from .utils import ensure_ants_image, reject_reserved_kwargs

# Public interpolation name -> ANTs ``interp_type`` integer code, as accepted by
# :func:`ants.resample_image`. Keep in sync with the `ResampleInterpolation` literal.
_RESAMPLE_INTERP_CODES: dict[str, int] = {
    "linear": 0,
    "nearest": 1,
    "gaussian": 2,
    "sinc": 3,
    "bspline": 4,
}

ResampleInterpolation = Literal["linear", "nearest", "gaussian", "sinc", "bspline"]

# Interpolation modes accepted by :func:`ants.resample_image_to_target`. These are
# the ANTs names passed straight through as ``interp_type``.
TargetInterpolation = Literal[
    "linear",
    "nearestNeighbor",
    "multiLabel",
    "gaussian",
    "bSpline",
    "cosineWindowedSinc",
    "welchWindowedSinc",
    "hammingWindowedSinc",
    "lanczosWindowedSinc",
    "genericLabel",
]


def ants_resample(
    image: ANTsImage,
    resample_params: Sequence[int | float],
    use_voxels: bool = False,
    interpolation: ResampleInterpolation = "linear",
) -> ANTsImage:
    """Resample an ANTsImage by spacing or voxel count.

    Wraps :func:`ants.resample_image`. The image is resampled either to a
    target voxel spacing (the default) or to an explicit voxel count when
    `use_voxels` is ``True``.

    Args:
        image:
            The :class:`ants.core.ANTsImage` to resample.
        resample_params:
            One value per spatial dimension. Interpreted as the target
            spacing (in physical units) when `use_voxels` is ``False``, or as
            the target number of voxels when `use_voxels` is ``True``.
        use_voxels:
            If ``True``, treat `resample_params` as voxel counts rather than
            spacings.
        interpolation:
            Interpolation mode. Corresponds to the ``interp_type`` argument of
            :func:`ants.resample_image`: one of ``"linear"``, ``"nearest"``,
            ``"gaussian"``, ``"sinc"`` (windowed sinc), or ``"bspline"``.

    Returns:
        The resampled :class:`ants.core.ANTsImage`.

    Raises:
        ValueError: If `image` is not an ANTsImage or `interpolation` is not a
            supported value.
    """
    ensure_ants_image(image)
    try:
        interp_type = _RESAMPLE_INTERP_CODES[interpolation]
    except KeyError:
        raise ValueError(
            f"Invalid interpolation {interpolation!r}; expected one of "
            f"{sorted(_RESAMPLE_INTERP_CODES)}."
        ) from None
    return resample_image(
        image=image,
        resample_params=list(resample_params),
        use_voxels=use_voxels,
        interp_type=interp_type,
    )


def ants_resample_to_target(
    image: ANTsImage,
    target: ANTsImage,
    interpolation: TargetInterpolation = "linear",
    **kwargs: Any,
) -> ANTsImage:
    """Resample an ANTsImage into the space of a target ANTsImage.

    Wraps :func:`ants.resample_image_to_target`. The output is defined on
    `target`'s grid (origin, spacing, direction and shape) and inherits its
    pixel type.

    Args:
        image:
            The :class:`ants.core.ANTsImage` to resample.
        target:
            The :class:`ants.core.ANTsImage` whose grid defines the output
            space.
        interpolation:
            Interpolation mode. Corresponds to the ``interp_type`` argument of
            :func:`ants.resample_image_to_target`, e.g. ``"linear"``,
            ``"nearestNeighbor"``, ``"gaussian"``, ``"bSpline"``,
            ``"genericLabel"`` (recommended for label images), or one of the
            windowed-sinc variants.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`ants.resample_image_to_target` (e.g. ``imagetype`` or
            ``verbose``). The `image`, `target`, and `interp_type` arguments
            are reserved and set from `image`, `target`, and `interpolation`
            respectively.

    Returns:
        The resampled :class:`ants.core.ANTsImage` on `target`'s grid.

    Raises:
        ValueError: If `image` or `target` is not an ANTsImage.
        TypeError: If a reserved argument is passed via `kwargs`.
    """
    ensure_ants_image(image)
    ensure_ants_image(target, name="target")
    reject_reserved_kwargs(
        kwargs, ("image", "target", "interp_type"), func_name="ants_resample_to_target"
    )
    return cast(
        ANTsImage,
        resample_image_to_target(
            image=image, target=target, interp_type=interpolation, **kwargs
        ),
    )

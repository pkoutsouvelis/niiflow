"""Similarity metrics for ANTsImage objects.

``ants_similarity_metrics`` wraps :func:`ants.image_similarity`.
``dice_similarity_coefficient`` wraps the array primitive in
:mod:`niiflow.preproc.functional.array.metrics`. Throughout this module the
ANTs "fixed"/"moving" terminology is exposed as ``target``/``image``.
"""

from __future__ import annotations

__all__ = [
    "ants_similarity_metrics",
    "dice_similarity_coefficient",
]

from typing import Any

from ants import image_physical_space_consistency, image_similarity
from ants.core import ANTsImage

from niiflow.preproc.functional.array import metrics as _array_metrics

from .utils import ensure_ants_image, reject_reserved_kwargs


def _ensure_compatible_space(
    image: ANTsImage,
    target: ANTsImage,
    mask: ANTsImage | None = None,
) -> None:
    """Require ``image``, ``target``, and optional ``mask`` to share physical space."""
    ensure_ants_image(image)
    ensure_ants_image(target, name="target")
    others: list[tuple[str, ANTsImage]] = [("target", target)]
    if mask is not None:
        ensure_ants_image(mask, name="mask")
        others.append(("mask", mask))
    for name, other in others:
        if other.shape != image.shape:
            raise ValueError(
                f"`{name}` shape {tuple(other.shape)} is incompatible with "
                f"`image` shape {tuple(image.shape)}; resample explicitly "
                "before computing metrics"
            )
        if not image_physical_space_consistency(image, other):
            raise ValueError(
                f"`{name}` is not in the same physical space as `image`; "
                "resample explicitly before computing metrics"
            )


def ants_similarity_metrics(
    image: ANTsImage,
    target: ANTsImage,
    metric: str,
    mask: ANTsImage | None = None,
    **kwargs: Any,
) -> float:
    """Compute one ANTs image similarity score (higher is better).

    Wraps :func:`ants.image_similarity` for a single ``metric``. The name is
    forwarded unchanged as ANTs ``metric_type``, so any metric that function
    accepts can be used. Check the ANTsPy documentation for
    ``ants.image_similarity`` for the full list and for sampling options.
    Common choices are ``"MattesMutualInformation"``, ``"Correlation"``, and
    ``"MeanSquares"``.

    ANTs returns these metrics as distances: lower, or more negative, is a
    better match (Correlation of an image with itself is ``-1``). This
    wrapper negates that distance so the returned scalar is higher-is-better.

    Args:
        image:
            Moving :class:`ants.core.ANTsImage`, already in ``target`` space.
        target:
            Fixed :class:`ants.core.ANTsImage`.
        metric:
            One ANTs ``metric_type`` string per call, for example
            ``"MattesMutualInformation"``, ``"Correlation"``, or
            ``"MeanSquares"``.
        mask:
            Optional mask in the same physical space as ``image`` and
            ``target``. When set, evaluation is restricted to this mask on
            both images. When ``None``, the full image is used.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`ants.image_similarity`, such as ``sampling_strategy`` and
            ``sampling_percentage``. ``fixed``, ``moving``, ``metric_type``,
            ``fixed_mask``, and ``moving_mask`` are reserved.

    Returns:
        A float score for ``metric``. Higher is better.

    Raises:
        TypeError: If ``metric`` is not a string, or a reserved argument is
            passed via ``kwargs``.
        ValueError: If ``metric`` is empty, an input is not an ANTs image,
            or the images are not in a compatible shape and physical space.
    """
    if not isinstance(metric, str):
        raise TypeError(
            "`metric` must be a single ANTs metric name, "
            f"got {type(metric).__name__}"
        )
    if not metric:
        raise ValueError("`metric` must be a non-empty ANTs metric name")

    _ensure_compatible_space(image, target, mask)
    reject_reserved_kwargs(
        kwargs,
        ("fixed", "moving", "metric_type", "fixed_mask", "moving_mask"),
        func_name="ants_similarity_metrics",
    )
    # ANTs convention: dissimilarity / cost (Correlation of an image with
    # itself is -1). Negate so higher scores mean better agreement.
    cost = float(
        image_similarity(
            target,
            image,
            metric_type=metric,
            fixed_mask=mask,
            moving_mask=mask,
            **kwargs,
        )
    )
    return -cost


def dice_similarity_coefficient(image: ANTsImage, target: ANTsImage) -> float:
    """Dice similarity coefficient between two binary ANTs masks.

    Wraps
    :func:`~niiflow.preproc.functional.array.metrics.dice_similarity_coefficient`.
    ``image`` and ``target`` must share a voxel grid. Voxel values must be
    exactly ``0`` and ``1``. Boolean, integer, and floating-point images are
    accepted on that condition, including masks that ANTs reads back as float.

    Args:
        image:
            Binary mask.
        target:
            Binary mask on the same grid as ``image``.

    Returns:
        Dice coefficient in ``[0, 1]``. Empty masks that agree score ``1.0``.

    Raises:
        ValueError: If an input is not an ANTs image, the grids differ, or
            the voxel values are not binary.
    """
    _ensure_compatible_space(image, target)
    return _array_metrics.dice_similarity_coefficient(image.numpy(), target.numpy())

"""Registration functions for ANTsImage objects.

Thin wrappers around the ANTs registration routines (:func:`ants.registration` and
:func:`ants.apply_transforms`). Throughout this module the ANTs "fixed"/"moving"
terminology is exposed as `target`/`image`: `image` is the moving image that is warped
into the space of the fixed `target`. See the ANTsPy documentation for the full
behaviour of the underlying calls.
"""

from __future__ import annotations

__all__ = [
    "ants_apply_transforms",
    "ants_registration",
    "ants_similarity_metrics",
]

from collections.abc import Sequence
from typing import Any

from ants import image_physical_space_consistency, image_similarity
from ants.core import ANTsImage
from ants.registration import registration
from ants.registration import apply_transforms

from .utils import ensure_ants_image, reject_reserved_kwargs

# User-facing metric names -> ANTs ``image_similarity`` metric_type.
# ANTs returns dissimilarities (lower/more-negative is better); we negate so
# that every exposed score uses higher-is-better semantics.
_SIMILARITY_METRICS: dict[str, str] = {
    "correlation": "Correlation",
    "mattes_mutual_information": "MattesMutualInformation",
    "neighborhood_correlation": "ANTSNeighborhoodCorrelation",
}
_DEFAULT_SIMILARITY_METRICS: tuple[str, ...] = tuple(_SIMILARITY_METRICS)


def ants_apply_transforms(
    image: ANTsImage,
    target: ANTsImage,
    interpolation: str = "linear",
    **kwargs: Any,
) -> ANTsImage:
    """Apply a list of transforms to map `image` into `target` space.

    Wraps :func:`ants.apply_transforms`, mapping the moving `image` onto the
    fixed `target` grid. The list of transforms is supplied through
    `kwargs` as ``transformlist`` (a list of transform filenames, as produced
    by :func:`ants_registration`).

    Args:
        image:
            The moving :class:`ants.core.ANTsImage` to transform.
        target:
            The fixed :class:`ants.core.ANTsImage` defining the output space
            and pixel type.
        interpolation:
            Interpolation mode used when warping `image`. Corresponds to the
            `interpolator` argument of :func:`ants.apply_transforms`.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`ants.apply_transforms` (e.g. ``transformlist``,
            ``whichtoinvert``, ``defaultvalue``). The `fixed`, `moving`, and
            `interpolator` arguments are reserved and set from `target`, `image`,
            and `interpolation` respectively.

    Returns:
        The transformed :class:`ants.core.ANTsImage` on `target`'s grid.

    Raises:
        ValueError: If `image` or `target` is not an ANTsImage.
        TypeError: If a reserved argument is passed via `kwargs`.
    """
    ensure_ants_image(image)
    ensure_ants_image(target, name="target")
    reject_reserved_kwargs(
        kwargs, ("fixed", "moving", "interpolator"), func_name="ants_apply_transforms"
    )
    return apply_transforms(
        fixed=target,
        moving=image,
        interpolator=interpolation,
        **kwargs,
    )


def ants_registration(
    image: ANTsImage,
    target: ANTsImage,
    mode: str = "Affine",
    apply_forward: bool = True,
    interpolation: str = "linear",
    **kwargs: Any,
) -> dict[str, Any] | tuple[ANTsImage, dict[str, Any]]:
    """Register `image` to `target` and optionally apply the transform.

    Wraps :func:`ants.registration` to estimate the transform mapping the
    moving `image` onto the fixed `target`, then (unless `apply_forward`
    is ``False``) warps `image` with the resulting forward transform via
    :func:`ants.apply_transforms`.

    Args:
        image:
            The moving :class:`ants.core.ANTsImage` to register.
        target:
            The fixed :class:`ants.core.ANTsImage` to register against.
        mode:
            ANTs registration mode (``type_of_transform``), e.g. ``"Rigid"``,
            ``"Affine"``, or ``"SyN"``. Matching is case-sensitive; see
            :func:`ants.registration` for the full list.
        apply_forward:
            If ``False``, only estimate and return the transforms without
            warping `image`.
        interpolation:
            Interpolation mode used when warping `image` (only relevant when
            `apply_forward` is ``True``).
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`ants.registration` (e.g. ``mask``, ``random_seed``,
            ``verbose``). The `fixed`, `moving`, and `type_of_transform`
            arguments are reserved.

    Returns:
        If `apply_forward` is ``False``, a dict with keys ``"fwdtransforms"``
        and ``"invtransforms"`` holding the forward and inverse transform
        filename lists. Otherwise, a ``(warped, transforms)`` tuple where
        ``warped`` is the registered :class:`ants.core.ANTsImage` and
        ``transforms`` is that same dict.

    Raises:
        ValueError: If `image` or `target` is not an ANTsImage.
        TypeError: If a reserved argument is passed via `kwargs`.
    """
    ensure_ants_image(image)
    ensure_ants_image(target, name="target")
    reject_reserved_kwargs(
        kwargs,
        ("fixed", "moving", "type_of_transform"),
        func_name="ants_registration",
    )
    reg = registration(fixed=target, moving=image, type_of_transform=mode, **kwargs)
    result = {
        "fwdtransforms": reg["fwdtransforms"],
        "invtransforms": reg["invtransforms"],
    }
    if not apply_forward:
        return result
    warped = apply_transforms(
        fixed=target,
        moving=image,
        transformlist=result["fwdtransforms"],
        interpolator=interpolation,
    )
    return warped, result


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
                "before computing similarity metrics"
            )
        if not image_physical_space_consistency(image, other):
            raise ValueError(
                f"`{name}` is not in the same physical space as `image`; "
                "resample explicitly before computing similarity metrics"
            )


def ants_similarity_metrics(
    image: ANTsImage,
    target: ANTsImage,
    mask: ANTsImage | None = None,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute image similarity scores (higher is better), typically useful for
    registration quality control.

    Wraps :func:`ants.image_similarity` for the requested metrics. When ``mask``
    is provided, sampling is restricted to that mask on both images. ANTs returns
    dissimilarities for these metrics; each score is negated so callers can apply
    ``score >= cutoff`` thresholds uniformly.

    Args:
        image:
            Moving :class:`ants.core.ANTsImage` (already in ``target`` space).
        target:
            Fixed/target :class:`ants.core.ANTsImage`.
        mask:
            Optional binary (or weighted) mask in the same physical space as
            ``image`` and ``target``. When set, metric evaluation is restricted
            to this mask; when ``None``, the full image is used.
        metrics:
            Subset of ``correlation``, ``mattes_mutual_information``, and
            ``neighborhood_correlation``. Defaults to all three. Order is
            preserved in the returned mapping.

    Returns:
        Mapping from metric name to a float score (higher is better).

    Raises:
        ValueError: If inputs are not ANTs images, are not in compatible
            physical space / shape, ``metrics`` is empty, or a metric name is
            unknown.
    """
    _ensure_compatible_space(image, target, mask)
    if metrics is None:
        selected = _DEFAULT_SIMILARITY_METRICS
    else:
        if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
            raise TypeError(
                "`metrics` must be a sequence of metric names, got "
                f"{type(metrics).__name__}"
            )
        if len(metrics) == 0:
            raise ValueError("`metrics` must contain at least one metric name")
        unknown = [name for name in metrics if name not in _SIMILARITY_METRICS]
        if unknown:
            supported = ", ".join(sorted(_SIMILARITY_METRICS))
            raise ValueError(
                f"Unknown similarity metric(s) {unknown!r}; expected names "
                f"from {{{supported}}}"
            )
        selected = tuple(metrics)

    scores: dict[str, float] = {}
    for name in selected:
        # ANTs convention: dissimilarity / cost (e.g. Correlation of an image
        # with itself is -1). Negate so higher scores mean better agreement.
        cost = float(
            image_similarity(
                target,
                image,
                metric_type=_SIMILARITY_METRICS[name],
                fixed_mask=mask,
                moving_mask=mask,
            )
        )
        scores[name] = -cost
    return scores

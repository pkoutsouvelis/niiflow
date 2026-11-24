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
]

from typing import Any

from ants.core import ANTsImage
from ants.registration import registration
from ants.registration import apply_transforms

from .utils import ensure_ants_image, reject_reserved_kwargs


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

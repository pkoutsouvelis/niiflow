"""Skull-stripping functions for ANTsImage objects.

Wrappers around the ANTsPyNet deep-learning brain extraction
(:func:`antspynet.utilities.brain_extraction`) and the ANTs masking routine
(:func:`ants.mask_image`). See the ANTsPy / ANTsPyNet documentation for the full
behaviour of the underlying calls.
"""

from __future__ import annotations

__all__ = [
    "ants_apply_mask",
    "ants_brain_extraction",
]

from typing import Any

from ants.core import ANTsImage
from ants.ops import mask_image, threshold_image, morphology
from antspynet.utilities import brain_extraction

from .utils import ensure_ants_image


def ants_apply_mask(**kwargs: Any) -> ANTsImage:
    """Apply a (possibly multi-label) mask to an ANTsImage.

    Wraps :func:`ants.mask_image`. All arguments are forwarded as keyword
    arguments.

    Args:
        **kwargs:
            Keyword arguments forwarded to :func:`ants.mask_image`. Common
            ones include ``image`` (the :class:`ants.core.ANTsImage` to mask),
            ``mask`` (the mask or label image), ``level`` (label value(s) to
            keep), and ``binarize``.

    Returns:
        The masked :class:`ants.core.ANTsImage`.
    """
    return mask_image(**kwargs)


def ants_brain_extraction(
    image: ANTsImage,
    modality: str = "t1",
    apply_mask: bool = True,
    verbose: bool = False,
) -> ANTsImage | tuple[ANTsImage, ANTsImage]:
    """Skull-strip an ANTsImage using ANTsPyNet brain extraction.

    Runs :func:`antspynet.utilities.brain_extraction` to obtain a brain
    segmentation/probability map, derives a binary brain mask from it, and
    (unless `apply_mask` is ``False``) applies the mask to `image`.

    The mask is derived in a `modality`-dependent way:

    * ``"t1threetissue"``: ``brain_extraction`` returns a segmentation; the
      brain label (1) is extracted.
    * ``"t1combined"``: the combined label image is thresholded to its brain
      labels (2-3).
    * any other modality: the returned probability map is thresholded at 0.5,
      then morphologically closed and hole-filled to produce a clean mask.

    Args:
        image:
            The :class:`ants.core.ANTsImage` to skull-strip.
        modality:
            Image modality passed to
            :func:`antspynet.utilities.brain_extraction`, e.g. ``"t1"``,
            ``"t2"``, ``"flair"``, ``"t1combined"``, or ``"t1threetissue"``.
            See the ANTsPyNet documentation for the full list.
        apply_mask:
            If ``False``, return only the brain mask without applying it.
        verbose:
            If ``True``, print progress from ANTsPyNet.

    Returns:
        If `apply_mask` is ``False``, the binary brain mask as an
        :class:`ants.core.ANTsImage`. Otherwise, a ``(brain, mask)`` tuple
        where ``brain`` is the skull-stripped image and ``mask`` is the brain
        mask.

    Raises:
        ValueError: If `image` is not an ANTsImage.
    """
    ensure_ants_image(image)
    bet = brain_extraction(image, modality=modality, verbose=verbose)
    if modality == "t1threetissue":
        mask = threshold_image(bet["segmentation_image"], 1, 1, 1, 0)
    elif modality == "t1combined":
        mask = threshold_image(bet, 2, 3, 1, 0)
    else:
        mask = threshold_image(bet, 0.5, 1, 1, 0)
        mask = morphology(mask, "close", 6).iMath_fill_holes()
    if not apply_mask:
        return mask
    return ants_apply_mask(image=image, mask=mask), mask

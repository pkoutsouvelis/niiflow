"""Similarity metrics for numpy arrays."""

from __future__ import annotations

__all__ = [
    "dice_similarity_coefficient",
]

import numpy as np

from .utils import validate_binary_values


def dice_similarity_coefficient(image: np.ndarray, target: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

    ``image`` and ``target`` must contain only ``0`` and ``1`` (``False`` and
    ``True`` for boolean arrays). Boolean, integer, and floating-point arrays
    are accepted when every value is exactly ``0`` or ``1``.
    The score is the foreground overlap

    ``2 * |image ∩ target| / (|image| + |target|)``,

    where ``|·|`` counts voxels equal to ``1`` or ``True``. When the
    denominator is zero, the coefficient is ``1.0`` if the numerator is also
    zero and ``0.0`` otherwise. Both foregrounds empty is that
    zero-numerator case. When exactly one mask is empty the coefficient is
    ``0.0``.

    Args:
        image:
            Binary mask.
        target:
            Binary mask with the same shape as ``image``.

    Returns:
        Dice coefficient in ``[0, 1]``.

    Raises:
        ValueError: If an argument is not a numpy array, contains values
            other than ``0`` and ``1``, or the shapes differ.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"`image` must be a numpy array, got {type(image).__name__}")
    if not isinstance(target, np.ndarray):
        raise ValueError(f"`target` must be a numpy array, got {type(target).__name__}")
    validate_binary_values(image, name="image")
    validate_binary_values(target, name="target")
    image_fg = image.astype(bool, copy=False)
    target_fg = target.astype(bool, copy=False)
    if image_fg.shape != target_fg.shape:
        raise ValueError(
            f"`target` shape {target_fg.shape} is incompatible with "
            f"`image` shape {image_fg.shape}"
        )
    intersection = int(np.count_nonzero(image_fg & target_fg))
    numerator = 2 * intersection
    denominator = int(np.count_nonzero(image_fg) + np.count_nonzero(target_fg))
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator

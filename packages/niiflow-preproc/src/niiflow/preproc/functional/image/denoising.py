"""Denoising functions for ANTsImage objects.

Thin wrapper around the ANTs denoising routine (:func:`ants.denoise_image`), which
implements the spatially adaptive non-local means filter of Manjon et al. (2010). See
the ANTsPy documentation for the full behaviour of the underlying call.
"""

from __future__ import annotations

__all__ = [
    "denoise_ants",
]

from typing import Any

from ants.core import ANTsImage
from ants.ops import denoise_image


def denoise_ants(**kwargs: Any) -> ANTsImage:
    """Denoise an ANTsImage using ANTs adaptive non-local means.

    Wraps :func:`ants.denoise_image`. All arguments are forwarded as keyword
    arguments, so the input is provided as ``image=...``.

    Args:
        **kwargs:
            Keyword arguments forwarded to :func:`ants.denoise_image`. Common
            ones include ``image`` (the :class:`ants.core.ANTsImage` to
            denoise; required), ``mask``, ``noise_model`` (``"Rician"`` or
            ``"Gaussian"``), ``shrink_factor``, ``p`` (patch radius), ``r``
            (search radius), and ``v`` (verbosity).

    Returns:
        The denoised :class:`ants.core.ANTsImage`.
    """
    return denoise_image(**kwargs)

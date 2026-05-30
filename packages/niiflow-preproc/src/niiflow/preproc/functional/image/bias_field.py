"""Bias-field correction functions for ANTsImage objects.

Thin wrapper around the ANTs N4 bias-field correction routine
(:func:`ants.n4_bias_field_correction`). See the ANTsPy documentation for the full
behaviour of the underlying call.
"""

from __future__ import annotations

__all__ = [
    "bias_field_correction_ants",
]

from typing import Any

from ants.core import ANTsImage
from ants.ops import n4_bias_field_correction


def bias_field_correction_ants(**kwargs: Any) -> ANTsImage:
    """Correct the bias field of an ANTsImage using the N4 algorithm.

    Wraps :func:`ants.n4_bias_field_correction`. All arguments are forwarded
    as keyword arguments, so the input is provided as ``image=...``.

    Args:
        **kwargs:
            Keyword arguments forwarded to
            :func:`ants.n4_bias_field_correction`. Common ones include
            ``image`` (the :class:`ants.core.ANTsImage` to correct; required),
            ``mask``, ``weight_mask``, ``shrink_factor``, ``convergence``, and
            ``return_bias_field``.

    Returns:
        The bias-corrected :class:`ants.core.ANTsImage` (or the estimated bias
        field itself when ``return_bias_field=True`` is passed).
    """
    return n4_bias_field_correction(**kwargs)

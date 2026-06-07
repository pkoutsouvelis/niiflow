"""End-to-end image preprocessing pipelines.

Thin wrapper around the ANTsPyNet brain preprocessing pipeline
(:func:`antspynet.utilities.preprocess_brain_image`). See the ANTsPyNet
documentation for the full behaviour of the underlying call.
"""

from __future__ import annotations

__all__ = [
    "ants_preprocess_brain_image",
]

from typing import Any, Literal, overload

from ants.core import ANTsImage
from antspynet.utilities import preprocess_brain_image


@overload
def ants_preprocess_brain_image(
    return_metadata: Literal[False],
    **kwargs: Any,
) -> ANTsImage: ...


@overload
def ants_preprocess_brain_image(
    return_metadata: Literal[True],
    **kwargs: Any,
) -> tuple[ANTsImage, dict[str, Any]]: ...


def ants_preprocess_brain_image(
    return_metadata: bool = True,
    **kwargs: Any,
) -> ANTsImage | tuple[ANTsImage, dict[str, Any]]:
    """Run the ANTsPyNet brain preprocessing pipeline.

    Wraps :func:`antspynet.utilities.preprocess_brain_image`, which chains
    optional steps such as intensity truncation, skull-stripping, template
    registration, N4 bias correction, denoising, intensity matching, and
    normalization. All pipeline options are forwarded as keyword arguments,
    so the input is provided as ``image=...``.

    Args:
        return_metadata:
            If ``True`` (default), return ``(preprocessed, metadata)`` where
            ``metadata`` holds any auxiliary outputs produced by the pipeline
            (e.g. ``brain_mask``, ``bias_field``, ``template_transforms``).
            If ``False``, return only the preprocessed
            :class:`ants.core.ANTsImage`.
        **kwargs:
            Keyword arguments forwarded to
            :func:`antspynet.utilities.preprocess_brain_image`. Common ones
            include ``image`` (required), ``truncate_intensity``,
            ``brain_extraction_modality``, ``template_transform_type``,
            ``template``, ``do_bias_correction``, ``do_denoising``,
            ``intensity_matching_type``, ``reference_image``,
            ``intensity_normalization_type``, and ``verbose``.

    Returns:
        If `return_metadata` is ``False``, the preprocessed
        :class:`ants.core.ANTsImage`. Otherwise, a ``(preprocessed, metadata)``
        tuple.

    Raises:
        KeyError: If the underlying pipeline does not return
            ``"preprocessed_image"``.
    """
    result = preprocess_brain_image(**kwargs)
    try:
        preprocessed = result["preprocessed_image"]
    except KeyError as exc:
        raise KeyError(
            'preprocess_brain_image must return a "preprocessed_image" key'
        ) from exc
    if not return_metadata:
        return preprocessed
    metadata = {
        key: value for key, value in result.items() if key != "preprocessed_image"
    }
    return preprocessed, metadata

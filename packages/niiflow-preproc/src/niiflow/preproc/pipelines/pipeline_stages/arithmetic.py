"""Pipeline stages wrapping :mod:`niiflow.preproc.functional.image.arithmetic`."""

from __future__ import annotations

__all__ = [
    "PointwiseArithmetic",
]

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ants.core import ANTsImage

from niiflow.preproc.functional.image.arithmetic import pointwise_arithmetic
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write

_SUPPORTED_OPS = frozenset({"mul", "add", "div", "sub"})


class PointwiseArithmetic(PipelineStage):
    """Apply an ordered sequence of pointwise arithmetic operations.

    Wraps
    :func:`~niiflow.preproc.functional.image.arithmetic.pointwise_arithmetic`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``operations`` — non-empty sequence of single-key operation mappings.
      Supported keys are ``mul``, ``add``, ``div``, and ``sub``. Each operand
      may be a scalar, a :class:`numpy.ndarray`, an :class:`ants.core.ANTsImage`,
      or a path to load as an image.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — transformed image.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image", "operations"})

    def check_params(self, params: dict[str, Any]) -> None:
        operations = params["operations"]
        if isinstance(operations, (str, bytes)) or not isinstance(operations, Sequence):
            raise TypeError(
                "`operations` must be a sequence of single-key mappings, got "
                f"{type(operations).__name__}"
            )
        if len(operations) == 0:
            raise ValueError("`operations` must contain at least one operation")
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                raise TypeError(
                    f"`operations[{index}]` must be a mapping, got "
                    f"{type(operation).__name__}"
                )
            if len(operation) != 1:
                raise ValueError(
                    f"`operations[{index}]` must contain exactly one key, "
                    f"got {len(operation)}"
                )
            key = next(iter(operation))
            if key not in _SUPPORTED_OPS:
                supported = ", ".join(sorted(repr(name) for name in _SUPPORTED_OPS))
                raise ValueError(
                    f"unsupported operation {key!r} at `operations[{index}]`; "
                    f"expected one of {{{supported}}}"
                )

    def _load_operand(self, value: Any, *, label: str) -> Any:
        if isinstance(value, ANTsImage):
            return value
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (int, float, np.number)):
            return value
        try:
            return ants_image_read(value, reorient=True)
        except Exception as e:
            raise ValueError(f"Failed to read {label} from {value}") from e

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            return self._load_operand(value, label="image")
        if key == "operations":
            loaded: list[dict[str, Any]] = []
            for index, operation in enumerate(value):
                ((op_key, op_value),) = operation.items()
                loaded.append(
                    {
                        op_key: self._load_operand(
                            op_value, label=f"operations[{index}].{op_key}"
                        )
                    }
                )
            return loaded
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        image = params["image"]
        operations = params["operations"]
        return {"out_image": pointwise_arithmetic(image, *operations)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")

"""Pipeline stages for quality control."""

from __future__ import annotations

__all__ = [
    "CheckDimensions",
    "CheckVoxelSpacing",
]

import operator
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, write_json, write_txt

_COMPARE_OPS = {
    ">": operator.gt,
    "greater": operator.gt,
    "<": operator.lt,
    "less": operator.lt,
    "==": operator.eq,
    "equal": operator.eq,
    "!=": operator.ne,
    "not_equal": operator.ne,
    ">=": operator.ge,
    "greater_equal": operator.ge,
    "<=": operator.le,
    "less_equal": operator.le,
}


def _resolve_op(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"`op` must be a string, got {type(value).__name__}")
    op = value.strip().lower()
    if op not in _COMPARE_OPS:
        raise ValueError(
            f"Unknown comparison operator {value!r}; expected one of "
            f"{sorted(_COMPARE_OPS)}"
        )
    return op


class CheckVoxelSpacing(PipelineStage):
    """Check image voxel spacing against an expected threshold.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``expected`` — per-axis spacing ``(sx, sy, sz, ...)``; a scalar is not accepted.
    * ``op`` — comparison operator (default ``"=="``). Symbolic (``">="``) or
      readable (``"greater_equal"``) forms are accepted.
    * ``log`` — when ``True``, log the observed value and pass/fail result
      (requires stage ``verbose=True``).
    * ``id`` — optional string included in ``report`` when set (e.g.
      ``"ctx.run_id"``); omitted from ``report`` when ``None``.

    **Outputs** (from :meth:`forward`):

    * ``passed`` — whether the spacing satisfies the comparison.
    * ``value`` — observed spacing as a tuple of floats.
    * ``report`` — JSON-serialisable summary (``check``, ``passed``, ``value``,
      and ``id`` when provided).

    **Persistence** (``save_options``):

    * ``passed`` — ``.txt`` path; one line ``True`` or ``False``.
    * ``value`` — ``.txt`` path; comma-separated observed spacing per axis.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"image", "expected"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "expected":
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                expected = tuple(float(v) for v in value)
                if not expected:
                    raise ValueError(
                        "`expected` must be a non-empty sequence of numbers"
                    )
                return expected
            raise TypeError(
                f"`expected` must be a sequence of numbers, got {type(value).__name__}"
            )

        if key == "op":
            return _resolve_op(value)

        if key == "log":
            if not isinstance(value, bool):
                raise TypeError(f"`log` must be a boolean, got {type(value).__name__}")
            return value

        if key == "id":
            if value is None:
                return None
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"`id` must be a non-empty string or None, got {value!r}"
                )
            return value

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        image = params["image"]
        expected = params["expected"]
        op = params.get("op", "==")
        log = params.get("log", False)
        id = params.get("id")

        value = tuple(float(s) for s in image.spacing)
        compare = _COMPARE_OPS[op]

        if len(value) != len(expected):
            raise ValueError(
                f"Spacing length {len(value)} does not match "
                f"expected length {len(expected)}"
            )
        passed = all(compare(a, e) for a, e in zip(value, expected))

        if log and self.verbose:
            self.log(
                f"[QC spacing] value={value} expected={expected} op={op!r} "
                f"passed={passed}"
            )

        report: dict[str, Any] = {
            "check": type(self).__name__,
            "passed": passed,
            "value": list(value),
        }
        if id is not None:
            report["id"] = id
        return {"passed": passed, "value": value, "report": report}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "passed":
            return write_txt(str(value), output_path)
        if key == "value":
            return write_txt(",".join(str(item) for item in value), output_path)
        if key == "report":
            return write_json(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class CheckDimensions(PipelineStage):
    """Check image dimensions against an expected threshold.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``expected`` — per-axis shape ``(nx, ny, nz, ...)``; a scalar is not accepted.
    * ``op`` — comparison operator (default ``"=="``). Symbolic (``">="``) or
      readable (``"greater_equal"``) forms are accepted.
    * ``log`` — when ``True``, log the observed value and pass/fail result
      (requires stage ``verbose=True``).
    * ``id`` — optional string included in ``report`` when set (e.g.
      ``"ctx.run_id"``); omitted from ``report`` when ``None``.

    **Outputs** (from :meth:`forward`):

    * ``passed`` — whether the shape satisfies the comparison.
    * ``value`` — observed shape as a tuple of integers.
    * ``report`` — JSON-serialisable summary (``check``, ``passed``, ``value``,
      and ``id`` when provided).

    **Persistence** (``save_options``):

    * ``passed`` — ``.txt`` path; one line ``True`` or ``False``.
    * ``value`` — ``.txt`` path; comma-separated observed shape per axis.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"image", "expected"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "expected":
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                expected = tuple(int(v) for v in value)
                if not expected:
                    raise ValueError(
                        "`expected` must be a non-empty sequence of integers"
                    )
                return expected
            raise TypeError(
                f"`expected` must be a sequence of integers, got {type(value).__name__}"
            )

        if key == "op":
            return _resolve_op(value)

        if key == "log":
            if not isinstance(value, bool):
                raise TypeError(f"`log` must be a boolean, got {type(value).__name__}")
            return value

        if key == "id":
            if value is None:
                return None
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"`id` must be a non-empty string or None, got {value!r}"
                )
            return value

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        image = params["image"]
        expected = params["expected"]
        op = params.get("op", "==")
        log = params.get("log", False)
        id = params.get("id")

        value = tuple(int(s) for s in image.shape)
        compare = _COMPARE_OPS[op]

        if len(value) != len(expected):
            raise ValueError(
                f"Shape length {len(value)} does not match "
                f"expected length {len(expected)}"
            )
        passed = all(compare(a, e) for a, e in zip(value, expected))

        if log and self.verbose:
            self.log(
                f"[QC dimensions] value={value} expected={expected} op={op!r} "
                f"passed={passed}"
            )

        report: dict[str, Any] = {
            "check": type(self).__name__,
            "passed": passed,
            "value": list(value),
        }
        if id is not None:
            report["id"] = id
        return {"passed": passed, "value": value, "report": report}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "passed":
            return write_txt(str(value), output_path)
        if key == "value":
            return write_txt(",".join(str(item) for item in value), output_path)
        if key == "report":
            return write_json(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")

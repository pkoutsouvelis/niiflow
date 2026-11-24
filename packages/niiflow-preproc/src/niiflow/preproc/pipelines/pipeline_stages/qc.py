"""Pipeline stages for quality control."""

from __future__ import annotations

__all__ = [
    "AggregateChecks",
    "CheckDimensions",
    "CheckMetric",
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
_SHARED_QC_PARAMS = frozenset({"op", "log", "raise_on_fail", "id"})
_AGGREGATE_RULES = frozenset({"AND", "OR"})


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


def _load_image(value: Any) -> ANTsImage:
    """Load ``value`` as an ANTs image when it is not one already."""
    if isinstance(value, ANTsImage):
        return value
    try:
        return ants_image_read(value, reorient=True)
    except Exception as e:
        raise ValueError(f"Failed to read image from {value}") from e


def _load_shared_qc_params(key: str, value: Any) -> Any:
    """Load ``op``, ``log``, ``raise_on_fail``, and ``id``."""
    if key == "op":
        return _resolve_op(value)
    if key == "log":
        if not isinstance(value, bool):
            raise TypeError(f"`log` must be a boolean, got {type(value).__name__}")
        return value
    if key == "raise_on_fail":
        if not isinstance(value, bool):
            raise TypeError(
                f"`raise_on_fail` must be a boolean, got {type(value).__name__}"
            )
        return value
    if key == "id":
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise TypeError(f"`id` must be a non-empty string or None, got {value!r}")
        return value
    raise KeyError(key)


def _qc_outputs(
    check: str,
    passed: bool,
    value: Any,
    *,
    id: str | None = None,
    raise_on_fail: bool = False,
) -> dict[str, Any]:
    """Build the ``passed`` / ``value`` / ``report`` triple shared by QC stages.

    ``report`` records the check label, the pass/fail flag, and the observed
    value. ``id`` is included when set. A failing check raises when
    ``raise_on_fail`` is true.
    """
    report_value = list(value) if isinstance(value, tuple) else value
    report: dict[str, Any] = {
        "check": check,
        "passed": passed,
        "value": report_value,
    }
    if id is not None:
        report["id"] = id
    if raise_on_fail and not passed:
        raise RuntimeError(f"{check} failed QC check (passed=False) with value={value}")
    return {"passed": passed, "value": value, "report": report}


def _save_qc_output(stage: str, key: str, value: Any, output_path: Path) -> Path:
    """Persist ``passed`` and ``value`` as text and ``report`` as JSON.

    Sequence values are written as comma-separated text. A scalar value is
    written as one line.
    """
    if key == "passed":
        return write_txt(str(value), output_path)
    if key == "value":
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            text = ",".join(str(item) for item in value)
        else:
            text = str(value)
        return write_txt(text, output_path)
    if key == "report":
        return write_json(value, output_path)
    raise KeyError(f"Unknown output key {key!r} for {stage}")


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
    * ``raise_on_fail`` — when ``True``, raise from :meth:`forward` if
      ``passed`` is ``False`` (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``passed`` — whether the spacing satisfies the comparison.
    * ``value`` — observed spacing as a tuple of floats.
    * ``report`` — JSON-serialisable summary (``check``, ``passed``, ``value``,
      and ``id`` when provided).

    **Persistence** (``save_outputs``):

    * ``passed`` — ``.txt`` path; one line ``True`` or ``False``.
    * ``value`` — ``.txt`` path; comma-separated observed spacing per axis.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"image", "expected"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            return _load_image(value)

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

        if key in _SHARED_QC_PARAMS:
            return _load_shared_qc_params(key, value)

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

        return _qc_outputs(
            type(self).__name__,
            passed,
            value,
            id=id,
            raise_on_fail=params.get("raise_on_fail", False),
        )

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        return _save_qc_output(type(self).__name__, key, value, output_path)


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
    * ``raise_on_fail`` — when ``True``, raise from :meth:`forward` if
      ``passed`` is ``False`` (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``passed`` — whether the shape satisfies the comparison.
    * ``value`` — observed shape as a tuple of integers.
    * ``report`` — JSON-serialisable summary (``check``, ``passed``, ``value``,
      and ``id`` when provided).

    **Persistence** (``save_outputs``):

    * ``passed`` — ``.txt`` path; one line ``True`` or ``False``.
    * ``value`` — ``.txt`` path; comma-separated observed shape per axis.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"image", "expected"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            return _load_image(value)

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

        if key in _SHARED_QC_PARAMS:
            return _load_shared_qc_params(key, value)

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

        return _qc_outputs(
            type(self).__name__,
            passed,
            value,
            id=id,
            raise_on_fail=params.get("raise_on_fail", False),
        )

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        return _save_qc_output(type(self).__name__, key, value, output_path)


class CheckMetric(PipelineStage):
    """Compare one scalar metric against an expected threshold.

    **Parameters** (``params``):

    * ``metric`` — observed scalar score.
    * ``expected`` — scalar threshold. A sequence is not accepted.
    * ``op`` — comparison operator (default ``">="``). Symbolic (``">="``) or
      readable (``"greater_equal"``) forms are accepted.
    * ``name`` — optional label appended to the check name in logs, the
      failure message, and ``report["check"]``. ``"DSC"`` yields
      ``CheckMetric-DSC``. Omitted or ``None`` leaves the class name.
    * ``log`` — when ``True``, log the observed value and pass/fail result
      (requires stage ``verbose=True``).
    * ``id`` — optional string included in ``report`` when set; omitted when
      ``None``.
    * ``raise_on_fail`` — when ``True``, raise from :meth:`forward` if
      ``passed`` is ``False`` (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``passed`` — whether ``op(metric, expected)`` holds.
    * ``value`` — the observed metric as a float.
    * ``report`` — JSON-serialisable summary (``check``, ``passed``, ``value``,
      and ``id`` when provided).

    **Persistence** (``save_outputs``):

    * ``passed`` — ``.txt`` path; one line ``True`` or ``False``.
    * ``value`` — ``.txt`` path; one line with the score.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"metric", "expected"})

    def load_param(self, key: str, value: Any) -> Any:
        if key in {"metric", "expected"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{key}` must be a number, got {type(value).__name__}")
            number = float(value)
            if number != number or number in {float("inf"), float("-inf")}:
                raise ValueError(f"`{key}` must be finite, got {value!r}")
            return number

        if key == "name":
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"`name` must be a non-empty string or None, got {value!r}"
                )
            return value.strip()

        if key in _SHARED_QC_PARAMS:
            return _load_shared_qc_params(key, value)

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        value = params["metric"]
        expected = params["expected"]
        op = params.get("op", ">=")
        log = params.get("log", False)
        id = params.get("id")
        name = params.get("name")
        label = type(self).__name__ if not name else f"{type(self).__name__}-{name}"
        passed = bool(_COMPARE_OPS[op](value, expected))

        if log and self.verbose:
            self.log(
                f"[QC {label}] value={value} expected={expected} op={op!r} "
                f"passed={passed}"
            )

        return _qc_outputs(
            label,
            passed,
            value,
            id=id,
            raise_on_fail=params.get("raise_on_fail", False),
        )

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        return _save_qc_output(type(self).__name__, key, value, output_path)


class AggregateChecks(PipelineStage):
    """Combine QC pass flags with AND or OR.

    **Parameters** (``params``):

    * ``flags`` — non-empty sequence of boolean ``passed`` values from earlier
      QC stages.
    * ``rule`` — ``"AND"`` requires every flag to pass; ``"OR"`` requires at
      least one. Matching is case-insensitive.
    * ``log`` — when ``True``, log the flags and pass/fail result (requires
      stage ``verbose=True``).
    * ``id`` — optional string included in ``report`` when set; omitted when
      ``None``.
    * ``raise_on_fail`` — when ``True``, raise from :meth:`forward` if
      ``passed`` is ``False`` (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``passed`` — whether the flags satisfy ``rule``.
    * ``value`` — the flags as a tuple of booleans.
    * ``report`` — JSON-serialisable summary (``check``, ``passed``, ``value``,
      and ``id`` when provided).

    **Persistence** (``save_outputs``):

    * ``passed`` — ``.txt`` path; one line ``True`` or ``False``.
    * ``value`` — ``.txt`` path; comma-separated flags.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"flags", "rule"})

    def check_params(self, params: dict[str, Any]) -> None:
        rule = params["rule"]
        if not isinstance(rule, str):
            raise TypeError(f"`rule` must be a string, got {type(rule).__name__}")
        if rule.strip().upper() not in _AGGREGATE_RULES:
            raise ValueError(
                f"`rule` must be one of {sorted(_AGGREGATE_RULES)}, got {rule!r}"
            )
        flags = params["flags"]
        if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
            raise TypeError(
                f"`flags` must be a sequence of booleans, got {type(flags).__name__}"
            )
        if len(flags) == 0:
            raise ValueError("`flags` must be a non-empty sequence of booleans")

    def load_param(self, key: str, value: Any) -> Any:
        if key == "flags":
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise TypeError(
                    f"`flags` must be a sequence of booleans, "
                    f"got {type(value).__name__}"
                )
            flags = tuple(value)
            if not flags:
                raise ValueError("`flags` must be a non-empty sequence of booleans")
            for index, flag in enumerate(flags):
                if not isinstance(flag, bool):
                    raise TypeError(
                        f"`flags[{index}]` must be a boolean, "
                        f"got {type(flag).__name__}"
                    )
            return flags

        if key == "rule":
            if not isinstance(value, str):
                raise TypeError(f"`rule` must be a string, got {type(value).__name__}")
            rule = value.strip().upper()
            if rule not in _AGGREGATE_RULES:
                raise ValueError(
                    f"`rule` must be one of {sorted(_AGGREGATE_RULES)}, got {value!r}"
                )
            return rule

        if key in _SHARED_QC_PARAMS:
            return _load_shared_qc_params(key, value)

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        value = tuple(params["flags"])
        rule = params["rule"]
        log = params.get("log", False)
        id = params.get("id")
        passed = all(value) if rule == "AND" else any(value)

        if log and self.verbose:
            self.log(
                f"[QC {type(self).__name__}] flags={value} "
                f"rule={rule!r} passed={passed}"
            )

        return _qc_outputs(
            type(self).__name__,
            passed,
            value,
            id=id,
            raise_on_fail=params.get("raise_on_fail", False),
        )

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        return _save_qc_output(type(self).__name__, key, value, output_path)

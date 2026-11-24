"""Pipeline stages for image similarity metrics."""

from __future__ import annotations

__all__ = [
    "ANTsSimilarityMetric",
    "DiceSimilarityCoefficient",
]

from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.metrics import (
    ants_similarity_metrics,
    dice_similarity_coefficient,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, write_json, write_txt


def _load_image(value: Any, *, label: str) -> ANTsImage:
    """Load ``value`` as an ANTs image when it is not one already."""
    if isinstance(value, ANTsImage):
        return value
    try:
        return ants_image_read(value, reorient=True)
    except Exception as e:
        raise ValueError(f"Failed to read {label} from {value}") from e


def _metric_outputs(metric: str, value: float) -> dict[str, Any]:
    """Build the ``value`` / ``report`` pair shared by metric stages.

    ``report`` records the stage class name as ``metric`` and the score as
    ``value``. It has no pass/fail flag.
    """
    score = float(value)
    return {
        "value": score,
        "report": {
            "metric": metric,
            "value": score,
        },
    }


def _save_metric_output(stage: str, key: str, value: Any, output_path: Path) -> Path:
    """Persist ``value`` as text and ``report`` as JSON."""
    if key == "value":
        return write_txt(str(value), output_path)
    if key == "report":
        return write_json(value, output_path)
    raise KeyError(f"Unknown output key {key!r} for {stage}")


def _require_metric_name(value: Any) -> str:
    """Return ``value`` when it is a non-empty ANTs metric name."""
    if not isinstance(value, str):
        raise TypeError(f"`metric` must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError("`metric` must be a non-empty ANTs metric name")
    return value


class ANTsSimilarityMetric(PipelineStage):
    """Compute one ANTs image similarity score.

    Wraps
    :func:`~niiflow.preproc.functional.image.metrics.ants_similarity_metrics`.
    ``metric`` is one ANTs ``metric_type`` string per call. Check the ANTsPy
    documentation for ``ants.image_similarity`` for the available metrics.
    Common choices include ``"MattesMutualInformation"``, ``"Correlation"``,
    and ``"MeanSquares"``. The returned score is higher-is-better: the ANTs
    distance is negated.

    **Parameters** (``params``):

    * ``image`` — moving :class:`ants.core.ANTsImage` or path, already in
      ``target`` space.
    * ``target`` — fixed :class:`ants.core.ANTsImage` or path.
    * ``metric`` — one ANTs metric name, for example
      ``"MattesMutualInformation"``.
    * ``mask`` — optional mask :class:`ants.core.ANTsImage` or path in the
      same space. ``None`` or omitted evaluates the full image.
    * Additional kwargs are forwarded to ``ants_similarity_metrics`` (for
      example ``sampling_strategy`` and ``sampling_percentage``).

    **Outputs** (from :meth:`forward`):

    * ``value`` — float similarity score. Higher is better.
    * ``report`` — JSON-serialisable summary with ``metric`` (this stage's
      class name) and ``value``. No pass/fail flag.

    **Persistence** (``save_outputs``):

    * ``value`` — ``.txt`` path; one line with the score.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"image", "target", "metric"})

    def check_params(self, params: dict[str, Any]) -> None:
        _require_metric_name(params["metric"])

    def load_param(self, key: str, value: Any) -> Any:
        if key in {"image", "target"}:
            return _load_image(value, label=key)
        if key == "mask":
            if value is None:
                return None
            return _load_image(value, label=key)
        if key == "metric":
            return _require_metric_name(value)
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        score = ants_similarity_metrics(**params)
        return _metric_outputs(type(self).__name__, score)

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        return _save_metric_output(type(self).__name__, key, value, output_path)


class DiceSimilarityCoefficient(PipelineStage):
    """Dice similarity coefficient between two binary masks.

    Wraps
    :func:`~niiflow.preproc.functional.image.metrics.dice_similarity_coefficient`.
    ``image`` and ``target`` must contain only ``0`` and ``1``. Boolean and
    integer masks are scored directly. Floating-point masks are accepted only
    when every voxel is exactly ``0`` or ``1``. If users have multiple
    labels, they should use
    :class:`~niiflow.preproc.pipelines.pipeline_stages.masks.RelabelMask` in
    advance to create the desired masks.

    **Parameters** (``params``):

    * ``image`` — mask :class:`ants.core.ANTsImage` or path.
    * ``target`` — mask :class:`ants.core.ANTsImage` or path on the same grid.

    **Outputs** (from :meth:`forward`):

    * ``value`` — float Dice coefficient in ``[0, 1]``.
    * ``report`` — JSON-serialisable summary with ``metric`` (this stage's
      class name) and ``value``. No pass/fail flag.

    **Persistence** (``save_outputs``):

    * ``value`` — ``.txt`` path; one line with the score.
    * ``report`` — ``.json`` path.
    """

    REQUIRED_PARAMS = frozenset({"image", "target"})

    def load_param(self, key: str, value: Any) -> Any:
        if key in {"image", "target"}:
            return _load_image(value, label=key)
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        score = dice_similarity_coefficient(
            image=params["image"],
            target=params["target"],
        )
        return _metric_outputs(type(self).__name__, score)

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        return _save_metric_output(type(self).__name__, key, value, output_path)

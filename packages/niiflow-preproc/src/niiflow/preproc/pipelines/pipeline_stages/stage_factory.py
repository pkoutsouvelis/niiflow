"""Stage discovery and instantiation for config-driven pipelines."""

from __future__ import annotations

__all__ = [
    "create_stage",
    "discover_stage_classes",
]

import importlib
import inspect
from typing import Any

from .pipeline_stage import PipelineStage

_STAGE_MODULES: tuple[str, ...] = (
    "niiflow.preproc.pipelines.pipeline_stages.bias_field",
    "niiflow.preproc.pipelines.pipeline_stages.croppad",
    "niiflow.preproc.pipelines.pipeline_stages.denoising",
    "niiflow.preproc.pipelines.pipeline_stages.intensity_normalization",
    "niiflow.preproc.pipelines.pipeline_stages.masks",
    "niiflow.preproc.pipelines.pipeline_stages.pipelines",
    "niiflow.preproc.pipelines.pipeline_stages.qc",
    "niiflow.preproc.pipelines.pipeline_stages.registration",
    "niiflow.preproc.pipelines.pipeline_stages.resampling",
    "niiflow.preproc.pipelines.pipeline_stages.skull_stripping",
    "niiflow.preproc.pipelines.pipeline_stages.utility",
)


def discover_stage_classes() -> dict[str, type[PipelineStage]]:
    """Return concrete :class:`PipelineStage` types keyed by class name.

    Scans the modules listed in :data:`_STAGE_MODULES` and excludes the abstract base
    :class:`PipelineStage`. Orchestrators such as :class:`Compose` are omitted by not
    listing their modules here.
    """
    registry: dict[str, type[PipelineStage]] = {}
    for module_name in _STAGE_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is PipelineStage:
                continue
            if inspect.isabstract(obj):
                continue
            if not issubclass(obj, PipelineStage):
                continue
            registry[name] = obj
    return registry


def _resolve_stage_class(
    name: str, registry: dict[str, type[PipelineStage]]
) -> type[PipelineStage]:
    if not isinstance(name, str) or not name:
        raise ValueError(f"Step `name` must be a non-empty string, got {name!r}.")
    if name == "Compose":
        raise ValueError(
            "`Compose` cannot be built from pipeline config; list steps "
            "explicitly or call `Compose()` directly."
        )
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown pipeline stage {name!r}; expected one of {sorted(registry)}."
        ) from exc


def create_stage(
    stage_name: str,
    stage_kwargs: dict[str, Any] | None = None,
    *,
    registry: dict[str, type[PipelineStage]] | None = None,
) -> PipelineStage:
    """Instantiate a :class:`PipelineStage` from a registered class name and constructor
    kwargs.

    ``stage_name`` must match a concrete stage listed by
    :func:`discover_stage_classes`, for example ``"ANTsDenoise"``. ``stage_kwargs`` is
    forwarded to that class's constructor. :class:`Compose` is not constructible this
    way; build it directly instead.

    Args:
        stage_name: Registered stage class name.
        stage_kwargs: Constructor keyword arguments, typically ``params``,
            ``save_outputs``, and ``verbose``. ``None`` is treated as ``{}``.
        registry: Optional pre-built name-to-class mapping. Defaults to
            :func:`discover_stage_classes`.

    Returns:
        The instantiated stage.

    Raises:
        TypeError: If ``stage_kwargs`` is neither a dictionary nor ``None``, or if the
            stage constructor rejects the supplied kwargs.
        ValueError: If ``stage_name`` is not a registered concrete stage.
    """
    if stage_kwargs is not None and not isinstance(stage_kwargs, dict):
        raise TypeError(
            f"`stage_kwargs` must be a dictionary or None, got "
            f"{type(stage_kwargs).__name__}."
        )

    stage_cls = _resolve_stage_class(stage_name, registry or discover_stage_classes())
    kwargs = {} if stage_kwargs is None else dict(stage_kwargs)

    try:
        return stage_cls(**kwargs)
    except TypeError as exc:
        raise TypeError(f"Failed to instantiate stage {stage_name!r}: {exc}") from exc

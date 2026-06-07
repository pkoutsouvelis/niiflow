"""Stage discovery and instantiation for config-driven pipelines."""

from __future__ import annotations

__all__ = [
    "create_stage",
    "discover_stage_classes",
]

import inspect
from typing import Any

import niiflow.preproc.pipelines.pipeline_stages as pipeline_stages
from niiflow.preproc.pipelines.pipeline_stages import Compose, PipelineStage


def discover_stage_classes() -> dict[str, type[PipelineStage]]:
    """Return concrete :class:`PipelineStage` types keyed by class name.

    Scans :mod:`niiflow.preproc.pipelines.pipeline_stages` and excludes the abstract
    base :class:`PipelineStage` and :class:`Compose`.
    """
    registry: dict[str, type[PipelineStage]] = {}
    for name, obj in inspect.getmembers(pipeline_stages, inspect.isclass):
        if obj is PipelineStage or obj is Compose:
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
            f"Unknown pipeline stage {name!r}; expected one of " f"{sorted(registry)}."
        ) from exc


def create_stage(
    step_label: str,
    step_spec: Any,
    *,
    registry: dict[str, type[PipelineStage]],
    default_verbose: bool,
) -> PipelineStage:
    """Instantiate a :class:`PipelineStage` from a step specification."""
    if isinstance(step_spec, PipelineStage):
        raise TypeError(
            f"Step {step_label!r} must be a configuration dictionary, not an "
            f"instantiated {type(step_spec).__name__}."
        )
    if not isinstance(step_spec, dict):
        raise TypeError(
            f"Step {step_label!r} must be a dictionary, got "
            f"{type(step_spec).__name__}."
        )
    if "name" not in step_spec:
        raise ValueError(f"Step {step_label!r} is missing required key 'name'.")

    stage_cls = _resolve_stage_class(step_spec["name"], registry)

    init_kwargs = {key: value for key, value in step_spec.items() if key != "name"}

    sig = inspect.signature(stage_cls.__init__)
    if "verbose" in sig.parameters and "verbose" not in init_kwargs:
        init_kwargs["verbose"] = default_verbose

    try:
        bound = sig.bind_partial(**init_kwargs)
        bound.apply_defaults()
        return stage_cls(*bound.args, **bound.kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate stage {step_spec['name']!r} for step "
            f"{step_label!r}: {exc}"
        ) from exc

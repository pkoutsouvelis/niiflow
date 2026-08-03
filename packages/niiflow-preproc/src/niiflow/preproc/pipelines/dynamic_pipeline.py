"""Pipeline discovery and instantiation for config-driven pipelines."""

from __future__ import annotations

__all__ = [
    "dynamic_pipeline",
]

import inspect
from typing import Any

from niiflow.preproc.pipelines.pipeline_stages import (
    Compose,
    PipelineStage,
    RuntimeContext,
    create_stage,
    discover_stage_classes,
)

_VALID_PIPELINE_KEYS_LIST = frozenset({"steps", "verbose"})
_VALID_PIPELINE_KEYS_MAPPING = frozenset({"steps", "order", "verbose"})


def _stage_from_spec(
    step_label: str,
    step_spec: Any,
    *,
    registry: dict[str, type[PipelineStage]],
    default_verbose: bool,
) -> PipelineStage:
    """Instantiate a :class:`PipelineStage` from a step specification.

    Validates the step-spec shape — the config format owned by this module — then
    delegates class lookup and construction to
    :func:`~niiflow.preproc.pipelines.pipeline_stages.create_stage`.
    """
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

    stage_name = step_spec["name"]
    stage_kwargs = {key: value for key, value in step_spec.items() if key != "name"}

    stage_cls = registry.get(stage_name) if isinstance(stage_name, str) else None
    if stage_cls is not None:
        sig = inspect.signature(stage_cls.__init__)
        if "verbose" in sig.parameters and "verbose" not in stage_kwargs:
            stage_kwargs["verbose"] = default_verbose

    try:
        return create_stage(stage_name, stage_kwargs, registry=registry)
    except TypeError as exc:
        raise TypeError(f"{exc} (step {step_label!r})") from exc


def _normalize_pipeline(
    pipeline: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, Any]]], list[str] | None]:
    """Return ordered ``(step_label, step_spec)`` pairs and optional step ids."""
    if not isinstance(pipeline, dict):
        raise TypeError(
            f"`pipeline` must be a dictionary, got {type(pipeline).__name__}."
        )
    if "steps" not in pipeline:
        raise ValueError("`pipeline` must contain a 'steps' entry.")

    steps_raw = pipeline["steps"]
    if isinstance(steps_raw, list):
        if "order" in pipeline:
            raise ValueError(
                "`order` is only supported when `steps` is a mapping; a list of "
                "steps already defines execution order by position."
            )
        unknown = sorted(set(pipeline) - _VALID_PIPELINE_KEYS_LIST)
        if unknown:
            raise ValueError(
                f"Unknown pipeline key(s): {unknown}. When `steps` is a list, "
                "allowed top-level keys are: ['steps', 'verbose']."
            )
        return [
            (f"steps[{index}]", step_spec) for index, step_spec in enumerate(steps_raw)
        ], None

    if isinstance(steps_raw, dict):
        unknown = sorted(set(pipeline) - _VALID_PIPELINE_KEYS_MAPPING)
        if unknown:
            raise ValueError(
                f"Unknown pipeline key(s): {unknown}. When `steps` is a mapping, "
                "allowed top-level keys are: ['order', 'steps', 'verbose']."
            )
        for step_id in steps_raw:
            if not isinstance(step_id, str) or not step_id:
                raise TypeError(f"Step ids must be non-empty strings, got {step_id!r}.")

        order_raw = pipeline.get("order")
        if order_raw is None:
            order = list(steps_raw.keys())
        else:
            if not isinstance(order_raw, list) or not order_raw:
                raise ValueError(
                    "`order` must be a non-empty list of step ids when supplied."
                )

            order = []
            seen: set[str] = set()
            for index, step_id in enumerate(order_raw):
                if not isinstance(step_id, str) or not step_id:
                    raise TypeError(
                        f"`order[{index}]` must be a non-empty string, got "
                        f"{step_id!r}."
                    )
                if step_id in seen:
                    raise ValueError(f"`order` contains duplicate step id {step_id!r}.")
                seen.add(step_id)
                order.append(step_id)

            missing = set(order) - steps_raw.keys()
            if missing:
                raise ValueError(
                    f"`order` references unknown step id(s): {sorted(missing)}."
                )
            unused = set(steps_raw.keys()) - set(order)
            if unused:
                raise ValueError(
                    f"`steps` contains id(s) not listed in `order`: "
                    f"{sorted(unused)}."
                )

        return [(step_id, steps_raw[step_id]) for step_id in order], order

    raise TypeError(
        f"`steps` must be a list or mapping, got {type(steps_raw).__name__}."
    )


def dynamic_pipeline(pipeline_spec: dict[str, Any], run_id: str | None = None) -> None:
    """Build and immediately execute a pipeline for one active file.

    Unlike the previous :func:`create_pipeline` factory (which returned a reusable
    :class:`Compose` object), this is an **executable forward function**: it
    instantiates stages, runs them under a fresh :class:`RuntimeContext`, and
    returns ``None``.

    The specification must contain ``steps``, either as:

    * a **mapping** keyed by step id, with optional top-level ``order``. When
      ``order`` is omitted, steps run in mapping insertion order (stable in
      Python 3.7+).
    * an **ordered list** of step specifications. List position defines
      execution order; step ids are assigned automatically at runtime by
      :class:`Compose`.

    Each step specification is a dictionary with:

    * ``name`` — concrete stage class name registered in
      :mod:`niiflow.preproc.pipelines.pipeline_stages` (not ``Compose``).
    * remaining keys — forwarded as keyword arguments to the stage constructor.
      Pipeline-level ``verbose`` (default ``True``) is used when a step omits
      its own ``verbose`` entry.

    Pre-instantiated :class:`PipelineStage` objects are not accepted; use
    :class:`Compose` directly when assembling stages in Python.

    Args:
        pipeline_spec: Specification of pipeline stages and parameters
            (``steps``, optional ``order`` / ``verbose``).
        run_id: A run id to pass to the pipeline's runtime context (default ``None``).
    """
    pipeline_verbose = pipeline_spec.get("verbose", True)
    if not isinstance(pipeline_verbose, bool):
        raise TypeError(
            f"Pipeline-level `verbose` must be a boolean, got "
            f"{type(pipeline_verbose).__name__}."
        )

    steps, step_ids = _normalize_pipeline(pipeline_spec)
    registry = discover_stage_classes()

    built = [
        _stage_from_spec(
            step_label,
            step_spec,
            registry=registry,
            default_verbose=pipeline_verbose,
        )
        for step_label, step_spec in steps
    ]

    pipeline = Compose(built, step_ids=step_ids, verbose=pipeline_verbose)
    ctx = RuntimeContext(run_id=run_id)
    pipeline.run(ctx)

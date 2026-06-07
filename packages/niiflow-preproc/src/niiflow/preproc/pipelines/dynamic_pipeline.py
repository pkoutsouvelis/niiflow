"""Config-driven dynamic pipeline builder."""

from __future__ import annotations

__all__ = [
    "build_dynamic_pipeline",
]

from typing import Any

from niiflow.preproc.pipelines import stage_factory
from niiflow.preproc.pipelines.pipeline_stages import Compose

_VALID_CONFIG_KEYS_LIST = frozenset({"steps", "verbose"})
_VALID_CONFIG_KEYS_MAPPING = frozenset({"order", "steps", "verbose"})


def _normalize_config(
    config: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, Any]]], list[str] | None]:
    """Return ordered ``(step_label, step_spec)`` pairs and optional step ids."""
    if not isinstance(config, dict):
        raise TypeError(f"`config` must be a dictionary, got {type(config).__name__}.")
    if "steps" not in config:
        raise ValueError("`config` must contain a 'steps' entry.")

    steps_raw = config["steps"]
    if isinstance(steps_raw, list):
        unknown = sorted(set(config) - _VALID_CONFIG_KEYS_LIST)
        if unknown:
            raise ValueError(
                f"Unknown pipeline config key(s): {unknown}. When `steps` is a "
                "list, allowed top-level keys are: ['steps', 'verbose']."
            )
        return [
            (f"steps[{index}]", step_spec) for index, step_spec in enumerate(steps_raw)
        ], None

    if isinstance(steps_raw, dict):
        unknown = sorted(set(config) - _VALID_CONFIG_KEYS_MAPPING)
        if unknown:
            raise ValueError(
                f"Unknown pipeline config key(s): {unknown}. When `steps` is a "
                "mapping, allowed top-level keys are: ['order', 'steps', 'verbose']."
            )
        for step_id in steps_raw:
            if not isinstance(step_id, str) or not step_id:
                raise TypeError(f"Step ids must be non-empty strings, got {step_id!r}.")

        order_raw = config.get("order")
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


def build_dynamic_pipeline(config: dict[str, Any]) -> Compose:
    """Build a :class:`Compose` pipeline from a configuration dictionary.

    The configuration must contain ``steps``, either as:

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
        config: Pipeline specification.

    Returns:
        A :class:`Compose` instance ready to :meth:`~PipelineStage.run`.
    """
    pipeline_verbose = config.get("verbose", True)
    if not isinstance(pipeline_verbose, bool):
        raise TypeError(
            f"Pipeline-level `verbose` must be a boolean, got "
            f"{type(pipeline_verbose).__name__}."
        )

    entries, step_ids = _normalize_config(config)
    registry = stage_factory.discover_stage_classes()

    built = [
        stage_factory.create_stage(
            step_label,
            step_spec,
            registry=registry,
            default_verbose=pipeline_verbose,
        )
        for step_label, step_spec in entries
    ]

    return Compose(built, step_ids=step_ids, verbose=pipeline_verbose)

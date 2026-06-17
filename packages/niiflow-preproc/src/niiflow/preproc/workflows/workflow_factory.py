"""Workflow discovery and instantiation for config-driven preprocessing."""

from __future__ import annotations

__all__ = [
    "create_workflow",
    "discover_workflow_classes",
]

import importlib
import inspect
from typing import Any

from .workflow import PlanningWorkflow, ProcessingWorkflow

_WORKFLOW_MODULES: tuple[str, ...] = ("niiflow.preproc.workflows.dynamic_workflow",)


def discover_workflow_classes() -> dict[str, type[ProcessingWorkflow]]:
    """Return concrete :class:`ProcessingWorkflow` types keyed by class name."""
    registry: dict[str, type[ProcessingWorkflow]] = {}
    for module_name in _WORKFLOW_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj in (ProcessingWorkflow, PlanningWorkflow):
                continue
            if inspect.isabstract(obj):
                continue
            if not issubclass(obj, ProcessingWorkflow):
                continue
            registry[name] = obj
    return registry


def _resolve_workflow_class(
    name: str, registry: dict[str, type[ProcessingWorkflow]]
) -> type[ProcessingWorkflow]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"`workflow_name` must be a non-empty string, got {name!r}")
    name = name.strip()
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown workflow {name!r}; expected one of {sorted(registry)}"
        ) from exc


def create_workflow(
    workflow_name: str,
    workflow_kwargs: dict[str, Any] | None = None,
    *,
    registry: dict[str, type[ProcessingWorkflow]] | None = None,
) -> ProcessingWorkflow:
    """Instantiate a :class:`ProcessingWorkflow` from a registered class name.

    ``workflow_name`` must match a concrete workflow listed by
    :func:`discover_workflow_classes`. ``workflow_kwargs`` is forwarded to that class's
    constructor.
    """
    if workflow_kwargs is not None and not isinstance(workflow_kwargs, dict):
        raise TypeError(
            f"`workflow_kwargs` must be a dictionary or None, got "
            f"{type(workflow_kwargs).__name__}"
        )

    workflow_cls = _resolve_workflow_class(
        workflow_name, registry or discover_workflow_classes()
    )
    kwargs = {} if workflow_kwargs is None else dict(workflow_kwargs)

    try:
        return workflow_cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate workflow {workflow_name!r}: {exc}"
        ) from exc

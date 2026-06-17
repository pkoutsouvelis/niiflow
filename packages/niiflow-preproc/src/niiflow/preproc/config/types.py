"""Typed configuration structures for preprocessing."""

from __future__ import annotations

__all__ = [
    "ArtifactsConfig",
    "PreprocConfig",
    "WorkflowConfig",
]

from typing import Any, NotRequired, TypedDict


class WorkflowConfig(TypedDict):
    """Configuration used to instantiate a workflow.

    Attributes:
        name: Registered workflow class name.
        kwargs: Keyword arguments forwarded to the workflow constructor.
    """

    name: str
    kwargs: NotRequired[dict[str, Any]]


class ArtifactsConfig(TypedDict, total=False):
    """Optional artifact paths used by CLI command actions."""

    plan_path: str


class PreprocConfig(TypedDict):
    """Top-level preprocessing config consumed by CLI commands.

    Attributes:
        workflow: Workflow instantiation configuration.
        run_inputs: Inputs passed to workflow planning/execution methods.
        artifacts: Optional artifact paths, such as a default plan path.
    """

    workflow: WorkflowConfig
    run_inputs: NotRequired[Any]
    artifacts: NotRequired[ArtifactsConfig]

"""Config-driven preprocessing CLI command actions.

This module contains the callable implementations behind CLI commands. Commands accept
already-loaded preprocessing configs plus CLI options, instantiate workflows, and invoke
the appropriate workflow methods.

The workflow classes remain the reusable Python API; this module defines only the
default command-line behavior.
"""

from __future__ import annotations

__all__ = [
    "execute",
    "plan",
    "execute_plan",
]

from pathlib import Path
from typing import Any, Literal, cast, overload

from niiflow.preproc.config.types import PreprocConfig, WorkflowConfig
from niiflow.preproc.utils.file import resolve_path
from niiflow.preproc.workflows.plan import RunPlan
from niiflow.preproc.workflows.workflow import PlanningWorkflow, ProcessingWorkflow
from niiflow.preproc.workflows.workflow_factory import create_workflow


def execute(
    config: PreprocConfig,
    *,
    save_plan: Path | str | None = None,
    dry_run: bool = False,
) -> RunPlan | None:
    """Execute a configured workflow.

    For planning workflows, this builds a run plan from ``config["run_inputs"]``. The
    generated plan is saved when either ``save_plan`` is provided or
    ``artifacts.plan_path`` exists in the config. If neither is available, the workflow
    executes without saving the plan.

    When ``dry_run`` is ``True``, the generated or loaded plan is printed and the
    workflow is not executed.
    """
    workflow = _create_configured_workflow(config)
    run_inputs = _require_key(cast(dict[str, Any], config), "run_inputs")

    if isinstance(workflow, PlanningWorkflow):
        run_plan = workflow.plan(run_inputs)
        if dry_run:
            print(run_plan.view())
            return run_plan

        output = _resolve_plan_output(config, save_plan, strict=False)
        if output is not None:
            run_plan.save(output)

        workflow.run_plan(run_plan)
        return run_plan

    if dry_run:
        raise TypeError(
            f"`--dry-run` is not supported for workflow {type(workflow).__name__}"
        )

    workflow.run(run_inputs)
    return None


def plan(
    config: PreprocConfig,
    *,
    output: Path | str | None = None,
    dry_run: bool = False,
) -> RunPlan:
    """Build and save a run plan from a configured planning workflow.

    Args:
        config: Already-loaded preprocessing config.
        output: Path where the run plan should be saved. If omitted, this falls
            back to ``config["artifacts"]["plan_path"]``.
        dry_run: When ``True``, print the plan and do not save it.

    Returns:
        The generated run plan.

    Raises:
        TypeError: If the configured workflow does not support planning.
        ValueError: If no output path is provided or configured.
    """
    workflow = _create_configured_workflow(config)

    if not isinstance(workflow, PlanningWorkflow):
        raise TypeError(f"Workflow {type(workflow).__name__} does not support planning")

    run_inputs = _require_key(cast(dict[str, Any], config), "run_inputs")
    run_plan = workflow.plan(run_inputs)
    if dry_run:
        print(run_plan.view())
        return run_plan

    run_plan.save(_resolve_plan_output(config, output, strict=True))
    return run_plan


def execute_plan(
    config: PreprocConfig,
    *,
    plan_path: Path | str,
    dry_run: bool = False,
) -> RunPlan:
    """Execute a saved run plan with the configured workflow.

    Args:
        config: Already-loaded preprocessing config.
        plan_path: Path to a saved run plan.
        dry_run: When ``True``, print the loaded plan and do not execute it.

    Returns:
        The loaded run plan.

    Raises:
        TypeError: If the configured workflow does not support run plans.
    """
    workflow = _create_configured_workflow(config)

    if not isinstance(workflow, PlanningWorkflow):
        raise TypeError(
            f"Workflow {type(workflow).__name__} does not support run plans"
        )

    run_plan = RunPlan.load(plan_path)
    if dry_run:
        print(run_plan.view())
        return run_plan

    workflow.run_plan(run_plan)
    return run_plan


def _create_configured_workflow(config: PreprocConfig) -> ProcessingWorkflow:
    """Instantiate the workflow described by ``config['workflow']``."""
    workflow_cfg = _get_workflow_config(config)
    return create_workflow(
        workflow_cfg["name"],
        workflow_cfg.get("kwargs", {}),
    )


def _get_workflow_config(config: PreprocConfig) -> WorkflowConfig:
    """Return normalized workflow configuration."""
    value = _require_mapping(cast(dict[str, Any], config), "workflow")

    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`workflow.name` must be a non-empty string")

    kwargs = value.get("kwargs", {})
    if kwargs is None:
        kwargs = {}

    if not isinstance(kwargs, dict):
        raise TypeError("`workflow.kwargs` must be a mapping")

    return {
        "name": name.strip(),
        "kwargs": kwargs,
    }


def _require_key(config: dict[str, Any], key: str) -> Any:
    """Return a required top-level config value."""
    if key not in config:
        raise ValueError(f"Missing required config section `{key}`")
    return config[key]


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required top-level mapping section."""
    value = _require_key(config, key)

    if not isinstance(value, dict):
        raise TypeError(f"`{key}` must be a mapping")

    return value


@overload
def _resolve_plan_output(
    config: PreprocConfig,
    output: Path | str | None,
    *,
    strict: Literal[True],
) -> Path: ...


@overload
def _resolve_plan_output(
    config: PreprocConfig,
    output: Path | str | None,
    *,
    strict: Literal[False],
) -> Path | None: ...


def _resolve_plan_output(
    config: PreprocConfig,
    output: Path | str | None,
    *,
    strict: bool,
) -> Path | None:
    """Resolve a run-plan output path.

    Explicit ``output`` takes precedence. If omitted, this falls back to
    ``artifacts.plan_path``. In strict mode, missing output raises. In non-strict mode,
    missing output returns ``None``.
    """
    if output is not None:
        return resolve_path(output)

    artifacts = config.get("artifacts", {})
    if isinstance(artifacts, dict):
        plan_path = artifacts.get("plan_path")
        if plan_path is not None:
            return resolve_path(plan_path)

    if strict:
        raise ValueError(
            "A run-plan output path is required. Provide one explicitly "
            "or set `artifacts.plan_path` in the config."
        )

    return None

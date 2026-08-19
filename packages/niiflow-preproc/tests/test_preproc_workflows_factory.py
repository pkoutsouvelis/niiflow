"""Tests for :mod:`niiflow.preproc.workflows.workflow_factory`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.staging import FileStager
from niiflow.preproc.workflows import DynamicProcessingWorkflow
from niiflow.preproc.workflows.workflow_factory import (
    create_workflow,
    discover_workflow_classes,
)


def test_discover_workflow_classes_contains_dynamic() -> None:
    registry = discover_workflow_classes()
    assert "DynamicProcessingWorkflow" in registry


def test_create_workflow_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown workflow"):
        create_workflow("MissingWorkflow")


def test_create_workflow_forwards_kwargs(tmp_path: Path) -> None:
    pointers = {"steps.echo.params.image": "input"}
    pipeline_params = {"steps": [], "output_path": str(tmp_path / "out.txt")}
    wf = create_workflow(
        "DynamicProcessingWorkflow",
        {
            "num_workers": 3,
            "timeout": 12.5,
            "logs_root": tmp_path / "logs",
            "staging_params": {
                "stager_name": "FileStager",
                "params": {"pointers": pointers},
            },
            "pipeline_params": pipeline_params,
        },
    )
    assert isinstance(wf, DynamicProcessingWorkflow)

    # Non-default values prove the kwargs reached the constructor rather than
    # falling back to defaults.
    assert wf.num_workers == 3
    assert wf.timeout == 12.5

    assert len(wf._stagers) == 3
    stager = wf._stagers[1]
    assert isinstance(stager, FileStager)
    assert stager.pointers == pointers

    assert wf._entry_params == pipeline_params

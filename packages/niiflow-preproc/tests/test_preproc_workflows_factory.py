"""Tests for :mod:`niiflow.preproc.workflows.workflow_factory`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.workflows import DynamicPreprocessingWorkflow
from niiflow.preproc.workflows.workflow_factory import (
    create_workflow,
    discover_workflow_classes,
)


def test_discover_workflow_classes_contains_dynamic() -> None:
    registry = discover_workflow_classes()
    assert "DynamicPreprocessingWorkflow" in registry


def test_create_workflow_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown workflow"):
        create_workflow("MissingWorkflow")


def test_create_workflow_forwards_kwargs(tmp_path: Path) -> None:
    wf = create_workflow(
        "DynamicPreprocessingWorkflow",
        {
            "num_workers": 1,
            "staging_params": {"stager_name": "FileStager", "params": {"pointers": {}}},
            "pipeline_params": {"steps": [], "output_path": str(tmp_path / "out.txt")},
        },
    )
    assert isinstance(wf, DynamicPreprocessingWorkflow)

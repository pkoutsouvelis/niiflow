"""Shared pytest fixtures for the ``niiflow-preproc`` test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.pipelines.pipeline_stages import PipelineStage
from stage_helpers import (
    SHIPPED_STAGE_CLASSES,
    StageFactory,
    make_stage,
    patch_shipped_stage_for_run,
)


@pytest.fixture(params=SHIPPED_STAGE_CLASSES, ids=lambda cls: cls.__name__)
def shipped_stage_cls(request: pytest.FixtureRequest) -> StageFactory:
    """Each shipped concrete stage class, one per parametrized run."""
    return request.param


@pytest.fixture
def shipped_stage(
    shipped_stage_cls: StageFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> PipelineStage:
    """A shipped stage instance with ``forward`` / ``save_output`` stubbed out."""
    stage = make_stage(shipped_stage_cls, tmp_path)
    patch_shipped_stage_for_run(stage, monkeypatch)
    return stage

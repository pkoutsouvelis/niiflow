"""Capability mixins for file-driven processing workflows."""

from __future__ import annotations

__all__ = [
    "FileDiscoveryMixin",
    "InputData",
    "StagingMixin",
]

from copy import deepcopy
from typing import Any, TypeAlias
from pathlib import Path
from collections.abc import Sequence

from niiflow.preproc.utils.file import resolve_path
from niiflow.preproc.data import get_data_explorer
from niiflow.preproc.staging import StagedEntry, Stager, make_entries, create_stager

InputData: TypeAlias = Path | str | Sequence[Path | str]


class FileDiscoveryMixin:
    """Discover *active files* that anchor each processing entry.

    An **active file** is the canonical path for one unit of work — the file
    that defines *what* is being processed. For example, a subject's T1w
    structural MRI may serve as the active file for an entry whose pipeline
    also locates FLAIR, a segmentation mask, and derivative outputs relative
    to that anchor.

    Discovery accepts explicit paths or an explorer configuration (root,
    pattern, optional filters) and returns one active path per entry.

    Intended for use as a mixin on :class:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow`
    subclasses, which provide :meth:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow.log`.
    """

    @property
    def explorer_params(self) -> dict[str, Any]:
        return deepcopy(self._explorer_params)

    def configure_explorer(self, explorer_params: dict[str, Any] | None) -> None:
        """Instantiate a data explorer with user-provided settings."""
        if explorer_params is None:
            self._explorer = None
            self._explorer_params = {}
            return

        if not isinstance(explorer_params, dict):
            raise TypeError(
                f"`explorer_params` must be a dictionary or None, got "
                f"{type(explorer_params).__name__}"
            )

        allowed = {"pattern", "filters"}
        unknown = set(explorer_params) - allowed
        if unknown:
            raise ValueError(
                f"`explorer_params` contains unsupported keys {sorted(unknown)}; "
                f"expected only {sorted(allowed)}"
            )

        pattern = explorer_params.get("pattern")
        if pattern is None:
            raise ValueError("`explorer_params.pattern` is required")

        try:
            explorer = get_data_explorer(
                pattern=pattern,
                filter_kwargs=explorer_params.get("filters"),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to bind arguments to data explorer; see "
                "``nifti_finder``'s documentation of "
                "``AllPurposeFileExplorer`` for more details."
            ) from exc

        self._explorer = explorer
        self._explorer_params = deepcopy(explorer_params)

    def configure_files(self, files: InputData) -> None:
        """Store and validate run inputs for subsequent active-file discovery.

        Explicit file paths are used directly. Directory paths require a configured data
        explorer and are treated as search roots.
        """
        if isinstance(files, (Path, str)):
            items = [files]
        elif isinstance(files, Sequence):
            items = list(files)
        else:
            raise TypeError(
                f"`files` must be a Path, str, or sequence of either; got "
                f"{type(files).__name__}"
            )

        if not items:
            raise ValueError("`files` must be non-empty")

        resolved: list[Path] = []
        for item in items:
            if not isinstance(item, (Path, str)):
                raise TypeError(
                    "Each item in `files` must be a Path or str, got "
                    f"{type(item).__name__}"
                )

            path = resolve_path(item)

            if not path.exists():
                raise FileNotFoundError(path)

            resolved.append(path)

        if self._explorer is None and any(path.is_dir() for path in resolved):
            raise ValueError(
                "Directory run inputs require a configured data explorer. "
                "Pass explicit files or configure `explorer_params`."
            )

        self._files = resolved

    def discover_active_files(self) -> list[Path]:
        """Return absolute active file paths from the configured run inputs."""
        log = getattr(self, "log", None)
        if not callable(log):
            raise AttributeError(
                f"{type(self).__name__} must inherit from ProcessingWorkflow; "
                "FileDiscoveryMixin relies on its log() method."
            )

        found: list[Path] = []

        if self._explorer is None:
            found = list(self._files)
        else:
            for path in self._files:
                if path.is_file():
                    found.append(path)
                    continue

                log(f"Extracting files using data explorer for directory {path}...")
                result = self._explorer.list(path, sort=True, unique=True)
                log(f"Found {len(result)} unique files under {path}.")
                found.extend(result)

        unique = list(dict.fromkeys(found))

        if not unique:
            raise RuntimeError(
                "No active files were found. Check the run inputs and explorer "
                "configuration."
            )

        return unique


class StagingMixin:
    """Build :class:`~niiflow.preproc.staging.StagedEntry` objects and run stagers.

    Processing revolves around **active files** as stable anchors. Each active
    file becomes one :class:`~niiflow.preproc.staging.StagedEntry`; the anchor
    path is never rewritten during staging.

    ``entry_params`` are attached to each entry before stagers run. They may be a
    single dictionary or a sequence aligned one-to-one with active files.
    Stagers transform ``entry_params`` in place (for example resolving file
    pointers). The concrete workflow decides what goes into ``entry_params``.

    ``staging_params`` optionally configure a stager chain via
    :func:`~niiflow.preproc.staging.create_stager`.

    Intended for use as a mixin on :class:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow`
    subclasses, which provide :meth:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow.log`.
    """

    @property
    def entry_params(self) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(self._entry_params, dict):
            return deepcopy(self._entry_params)
        return [deepcopy(item) for item in self._entry_params]

    @property
    def staging_params(self) -> list[dict[str, Any]]:
        return deepcopy(self._staging_params)

    def configure_staging(
        self,
        staging_params: dict[str, Any] | Sequence[dict[str, Any]] | None,
    ) -> None:
        if staging_params is None:
            self._stagers = []
            self._staging_params = []
            return

        if not isinstance(staging_params, (list, dict)):
            raise ValueError(
                f"`staging_params` must be a dictionary, sequence of dictionaries, "
                f"or None; got {type(staging_params).__name__}"
            )
        specs = (
            [staging_params]
            if isinstance(staging_params, dict)
            else list(staging_params)
        )
        if not all(isinstance(item, dict) for item in specs):
            raise ValueError("Each `staging_params` item must be a dictionary")

        stagers: list[Stager] = []
        staging_specs: list[dict[str, Any]] = []
        for spec in specs:
            stager_name = spec.get("stager_name", "FileStager")
            stager_params = spec.get("params")
            stagers.append(create_stager(stager_name, stager_kwargs=stager_params))
            staging_specs.append({"stager_name": stager_name, "params": stager_params})

        self._stagers = stagers
        self._staging_params = staging_specs

    def configure_entry_params(
        self,
        entry_params: dict[str, Any] | Sequence[dict[str, Any]],
    ) -> None:
        if not isinstance(entry_params, (dict, list, tuple)):
            raise TypeError(
                f"`entry_params` must be a dictionary or sequence of dictionaries; "
                f"got {type(entry_params).__name__}"
            )

        if isinstance(entry_params, (list, tuple)):
            if not all(isinstance(item, dict) for item in entry_params):
                raise TypeError("Each `entry_params` item must be a dictionary")
            self._entry_params = [deepcopy(item) for item in entry_params]
        else:
            self._entry_params = deepcopy(entry_params)

    def stage_active_files(self, active_files: Sequence[Path]) -> list[StagedEntry]:
        log = getattr(self, "log", None)
        if not callable(log):
            raise AttributeError(
                f"{type(self).__name__} must inherit from ProcessingWorkflow; "
                "StagingMixin relies on its log() method."
            )
        log(f"Staging {len(active_files)} entries...")
        entries = make_entries(active_files, self._entry_params)
        for stager in self._stagers:
            log(f"Staging with {stager.__class__.__name__}...")
            entries = stager.stage(entries)
        log(f"Staged {len(entries)} entries.")
        return entries

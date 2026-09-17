"""Capability mixins for file-driven processing workflows."""

from __future__ import annotations

__all__ = [
    "InputData",
    "SupportsInputDiscovery",
    "SupportsStaging",
]

from copy import deepcopy
from typing import Any
from pathlib import Path
from collections.abc import Sequence

from niiflow.preproc.utils.file import get_ext, resolve_path, write_txt
from niiflow.preproc.data import (
    NiftiFinderConfig,
    get_data_explorer,
    read_paths_from_file,
)
from niiflow.preproc.staging import (
    Stager,
    make_entries,
    create_stager,
    add_reference_staging_bookends,
)

from .plan import RunPlan
from .types import InputData


class SupportsInputDiscovery:
    """Collect active files that anchor each processing entry.

    An active file is the canonical path for one unit of work (for example a
    subject's T1w) around which companions and outputs are resolved.

    :meth:`collect_active_files` is the public entry point. It accepts
    :data:`~niiflow.preproc.workflows.types.InputData` — an explicit path, a
    ``search`` / ``from_file`` mapping, or a sequence of those — and may restrict
    modes with ``allow_explicit`` / ``allow_from_file`` / ``allow_search``.
    Paths are returned in discovery order with no deduplication.

    Existence rules are per source: explicit paths must exist as files; ``search``
    hits exist by construction (roots must exist); ``from_file`` existence follows
    optional ``strict`` (default ``True``) on
    :func:`~niiflow.preproc.data.read_paths_from_file`.

    Requires :meth:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow.log`
    from the concrete workflow.
    """

    def collect_active_files(
        self,
        inputs: InputData,
        *,
        save_to: Path | str | None = None,
        allow_explicit: bool = True,
        allow_from_file: bool = True,
        allow_search: bool = True,
    ) -> list[Path]:
        """Return absolute active file paths from ``inputs``.

        Args:
            inputs: Run inputs — a path, ``SearchInput``, ``FromFileInput``, or a
                sequence of those.
            save_to: Optional ``.txt`` path where the collected active files are
                written (one path per line). ``None`` skips saving.
            allow_explicit: Accept bare ``Path`` / ``str`` inputs.
            allow_from_file: Accept ``mode: from_file`` mappings.
            allow_search: Accept ``mode: search`` mappings.

        Returns:
            Absolute file paths in discovery order.

        Raises:
            FileNotFoundError: If an explicit path or search root does not exist.
            ValueError: If an explicit path is not a file, a mapping is invalid,
                a disallowed input mode is present, collection yields no files,
                or ``save_to`` is not a ``.txt`` path.
            TypeError: If ``inputs`` is the wrong type or a search mapping is
                missing ``explorer_params``.
            RuntimeError: If no data explorer can be built for a search input.
        """
        log = getattr(self, "log", None)
        if not callable(log):
            raise AttributeError(
                f"SupportsInputDiscovery requires a callable log() method, but "
                f"{type(self).__name__} has no attribute 'log'. "
                "Combine with ProcessingWorkflow (or otherwise define log)."
            )

        for name, value in (
            ("allow_explicit", allow_explicit),
            ("allow_from_file", allow_from_file),
            ("allow_search", allow_search),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"`{name}` must be a boolean, got {type(value).__name__}"
                )
        if not (allow_explicit or allow_from_file or allow_search):
            raise ValueError(
                "At least one of `allow_explicit`, `allow_from_file`, or "
                "`allow_search` must be True"
            )

        if isinstance(inputs, (str, Path, dict)):
            items: list[Any] = [inputs]
        elif isinstance(inputs, Sequence):
            items = list(inputs)
        else:
            raise TypeError(
                f"`inputs` must be a path, mapping, or sequence of either; "
                f"got {type(inputs).__name__}"
            )
        if not items:
            raise ValueError("`inputs` must be non-empty")

        log("Collecting active files...")
        found: list[Path] = []
        for item in items:
            if isinstance(item, (str, Path)):
                if not allow_explicit:
                    raise ValueError(
                        "Explicit path inputs are not allowed (`allow_explicit=False`)"
                    )
                found.append(self._collect_explicit_active_file(item))
                continue

            if isinstance(item, dict):
                mode = item.get("mode")
                if mode == "search":
                    if not allow_search:
                        raise ValueError(
                            "search inputs are not allowed (`allow_search=False`)"
                        )
                    found.extend(
                        self._search_active_files(
                            item.get("roots"),  # type: ignore[arg-type]
                            item.get("explorer_params"),  # type: ignore[arg-type]
                        )
                    )
                    continue
                if mode == "from_file":
                    if not allow_from_file:
                        raise ValueError(
                            "from_file inputs are not allowed (`allow_from_file=False`)"
                        )
                    path = item.get("path")
                    if path is None or not isinstance(path, (str, Path)):
                        raise ValueError("`from_file` input requires `path`")
                    found.extend(
                        self._collect_active_files_from_file(
                            path,
                            strict=item.get("strict", True),
                            skip_resolve_filepaths=item.get(
                                "skip_resolve_filepaths", False
                            ),
                        )
                    )
                    continue
                raise ValueError(
                    f"Unknown/missing run-input mode {mode!r}; expected 'search' or 'from_file'"
                )

            raise TypeError(
                f"Unsupported run-input entry type {type(item).__name__}; "
                "expected a path or mapping"
            )

        if not found:
            raise RuntimeError(
                "No active files were found. Check the run inputs and explorer "
                "configuration."
            )
        log(f"Found {len(found)} active file path(s).")

        if save_to is not None:
            out = resolve_path(save_to)
            if get_ext(out) != ".txt":
                raise ValueError(f"`save_to` must be a .txt path, got {out}")
            write_txt("\n".join(str(path) for path in found), out)
            log(f"Wrote {len(found)} active file path(s) to {out}.")

        return found

    def _collect_explicit_active_file(self, path: Path | str) -> Path:
        """Resolve one explicit active file path and require that it exists."""
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        if not resolved.is_file():
            raise ValueError(
                f"Explicit run input must be a file, got directory {resolved}"
            )
        return resolved

    def _collect_active_files_from_file(
        self,
        path: Path | str,
        *,
        strict: bool = True,
        skip_resolve_filepaths: bool = False,
    ) -> list[Path]:
        """Read active file paths from a ``.txt`` listing (one path per line)."""
        if not isinstance(path, (str, Path)):
            raise ValueError("`from_file` input requires `path`")
        list_path = resolve_path(path)
        self.log(f"Reading file paths from: {list_path}")  # type: ignore[attr-defined]
        return read_paths_from_file(
            list_path,
            strict=strict,
            skip_resolve_filepaths=skip_resolve_filepaths,
        )

    def _search_active_files(
        self,
        roots: Path | str | Sequence[Path | str],
        explorer_params: NiftiFinderConfig,
    ) -> list[Path]:
        """Search for active files under one or more roots with a FileFinder config."""
        if roots is None:
            raise ValueError("`search` input requires `roots`")
        if not isinstance(explorer_params, dict):
            raise TypeError("`search` input requires mapping `explorer_params`")
        if isinstance(roots, (str, Path)):
            root_list: list[Any] = [roots]
        elif isinstance(roots, Sequence):
            root_list = list(roots)
        else:
            raise TypeError(
                f"`roots` must be a path or sequence of paths, got {type(roots).__name__}"
            )
        if not root_list:
            raise ValueError("`roots` must be non-empty")

        try:
            explorer = get_data_explorer(
                patterns=explorer_params.get("patterns", "*.nii*"),
                levels=explorer_params.get("levels"),
                filters=explorer_params.get("filters"),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to bind arguments to data explorer; see "
                "``nifti_finder``'s documentation of ``FileFinder``."
            ) from exc

        found: list[Path] = []
        for root in root_list:
            root_path = resolve_path(root)
            if not root_path.exists():
                raise FileNotFoundError(root_path)
            if not root_path.is_dir():
                raise ValueError(f"Search root must be a directory, got {root_path}")
            self.log(f"Extracting files using data explorer for {root_path}...")  # type: ignore[attr-defined]
            result = explorer.list(root_path, sort=True, unique=True)
            self.log(f"Found {len(result)} unique files under {root_path}.")  # type: ignore[attr-defined]
            found.extend(result)
        return found


class SupportsStaging:
    """Build :class:`~niiflow.preproc.staging.StagedEntry` objects and run stagers.

    Processing revolves around active files as stable anchors. Each active
    file becomes one :class:`~niiflow.preproc.staging.StagedEntry`; the anchor
    path is never rewritten during staging.

    ``entry_params`` are attached to each entry before stagers run. They may be a
    single dictionary, a sequence aligned one-to-one with active files, or
    ``None`` (treated as an empty parameter mapping per entry). Stagers transform
    ``entry_params`` in place (for example resolving file pointers). The concrete
    workflow decides what goes into ``entry_params``.

    ``staging_params`` optionally configure a user stager chain via
    :func:`~niiflow.preproc.staging.create_stager`. The workflow always bookends
    that chain with
    :class:`~niiflow.preproc.staging.ResolveActiveReferences` and
    :class:`~niiflow.preproc.staging.ResolveParamReferences` (even when
    ``staging_params`` is ``None``), so ``{active.*}`` / leftover ``{params.*}``
    in ``entry_params`` are expanded consistently. Active existence is not
    enforced here; place
    :class:`~niiflow.preproc.staging.EnsureActivesExist` in the chain when needed.

    ``staging_workers`` is the thread count used by :meth:`stage_active_files`
    (default ``1``, serial).

    Intended for use as a mixin on :class:`~niiflow.preproc.workflows.plannable_workflow.PlannableWorkflow`
    subclasses, which provide :meth:`~niiflow.preproc.workflows.plannable_workflow.PlannableWorkflow.log`.
    """

    def configure_staging(
        self,
        staging_params: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        entry_params: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        staging_workers: int = 1,
    ) -> None:
        """Configure staging-related parameters.

        To be used in the constructor of a
        :class:`~niiflow.preproc.workflows.plannable_workflow.PlannableWorkflow` subclass.

        Args:
            staging_params: Optional stager specs for
                :func:`~niiflow.preproc.staging.create_stager`.
            entry_params: Params attached to each entry before stagers run.
            staging_workers: Thread count for :meth:`stage_active_files`
                (default ``1``, serial).
        """
        if isinstance(staging_workers, bool) or not isinstance(staging_workers, int):
            raise ValueError(
                f"`staging_workers` must be an integer >= 1, got `{staging_workers}`"
            )
        if staging_workers < 1:
            raise ValueError(
                f"`staging_workers` must be at least 1, got `{staging_workers}`"
            )
        self._staging_workers = staging_workers

        if staging_params is None:
            user_stagers: list[Stager] = []
            self._staging_params: list[dict[str, Any]] = []
        else:
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

            user_stagers = []
            staging_specs: list[dict[str, Any]] = []
            for spec in specs:
                stager_name = spec.get("stager_name", "FileStager")
                stager_params = spec.get("params")
                user_stagers.append(
                    create_stager(stager_name, stager_kwargs=stager_params)
                )
                staging_specs.append(
                    {"stager_name": stager_name, "params": stager_params}
                )

            self._staging_params = staging_specs

        self._stagers = add_reference_staging_bookends(user_stagers)

        if entry_params is None:
            self._entry_params: dict[str, Any] | list[dict[str, Any]] = {}
        elif isinstance(entry_params, dict):
            self._entry_params = deepcopy(entry_params)
        elif isinstance(entry_params, (list, tuple)):
            if not all(isinstance(item, dict) for item in entry_params):
                raise TypeError("Each `entry_params` item must be a dictionary")
            self._entry_params = [deepcopy(item) for item in entry_params]
        else:
            raise TypeError(
                f"`entry_params` must be a dictionary, sequence of dictionaries, "
                f"or None; got {type(entry_params).__name__}"
            )

    def stage_active_files(
        self,
        active_files: Sequence[Path],
        *,
        save_to: Path | str | None = None,
    ) -> RunPlan:
        """Stage active files into a :class:`~niiflow.preproc.workflows.plan.RunPlan`.

        Args:
            active_files: Absolute active file paths to stage.
            save_to: Optional ``.duckdb`` path where the staged plan is written
                (``.json`` is deprecated until v0.5.0). ``None`` skips saving.

        Returns:
            A run plan containing one staged entry per active file.
        """
        log = getattr(self, "log", None)
        if not callable(log):
            raise AttributeError(
                f"SupportsStaging requires a callable log() method, but "
                f"{type(self).__name__} has no attribute 'log'. "
                "Combine with ProcessingWorkflow (or otherwise define log)."
            )
        log(f"Staging {len(active_files)} entries...")
        entries = make_entries(active_files, self._entry_params)
        num_workers = self._staging_workers
        for stager in self._stagers:
            if num_workers > 1:
                log(
                    f"Staging with {stager.__class__.__name__} "
                    f"({num_workers} threads)..."
                )
            else:
                log(f"Staging with {stager.__class__.__name__}...")
            entries = stager.stage(entries, num_workers=num_workers)
        log(f"Staged {len(entries)} entries.")

        plan = RunPlan(entries=tuple(entries))
        if save_to is not None:
            out = plan.save(save_to)
            log(f"Wrote run plan ({len(entries)} entries) to {out}.")
        return plan

"""File stager implementation."""

from __future__ import annotations

__all__ = [
    "FileStager",
]

from copy import deepcopy
from pathlib import Path
from typing import Any

from niiflow.preproc.data.explorer_factory import get_data_explorer
from niiflow.preproc.utils.misc import (
    get_by_dotted_path,
    set_by_dotted_path,
)
from niiflow.preproc.utils.file import resolve_path
from .dynamic import resolve_dynamic_refs
from .search import (
    match_parent,
    mirror_root,
    parent_up,
    explorer_cache_key,
    resolve_search_result,
)
from .stager import (
    StageContext,
    StagedEntry,
    Stager,
    FileStagingError,
)
from .types import (
    InputSpec,
    OutputSpec,
    PointerKind,
    RootSpec,
)
from .validation import (
    ensure_file,
    ensure_directory,
    ensure_no_overwrite,
    check_allowed_keys,
    require_keys,
    require_mapping,
    validate_pointers,
)


class FileStager(Stager):
    """Resolve input/output file parameters around stable active files.

    A :class:`FileStager` walks declared *pointers* (dotted paths into each
    entry's ``params`` dict) and materialises file paths before pipeline
    execution. The entry's :attr:`~niiflow.preproc.staging.stager.StagedEntry.active`
    path is the stable run anchor; it is never rewritten by staging.

    Construction accepts a ``pointers`` mapping from dotted parameter paths to
    either ``"input"`` or ``"output"``. Input pointers are resolved first so
    output templates can reference them via ``{params.<dotted-path>}`` dynamic
    references. String specs may also use ``{active.<attr>}`` (``path``, ``name``,
    ``stem``, ``parent``).

    ``pointers`` may be omitted (or ``None``) to stage with dynamic references
    only: every ``{active.*}`` and ``{params.*}`` reference in ``params`` is still
    expanded, but no parameter is treated as a file to resolve or materialise.

    Args:
        pointers:
            Mapping from dotted parameter paths to either ``"input"`` or ``"output"``.
            ``None`` and ``{}`` both mean "no pointers"; dynamic references are
            still resolved.

        ensure_inputs_exist:
            When ``True``, direct input paths must exist and be files. Search-mode
            inputs always require their roots to exist, because otherwise search
            cannot run.

        allow_overwrite:
            When ``False``, resolved output paths must not already exist.
            Defaults to ``True``.

        allow_failed_entries:
            When ``False``, the first staging error aborts :meth:`stage`. When
            ``True``, the error is recorded on the returned entry and staging
            continues with the remaining entries.
    """

    def __init__(
        self,
        pointers: dict[str, PointerKind] | None = None,
        *,
        ensure_inputs_exist: bool = True,
        allow_overwrite: bool = True,
        allow_failed_entries: bool = False,
    ) -> None:
        self.pointers = validate_pointers(pointers)
        self.ensure_inputs_exist = bool(ensure_inputs_exist)
        self.allow_overwrite = bool(allow_overwrite)
        self.allow_failed_entries = bool(allow_failed_entries)

        self._explorer_cache: dict[tuple[Any, ...], Any] = {}

    def stage_single(self, entry: StagedEntry, *, index: int = 0) -> StagedEntry:
        """Stage one entry while preserving its
        :attr:`~niiflow.preproc.staging.stager.StagedEntry.active` file."""
        if entry.errors:
            return entry

        out_params = deepcopy(entry.params)
        current_pointer: str | None = None
        current_kind: PointerKind | None = None

        try:
            active = ensure_file(entry.active, must_exist=True)
            ctx = StageContext(active=active)

            # Pass 1: active refs can be resolved immediately; params refs are
            # preserved so output templates can refer to inputs resolved below.
            out_params = resolve_dynamic_refs(
                out_params,
                params=out_params,
                ctx=ctx,
                resolve_params=False,
            )

            # Inputs first: output specs may reference resolved input paths.
            for pointer, kind in self.pointers.items():
                if kind != "input":
                    continue
                current_pointer = pointer
                current_kind = kind
                spec = get_by_dotted_path(out_params, pointer)
                resolved = self.get_input_file(spec, ctx=ctx, pointer=pointer)
                set_by_dotted_path(out_params, pointer, resolved)

            # Outputs: expand params refs per pointer, then resolve paths.
            for pointer, kind in self.pointers.items():
                if kind != "output":
                    continue
                current_pointer = pointer
                current_kind = kind
                spec = get_by_dotted_path(out_params, pointer)
                spec = resolve_dynamic_refs(
                    spec,
                    params=out_params,
                    ctx=ctx,
                    resolve_params=True,
                )
                set_by_dotted_path(out_params, pointer, spec)
                resolved = self.get_output_path(spec, ctx=ctx, pointer=pointer)
                set_by_dotted_path(out_params, pointer, resolved)

            # Expand any remaining params refs outside declared output pointers.
            out_params = resolve_dynamic_refs(
                out_params,
                params=out_params,
                ctx=ctx,
                resolve_params=True,
            )

            return StagedEntry(
                active=entry.active, params=out_params, errors=entry.errors
            )

        except FileStagingError:
            raise
        except Exception as exc:
            if current_pointer is not None and current_kind is not None:
                message = (
                    f"Failed while resolving {current_kind} pointer "
                    f"{current_pointer!r}: {exc}"
                )
            else:
                message = f"Failed while staging active file {entry.active}: {exc}"
            raise FileStagingError(message) from exc

    def get_root(self, spec: RootSpec, *, ctx: StageContext, must_exist: bool) -> Path:
        """Resolve one root directory from a
        :data:`~niiflow.preproc.staging.types.RootSpec`."""
        if spec is None:
            return ensure_directory(ctx.active.parent, must_exist=must_exist)

        if isinstance(spec, (str, Path)):
            return ensure_directory(spec, must_exist=must_exist)

        if isinstance(spec, list):
            raise TypeError(
                "get_root() expects one root spec; use get_roots() for lists."
            )

        root_spec = require_mapping(spec, "Root spec")
        check_allowed_keys(
            root_spec, {"mode", "value", "mirror", "selection"}, "Root spec"
        )
        require_keys(root_spec, ["mode"], "Root spec")

        mode = root_spec["mode"]
        if mode == "active":
            if "value" in root_spec and root_spec["value"] is not None:
                raise ValueError(
                    f"Root spec with mode='active' must not define a value: {root_spec!r}"
                )
            result = ctx.active.parent

        elif mode == "path":
            require_keys(root_spec, ["value"], "Root spec with mode='path'")
            value = root_spec["value"]
            if not isinstance(value, (str, Path)):
                raise TypeError(
                    "Root spec value for mode='path' must be str or Path, "
                    f"got {type(value).__name__}."
                )
            result = resolve_path(value)

        elif mode == "parent_up":
            require_keys(root_spec, ["value"], "Root spec with mode='parent_up'")
            value = root_spec["value"]
            if not isinstance(value, int):
                raise TypeError(
                    "Root spec value for mode='parent_up' must be an integer, "
                    f"got {type(value).__name__}."
                )
            result = parent_up(ctx.active, value)

        elif mode == "parent_match":
            require_keys(root_spec, ["value"], "Root spec with mode='parent_match'")
            value = root_spec["value"]
            if not isinstance(value, str):
                raise TypeError(
                    "Root spec value for mode='parent_match' must be a string, "
                    f"got {type(value).__name__}."
                )
            selection = root_spec.get("selection", "most_local")
            result = match_parent(ctx.active.parent, value, selection=selection)

        else:
            raise ValueError(
                f"Unknown root mode {mode!r}. Expected one of: active, path, parent_up, parent_match."
            )

        mirror = root_spec.get("mirror")
        if mirror is not None:
            mirror_spec = require_mapping(mirror, "Mirror spec")
            check_allowed_keys(mirror_spec, {"source", "target"}, "Mirror spec")
            require_keys(mirror_spec, ["source", "target"], "Mirror spec")
            source = mirror_spec["source"]
            target = mirror_spec["target"]
            if not isinstance(source, (str, Path)):
                raise TypeError(
                    "Mirror spec 'source' must be str or Path, "
                    f"got {type(source).__name__}."
                )
            if not isinstance(target, (str, Path)):
                raise TypeError(
                    "Mirror spec 'target' must be str or Path, "
                    f"got {type(target).__name__}."
                )
            result = mirror_root(result, source=source, target=target)

        return ensure_directory(result, must_exist=must_exist)

    def get_roots(
        self, spec: RootSpec, *, ctx: StageContext, must_exist: bool
    ) -> list[Path]:
        """Resolve one or more root directories from a
        :data:`~niiflow.preproc.staging.types.RootSpec`."""
        if isinstance(spec, list):
            if not spec:
                raise ValueError("Root spec list cannot be empty.")
            return [
                self.get_root(item, ctx=ctx, must_exist=must_exist) for item in spec
            ]
        return [self.get_root(spec, ctx=ctx, must_exist=must_exist)]

    def get_input_file(
        self,
        spec: InputSpec,
        *,
        ctx: StageContext,
        pointer: str | None = None,
    ) -> Path | list[Path]:
        """Resolve an :data:`~niiflow.preproc.staging.types.InputSpec` to one or more
        files."""
        if spec is None:
            return ensure_file(ctx.active, must_exist=self.ensure_inputs_exist)

        if isinstance(spec, (str, Path)):
            return ensure_file(spec, must_exist=self.ensure_inputs_exist)

        input_spec = require_mapping(spec, "Input spec")
        check_allowed_keys(
            input_spec, {"root", "search", "resolve_results"}, "Input spec"
        )
        require_keys(input_spec, ["search"], "Input spec")

        roots = self.get_roots(input_spec.get("root"), ctx=ctx, must_exist=True)
        search_spec = require_mapping(input_spec["search"], "Input spec 'search'")
        explorer = self.get_explorer(search_spec)

        found: list[Path] = []
        for root in roots:
            found.extend(
                Path(path) for path in explorer.list(root, sort=True, unique=True)
            )

        policy = input_spec.get("resolve_results", "first")
        return resolve_search_result(found, policy=policy, pointer=pointer)

    def get_explorer(self, search_spec: dict[str, Any]) -> Any:
        """Return a data explorer, reusing a cached instance when the spec is
        hashable."""
        key = explorer_cache_key(search_spec)
        if key is None:
            return get_data_explorer(**search_spec)
        if key not in self._explorer_cache:
            self._explorer_cache[key] = get_data_explorer(**search_spec)
        return self._explorer_cache[key]

    def get_output_path(
        self,
        spec: OutputSpec,
        *,
        ctx: StageContext,
        pointer: str | None = None,
    ) -> Path:
        """Resolve an :data:`~niiflow.preproc.staging.types.OutputSpec` to a target
        path.

        Parent directories are **not** created here; downstream savers are responsible
        for creating missing parents before writing.
        """
        if spec is None:
            raise ValueError(f"Output spec for pointer {pointer!r} cannot be None.")

        if isinstance(spec, (str, Path)):
            return ensure_no_overwrite(spec, allow_overwrite=self.allow_overwrite)

        output_spec = require_mapping(spec, "Output spec")
        check_allowed_keys(output_spec, {"root", "name"}, "Output spec")
        require_keys(output_spec, ["name"], "Output spec")

        root_spec = output_spec.get("root")
        if isinstance(root_spec, list):
            raise TypeError(
                "Output root spec cannot be a list; outputs require one root."
            )

        root = self.get_root(root_spec, ctx=ctx, must_exist=False)
        name = output_spec["name"]
        if not isinstance(name, str):
            raise TypeError(
                f"Output spec 'name' must be a string, got {type(name).__name__}."
            )

        return ensure_no_overwrite(root / name, allow_overwrite=self.allow_overwrite)

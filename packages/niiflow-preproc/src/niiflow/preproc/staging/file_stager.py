"""File stager implementation."""

from __future__ import annotations

__all__ = [
    "FileStager",
]

from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

from niiflow.preproc.data.explorer_factory import get_data_explorer
from niiflow.preproc.utils.misc import (
    get_by_dotted_path,
    set_by_dotted_path,
)
from niiflow.preproc.utils.file import resolve_path
from .dynamic_referencing import resolve_dynamic_refs
from .search import (
    match_parent,
    mirror_root,
    parent_up,
    explorer_cache_key,
    resolve_search_result,
)
from .stager import (
    StagingContext,
    StagedEntry,
    Stager,
    FileStagingError,
)
from .types import (
    InputSpec,
    OutputSpec,
    PointerKind,
    RootSpec,
    SingleRootSpec,
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

    A :class:`FileStager` materialises file paths in each entry's pipeline
    parameters before pipeline execution. It does not modify the entry's
    :attr:`~niiflow.preproc.staging.stager.StagedEntry.active` path, which is
    treated as its stable anchor for the run. The active path need not exist;
    existence checks apply to declared input pointers (see
    ``ensure_inputs_exist``) and to search roots.

    Pointers are a construction-time map from dotted ``params`` paths to
    ``"input"`` or ``"output"``. Declared paths are resolved into concrete file
    locations (search / ``root`` + ``name`` / path join / existence checks).
    The mapping must be non-empty.

    Dynamic references (``{active.*}``, ``{params.*}``) are a separate concern.
    Prefer running :class:`~niiflow.preproc.staging.dynamic_referencing.ResolveActiveReferences`
    before this stager and
    :class:`~niiflow.preproc.staging.dynamic_referencing.ResolveParamReferences` after (see
    :func:`~niiflow.preproc.staging.dynamic_referencing.add_reference_staging_bookends`). Within this
    stager, ``{params.*}`` are expanded only on each input/output pointer spec
    immediately before that pointer is materialised.

    Bare relative ``str`` / ``Path`` input and output specs are anchored to
    ``active.parent`` (same default as omitting ``root``, or omitting ``mode``
    on a structured root spec); absolute paths are left unchanged.

    Args:
        pointers:
            Non-empty mapping from dotted parameter paths to either ``"input"``
            or ``"output"``.

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
        pointers: dict[str, PointerKind],
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
        self._explorer_lock = Lock()

    def stage_single(self, entry: StagedEntry, *, index: int = 0) -> StagedEntry:
        """Stage one entry while preserving its
        :attr:`~niiflow.preproc.staging.stager.StagedEntry.active` file."""
        if entry.errors:
            return entry

        out_params = deepcopy(entry.params)
        current_pointer: str | None = None
        current_kind: PointerKind | None = None

        try:
            active = ensure_file(entry.active, must_exist=False)
            ctx = StagingContext(active=active)

            # Inputs first: expand params refs on each input spec, then materialize.
            for pointer, kind in self.pointers.items():
                if kind != "input":
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
                resolved = self.get_input_file(spec, ctx=ctx, pointer=pointer)
                set_by_dotted_path(out_params, pointer, resolved)

            # Outputs: expand params refs per pointer, then materialize paths.
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

            return StagedEntry(
                active=entry.active,
                id=entry.id,
                params=out_params,
                errors=entry.errors,
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

    def get_root(
        self, spec: SingleRootSpec, *, ctx: StagingContext, must_exist: bool
    ) -> Path:
        """Resolve one root directory from a
        :data:`~niiflow.preproc.staging.types.SingleRootSpec`.

        ``None`` and a mapping without ``mode`` both start from
        ``ctx.active.parent``. ``mirror`` may still be applied on a mapping.
        """
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

        mode = root_spec.get("mode")
        if mode is None:
            if root_spec.get("value") is not None:
                raise ValueError(
                    "Root spec without mode must not define a value; "
                    "omit value to use the active parent, or set mode='path'."
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
                f"Unknown root mode {mode!r}. Expected one of: path, parent_up, parent_match."
            )

        mirror = root_spec.get("mirror")
        if mirror is not None:
            mirror_spec = require_mapping(mirror, "Mirror spec")
            check_allowed_keys(
                mirror_spec,
                {"source", "target", "allow_missing_source"},
                "Mirror spec",
            )
            require_keys(mirror_spec, ["source", "target"], "Mirror spec")
            source = mirror_spec["source"]
            target = mirror_spec["target"]
            if isinstance(source, list):
                if not source:
                    raise ValueError("Mirror spec 'source' list cannot be empty.")
                if not all(isinstance(item, (str, Path)) for item in source):
                    raise TypeError(
                        "Mirror spec 'source' list entries must be str or Path."
                    )
            elif not isinstance(source, (str, Path)):
                raise TypeError(
                    "Mirror spec 'source' must be str, Path, or list of those, "
                    f"got {type(source).__name__}."
                )
            if not isinstance(target, (str, Path)):
                raise TypeError(
                    "Mirror spec 'target' must be str or Path, "
                    f"got {type(target).__name__}."
                )
            allow_missing_source = bool(mirror_spec.get("allow_missing_source", False))
            result = mirror_root(
                result,
                source=source,
                target=target,
                allow_missing_source=allow_missing_source,
            )

        return ensure_directory(result, must_exist=must_exist)

    def get_roots(
        self, spec: RootSpec, *, ctx: StagingContext, must_exist: bool
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
        ctx: StagingContext,
        pointer: str | None = None,
    ) -> Path | list[Path]:
        """Resolve an :data:`~niiflow.preproc.staging.types.InputSpec` to one or more
        files.

        Mapping specs must define exactly one of ``search`` or ``name``. ``search``
        locates files under ``root`` with an explorer. ``name`` joins ``root / name``,
        the same shape as an output pointer.
        """
        if spec is None:
            return ensure_file(ctx.active, must_exist=self.ensure_inputs_exist)

        if isinstance(spec, (str, Path)):
            return ensure_file(
                self._anchor_to_active(spec, ctx=ctx),
                must_exist=self.ensure_inputs_exist,
            )

        input_spec = require_mapping(spec, "Input spec")
        check_allowed_keys(
            input_spec, {"root", "search", "name", "resolve_results"}, "Input spec"
        )
        has_search = "search" in input_spec
        has_name = "name" in input_spec
        if has_search == has_name:
            raise ValueError(
                "Input spec must define exactly one of 'search' or 'name'."
            )
        if has_name and "resolve_results" in input_spec:
            raise ValueError(
                "Input spec 'resolve_results' is only valid with 'search'."
            )

        if has_name:
            root_spec = input_spec.get("root")
            if isinstance(root_spec, list):
                raise TypeError(
                    "Input spec root cannot be a list when using 'name'; "
                    "name-based inputs require one root."
                )
            name = input_spec["name"]
            if not isinstance(name, str):
                raise TypeError(
                    f"Input spec 'name' must be a string, got {type(name).__name__}."
                )
            root = self.get_root(
                root_spec, ctx=ctx, must_exist=self.ensure_inputs_exist
            )
            return ensure_file(root / name, must_exist=self.ensure_inputs_exist)

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
        with self._explorer_lock:
            if key not in self._explorer_cache:
                self._explorer_cache[key] = get_data_explorer(**search_spec)
            return self._explorer_cache[key]

    @staticmethod
    def _anchor_to_active(path: str | Path, *, ctx: StagingContext) -> Path:
        """Anchor a relative path to ``ctx.active.parent``; leave absolutes as-is."""
        return ctx.active.parent / Path(path)

    def get_output_path(
        self,
        spec: OutputSpec,
        *,
        ctx: StagingContext,
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
            return ensure_no_overwrite(
                self._anchor_to_active(spec, ctx=ctx),
                allow_overwrite=self.allow_overwrite,
            )

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

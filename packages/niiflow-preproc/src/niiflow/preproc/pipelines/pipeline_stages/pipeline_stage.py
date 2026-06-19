"""Base class for pipeline stages."""

from __future__ import annotations

__all__ = [
    "RuntimeContext",
    "PipelineStage",
]

import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeAlias, get_args

from niiflow.preproc.utils.file import resolve_path
from niiflow.preproc.utils.misc import split_dotted_path

logger = logging.getLogger(__name__)


LogLevel = Literal["debug", "info", "warning", "error", "critical"]
_VALID_LOG_LEVELS: tuple[str, ...] = get_args(LogLevel)
_CTX_PREFIX = "ctx."


@dataclass
class RuntimeContext:
    """Runtime context threaded through pipeline stages.

    Attributes:
        run_id: Optional runner-defined string identifying the unit currently
            being processed in this pipeline run (e.g. a subject id, session id,
            or filepath). Set once per workflow entry by the caller; it is not
            the pipeline template name or a batch/job id.
        step_id: Optional id of the step currently being executed. Callers may
            set this before :meth:`PipelineStage.run`; when omitted, an id of the
            form ``step_0000``, ``step_0001``, ... is derived from
            ``len(steps_completed)``.
        metadata: Per-step metadata, keyed by ``step_id``. Each
            :meth:`PipelineStage.run` merges the dict from
            :meth:`PipelineStage.update_metadata` with a ``saved_paths`` entry
            (when outputs were written) listing output keys and absolute path(s)
            written per key (a single :class:`pathlib.Path` or, for multi-file
            outputs such as transform lists, ``tuple[Path, ...]``).
        artifacts: In-memory outputs from completed steps, keyed by ``step_id``.
            Each value is the mapping returned by :meth:`PipelineStage.forward`
            (e.g. ANTs images, arrays). Disk persistence does **not** replace
            these values; downstream stages consume them via ``ctx.`` refs.
        steps_completed: Step ids published so far, in execution order.

    Only :meth:`PipelineStage.run` (and the pipeline runner) write to
    ``artifacts``, ``metadata``, and ``steps_completed``; stage hooks must not
    mutate them directly.
    """

    run_id: str | None = None
    step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    steps_completed: list[str] = field(default_factory=list)


def _resolve_ctx_path(ctx: RuntimeContext, ref: str) -> Any:
    """Resolve a dotted ``ctx.<path>`` parameter reference against ``ctx``.

    Walks attributes on :class:`RuntimeContext`, keys on nested dicts, and list indices
    written as ``[idx]`` (e.g. ``ctx.metadata.subject``,
    ``ctx.artifacts.<step_id>.<output>``, ``ctx.steps_completed.[0]``). Artifact outputs
    are addressed by explicit pipeline step id, not by position.
    """
    if not isinstance(ref, str) or not ref.startswith(_CTX_PREFIX):
        raise ValueError(f"Context path must start with 'ctx.', got {ref!r}")
    try:
        parts = split_dotted_path(ref[len(_CTX_PREFIX) :])
    except ValueError as exc:
        raise ValueError(f"Context path {ref!r} has invalid segments") from exc
    if not parts:
        raise ValueError(f"Context path {ref!r} has empty segments")

    obj: Any = ctx
    walked: list[str] = ["ctx"]
    for part in parts:
        if isinstance(part, int):
            walked.append(f"[{part}]")
            try:
                obj = obj[part]
            except (IndexError, KeyError, TypeError) as exc:
                raise KeyError(
                    f"Failed to resolve context path '{'.'.join(walked)}': {exc}"
                ) from exc
        else:
            walked.append(part)
            try:
                obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
            except (AttributeError, KeyError, TypeError) as exc:
                raise KeyError(
                    f"Failed to resolve context path '{'.'.join(walked)}': {exc}"
                ) from exc
    return obj


@dataclass(frozen=True)
class ArtifactPathRecord:
    """Disk record for a saved logical artifact.

    path:
        Canonical artifact path. This is the path users/loaders should use
        to reload the artifact.

    payload:
        Additional files written as part of the artifact, if any.
    """

    path: Path
    payload: tuple[Path, ...] = ()


SavedPathRecord: TypeAlias = Path | ArtifactPathRecord


def _serialize_saved_paths(
    saved_paths: dict[str, SavedPathRecord],
) -> dict[str, str | dict[str, Any]]:
    """Serialize ``metadata[step_id]['saved_paths']``.

    Converts runtime path records into JSON/YAML-compatible values:

    - ``Path`` -> ``str``
    - ``ArtifactPathRecord`` -> ``{"path": str, "payload": list[str]}``
    """
    serialized: dict[str, str | dict[str, Any]] = {}

    for key, record in saved_paths.items():
        if isinstance(record, Path):
            serialized[key] = str(record)
            continue

        if isinstance(record, ArtifactPathRecord):
            serialized[key] = {
                "path": str(record.path),
                "payload": [str(path) for path in record.payload],
            }
            continue

        raise TypeError(
            f"`saved_paths['{key}']` must be a Path or ArtifactPathRecord, "
            f"got {type(record).__name__}"
        )

    return serialized


class PipelineStage(ABC):
    """Base class for atomic pipeline stages.

    A stage encapsulates a single preprocessing step: validated input parameters
    (via :attr:`REQUIRED_PARAMS` and optionally :meth:`check_params`), lazy
    materialisation of each parameter (via :meth:`load_param`), a :meth:`forward`
    pass producing named outputs, optional persistence of those outputs, and
    optional per-step metadata (via :meth:`update_metadata`).

    Construction accepts ``params`` and ``save_options`` dicts.
    Each ``save_options`` entry is a path string, a :class:`pathlib.Path`, or
    ``None`` (skip). The :attr:`save_options` setter drops ``None`` entries and
    normalises the rest to absolute :class:`pathlib.Path` objects via
    :func:`~niiflow.preproc.utils.file.resolve_path` (paths need not exist yet).

    Subclasses may set :attr:`REQUIRED_PARAMS` to declare keys that must be
    present in ``params``. Cross-parameter and conditional validation can be
    provided by overriding :meth:`check_params`.

    Subclasses MUST implement :meth:`load_param`, :meth:`save_output`, and
    :meth:`forward`; :meth:`check_params` and :meth:`update_metadata` are
    optional hooks with empty default implementations.

    String parameter values may reference :class:`RuntimeContext` fields using
    ``"ctx.<dotted.path>"`` (e.g. ``ctx.run_id``, ``ctx.metadata.subject``,
    ``ctx.artifacts.<step_id>.<output>``).
    References are resolved in
    :meth:`_load_params` before :meth:`load_param` is invoked for every key
    (except ``enable``, which is evaluated earlier via :meth:`_check_enabled`).

    An optional ``enable`` entry in ``params`` (default ``True``) is resolved and
    evaluated by :meth:`_check_enabled` **before** other parameters are loaded.
    When ``enable`` is ``False``, the stage returns without calling
    :meth:`forward`, writing outputs, or updating ``ctx.artifacts``,
    ``ctx.metadata``, or ``ctx.steps_completed``. Other ``params`` are not
    materialised when the stage is skipped. This pairs naturally with boolean QC
    artifacts (e.g. ``"enable": "ctx.artifacts.<qc_step>.passed"``).

    The machinery in :meth:`run` -- not subclasses -- owns publishing outputs to
    ``ctx.artifacts[step_id]`` and appending to ``ctx.steps_completed``.
    """

    REQUIRED_PARAMS: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        save_options: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        if not isinstance(verbose, bool):
            raise TypeError(
                f"`verbose` must be a boolean, got {type(verbose).__name__}"
            )
        self.verbose = verbose
        self.params = params
        self.save_options = save_options

    @property
    def params(self) -> dict[str, Any]:
        return deepcopy(self._params)

    @params.setter
    def params(self, params: dict[str, Any] | None) -> None:
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise TypeError(
                f"`params` must be a dictionary, got {type(params).__name__}"
            )
        missing = type(self).REQUIRED_PARAMS - params.keys()
        if missing:
            raise ValueError(
                f"Missing required parameter(s) for {type(self).__name__}: "
                f"{sorted(missing)}"
            )
        self.check_params(deepcopy(params))
        self._params = deepcopy(params)

    @property
    def save_options(self) -> dict[str, Path]:
        return deepcopy(self._save_options)

    @save_options.setter
    def save_options(self, save_options: dict[str, Any] | None) -> None:
        """Validate and normalise pipeline save paths.

        Accepts per-output ``None`` (skip), a path string or :class:`pathlib.Path`. Each
        path is expanded and resolved to an absolute :class:`pathlib.Path`.
        """
        if save_options is None:
            save_options = {}
        if not isinstance(save_options, dict):
            raise TypeError(
                f"`save_options` must be a dictionary, got "
                f"{type(save_options).__name__}"
            )

        normalized: dict[str, Path] = {}

        for key, value in deepcopy(save_options).items():
            if value is None:
                continue
            elif isinstance(value, (str, Path)):
                normalized[key] = resolve_path(value)
            else:
                raise TypeError(
                    f"`save_options['{key}']` must be a path string, Path object, "
                    f"or None; got {type(value).__name__}"
                )

        self._save_options = normalized

    def log(self, message: str, level: LogLevel = "info") -> None:
        """Emit ``message`` through the module logger at ``level``.

        ``DEBUG``/``INFO`` messages are suppressed when ``self.verbose`` is False;
        ``WARNING`` and above are always emitted so diagnostics are not lost.
        """
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"`level` must be one of {_VALID_LOG_LEVELS}, got {level!r}"
            )
        if level in ("debug", "info") and not self.verbose:
            return
        getattr(logger, level)(message)

    def check_params(self, params: dict[str, Any]) -> None:
        """Validate the configured ``params`` mapping.

        Optional hook; the default is a no-op. Called automatically by the
        :attr:`params` setter after :attr:`REQUIRED_PARAMS` presence checks and before
        the parameters are stored. Override to validate value types, cross-parameter
        constraints, and algorithm-specific options. Raise ``ValueError`` or
        ``TypeError`` when validation fails.

        Unrecognised keys are accepted by default so extra kwargs can be forwarded to
        the underlying functional.
        """
        return

    @abstractmethod
    def load_param(self, key: str, value: Any) -> Any:
        """Lazily materialise the parameter ``key`` from its configured ``value``.

        Called from :meth:`run` (via :meth:`_load_params`) for each entry in ``params``
        when the stage is enabled, after any ``ctx.``-style reference has been resolved.
        Implementations should pass through values that are already in final form
        (scalars, in-memory images resolved from ``ctx.artifacts``, pre-loaded arrays)
        and materialise others (paths, remote URIs) as needed for :meth:`forward`.
        """
        ...

    @abstractmethod
    def save_output(self, key: str, value: Any, output_path: Path) -> SavedPathRecord:
        """Persist one entry from :meth:`forward` to its configured artifact path.

        Called from :meth:`_save_outputs` when ``save_options`` contains a non-``None``
        entry for ``key``. ``output_path`` is one absolute :class:`pathlib.Path` that
        points to a single location to save the artifact for ``key``.

        For single-file artifacts, ``output_path`` is the file written directly. For
        compound artifacts, ``output_path`` may be an index/manifest file or equivalent
        primary record from which the full artifact can be recovered.

        Returns the :class:`SavedPathRecord` for the written artifact.
        """
        ...

    @abstractmethod
    def forward(self, **params: Any) -> dict[str, Any]:
        """Run the stage's transformation on loaded parameters.

        Receives materialised parameters as keyword arguments (one per ``params`` key).

        Must return a mapping of output names to values that represent distinct logical
        outputs of the stage; e.g., a chain of forward transforms from ANTs registration
        should be a single output.
        """
        ...

    def update_metadata(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Return per-step metadata to record under ``ctx.metadata[step_id]``.

        Optional hook; the default returns an empty dict. ``outputs`` is the in-memory
        mapping published at ``ctx.artifacts[step_id]`` (the ``forward`` return values,
        not substituted with save paths).

        :meth:`run` merges the returned dict and adds ``saved_paths`` when
        :meth:`save_outputs` wrote any files. Implementations must **not** touch the
        context directly.
        """
        return {}

    def _check_enabled(self, params: dict[str, Any], ctx: RuntimeContext) -> bool:
        """Resolve and evaluate ``enable`` without loading other parameters.

        ``params`` is not mutated. Returns ``True`` when the stage should run.
        """
        enable = params.get("enable", True)
        if isinstance(enable, str) and enable.startswith(_CTX_PREFIX):
            enable = _resolve_ctx_path(ctx, enable)
        if not isinstance(enable, bool):
            enable = self.load_param("enable", enable)
        if not isinstance(enable, bool):
            raise TypeError(f"`enable` must be a boolean, got {type(enable).__name__}")
        return enable

    def _load_params(
        self, params: dict[str, Any], ctx: RuntimeContext
    ) -> dict[str, Any]:
        """Resolve ``ctx.`` references and materialise every parameter via
        :meth:`load_param`.

        Returns a fresh dictionary; the input ``params`` is not mutated.
        """
        resolved: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith(_CTX_PREFIX):
                value = _resolve_ctx_path(ctx, value)
            resolved[key] = self.load_param(key, value)
        return resolved

    def _save_outputs(self, outputs: dict[str, Any]) -> dict[str, SavedPathRecord]:
        """Persist configured outputs and return saved artifact path records."""
        written: dict[str, SavedPathRecord] = {}
        for key, target in self.save_options.items():
            if target is None:
                continue
            if key not in outputs:
                raise ValueError(
                    f"`save_options` contains {key!r}, but "
                    f"`{type(self).__name__}.forward` did not return this output. "
                    f"Available outputs are {sorted(outputs)}."
                )
            output_path = resolve_path(target)
            record = self.save_output(key, outputs[key], output_path)
            self._validate_saved_path_record(
                key=key,
                record=record,
                output_path=output_path,
            )
            written[key] = record
        return written

    def _validate_saved_path_record(
        self,
        key: str,
        record: SavedPathRecord,
        output_path: Path,
    ) -> None:
        """Validate the saved-path record returned by ``save_output``.

        The configured ``output_path`` must be the canonical artifact path recorded for
        this output.
        """
        if isinstance(record, Path):
            if record != output_path:
                raise ValueError(
                    f"`{type(self).__name__}.save_output({key!r}, ...)` returned "
                    f"{record!s}, but the configured output path is {output_path!s}. "
                    "The returned Path must match the canonical artifact path."
                )
            return

        if isinstance(record, ArtifactPathRecord):
            if record.path != output_path:
                raise ValueError(
                    f"`{type(self).__name__}.save_output({key!r}, ...)` returned "
                    f"an ArtifactPathRecord with path {record.path!s}, but the "
                    f"configured output path is {output_path!s}. "
                    "`ArtifactPathRecord.path` must match the canonical artifact path."
                )

            if not isinstance(record.payload, tuple):
                raise TypeError(
                    f"`ArtifactPathRecord.payload` for output {key!r} must be a tuple "
                    f"of Path objects, got {type(record.payload).__name__}."
                )

            bad_payload = [
                path for path in record.payload if not isinstance(path, Path)
            ]
            if bad_payload:
                bad_types = sorted({type(path).__name__ for path in bad_payload})
                raise TypeError(
                    f"`ArtifactPathRecord.payload` for output {key!r} must contain "
                    f"only Path objects; got {bad_types}."
                )

            return

        raise TypeError(
            f"`{type(self).__name__}.save_output({key!r}, ...)` must return a "
            f"Path or ArtifactPathRecord, got {type(record).__name__}."
        )

    def _resolve_step_id(self, ctx: RuntimeContext) -> str:
        """Resolve the id under which this run publishes outputs.

        When ``ctx.step_id`` is omitted, an auto-generated id of the form ``step_0000``,
        ``step_0001``, ... is assigned from ``len(ctx.steps_completed)``. Raises when
        the resolved id already appears among previous steps.
        """
        completed = ctx.steps_completed
        if not isinstance(completed, list):
            raise TypeError(
                f"`ctx.steps_completed` must be a list, "
                f"got {type(completed).__name__}"
            )

        current_index = len(completed)
        if ctx.step_id is not None:
            step_id = ctx.step_id
            source = "user-provided"
        else:
            step_id = f"step_{current_index:04d}"
            source = "auto-generated"

        if not isinstance(step_id, str) or not step_id:
            raise TypeError(f"step id must be a non-empty string, got {step_id!r}")

        if step_id in completed:
            prior_index = completed.index(step_id)
            raise ValueError(
                f"{source} step_id {step_id!r} at index {current_index} overlaps "
                f"with index {prior_index}; step ids must be unique within a "
                f"pipeline."
            )

        return step_id

    def run(self, ctx: RuntimeContext | None = None) -> RuntimeContext:
        """Execute the stage end-to-end and return the resulting context.

        Args:
            ctx: Runtime context to thread through the stage. A fresh
                :class:`RuntimeContext` is created when omitted; ``ctx.step_id``
                may be omitted and will be auto-generated from
                ``ctx.steps_completed``.

        Returns:
            The mutated :class:`RuntimeContext`.
        """
        if ctx is None:
            ctx = RuntimeContext()
        elif not isinstance(ctx, RuntimeContext):
            raise TypeError(
                f"`ctx` must be a RuntimeContext or None, got {type(ctx).__name__}"
            )

        name = type(self).__name__
        step_id = self._resolve_step_id(ctx)
        ctx.step_id = step_id

        self.log(f"[Stage {name} | {step_id}] Running...")
        start = time.perf_counter()

        params = deepcopy(self.params)
        if not self._check_enabled(params, ctx):
            elapsed = time.perf_counter() - start
            self.log(
                f"[Stage {name} | {step_id}] Skipped (enable=False) "
                f"in {elapsed:.2f}s"
            )
            return ctx

        params.pop("enable", None)
        params = self._load_params(params, ctx)

        outputs = self.forward(**params)

        if not isinstance(outputs, dict):
            raise TypeError(
                f"`{name}.forward` must return a dict, got {type(outputs).__name__}"
            )

        ctx.artifacts[step_id] = dict(outputs)

        self.log(f"[Stage {name} | {step_id}] Writing outputs...")
        written = self._save_outputs(outputs)

        metadata = self.update_metadata(ctx.artifacts[step_id])
        if not isinstance(metadata, dict):
            raise TypeError(
                f"`{name}.update_metadata` must return a dict, "
                f"got {type(metadata).__name__}"
            )
        if written:
            metadata = {**metadata, "saved_paths": _serialize_saved_paths(written)}
        ctx.metadata[step_id] = metadata

        ctx.steps_completed.append(step_id)

        elapsed = time.perf_counter() - start
        self.log(f"[Stage {name} | {step_id}] Finished in {elapsed:.2f}s")

        return ctx

    __call__ = run

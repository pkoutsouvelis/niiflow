"""Base class for preprocessing stages."""

from __future__ import annotations

__all__ = [
    "ProcessingStage",
    "RuntimeContext",
]

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args
import inspect

from niiflow.preproc.utils.file import resolve_path

logger = logging.getLogger(__name__)


LogLevel = Literal["debug", "info", "warning", "error", "critical"]
_VALID_LOG_LEVELS: tuple[str, ...] = get_args(LogLevel)
_CTX_PREFIX = "ctx."


@dataclass
class RuntimeContext:
    """Runtime context threaded through preprocessing stages.

    Attributes:
        input_root: Optional root directory for the inputs being processed.
            Resolved to an absolute :class:`pathlib.Path` on construction.
        metadata: Free-form mapping describing the current execution
            (e.g., subject id, modality).
        artifacts: Mapping of intermediate outputs produced by a stage that
            downstream stages may consume.
        state: Mapping of bookkeeping state used by pipeline composition.
    """

    input_root: Path | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_root is not None:
            self.input_root = resolve_path(self.input_root)


def _resolve_ctx_path(ctx: RuntimeContext, dotted: str) -> Any:
    """Resolve a dotted ``ctx.`` reference against a :class:`RuntimeContext`.

    Each path segment is dispatched as an item lookup when the current object is
    a :class:`dict`, and as an attribute lookup otherwise. This lets references
    such as ``"ctx.input_root"``, ``"ctx.artifacts.brain_mask"`` or
    ``"ctx.metadata.subject_id"`` all work uniformly.
    """
    if not isinstance(dotted, str) or not dotted.startswith(_CTX_PREFIX):
        raise ValueError(f"Context path must start with 'ctx.', got {dotted!r}")
    parts = dotted.split(".")[1:]
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"Context path {dotted!r} has empty segments")

    obj: Any = ctx
    walked: list[str] = ["ctx"]
    for part in parts:
        walked.append(part)
        try:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        except (AttributeError, KeyError) as e:
            raise KeyError(
                f"Failed to resolve context path '{'.'.join(walked)}': {e}"
            ) from e
    return obj


def _validate_key_set(
    actual: dict[str, Any], expected: list[str] | tuple[str, ...], label: str
) -> None:
    """Raise ``ValueError`` if ``actual``'s keys don't match ``expected`` exactly."""
    actual_keys = set(actual.keys())
    expected_keys = set(expected)
    if actual_keys == expected_keys:
        return
    parts: list[str] = []
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        parts.append(f"missing {missing}")
    if extra:
        parts.append(f"unexpected {extra}")
    raise ValueError(
        f"`{label}` must match keys {sorted(expected_keys)}; " + ", ".join(parts) + "."
    )


class ProcessingStage(ABC):
    """Base class for atomic preprocessing stages.

    A stage encapsulates a single transformation step: validated input parameters,
    a deterministic :meth:`forward` pass producing named outputs, optional
    persistence of those outputs, and an update of the shared
    :class:`RuntimeContext`.

    Concrete subclasses MUST declare three class attributes that describe the
    stage's interface:

    * :attr:`PARAM_KEYS` -- names of arguments consumed by :meth:`forward`.
    * :attr:`FILE_INPUT_KEYS` -- subset of :attr:`PARAM_KEYS` whose values are
      file inputs and require lazy loading via :meth:`load_params`.
    * :attr:`OUTPUT_KEYS` -- names of values returned by :meth:`forward`.

    Subclasses MUST also implement :meth:`check_params`, :meth:`load_params`,
    :meth:`save_outputs`, :meth:`forward`, and :meth:`update_ctx`. The above
    contract is enforced at class-creation time on every concrete subclass via
    :meth:`__init_subclass__`.

    Parameters values may reference fields stored on the :class:`RuntimeContext`
    using a ``"ctx.<dotted.path>"`` string. References are resolved by
    :meth:`_load_params` immediately before :meth:`forward` is invoked.
    """

    PARAM_KEYS: ClassVar[list[str]]
    FILE_INPUT_KEYS: ClassVar[list[str]]
    OUTPUT_KEYS: ClassVar[list[str]]

    _STAGE_CLASS_ATTRS: ClassVar[tuple[str, ...]] = (
        "PARAM_KEYS",
        "FILE_INPUT_KEYS",
        "OUTPUT_KEYS",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        for attr in cls._STAGE_CLASS_ATTRS:
            if not hasattr(cls, attr):
                raise TypeError(
                    f"Concrete `ProcessingStage` subclass `{cls.__name__}` must "
                    f"define class attribute `{attr}`."
                )
            value = getattr(cls, attr)
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(k, str) for k in value
            ):
                raise TypeError(
                    f"`{cls.__name__}.{attr}` must be a list or tuple of strings, "
                    f"got {value!r}."
                )
            if len(set(value)) != len(value):
                raise TypeError(
                    f"`{cls.__name__}.{attr}` must contain unique entries, "
                    f"got {list(value)!r}."
                )

        unknown_file_keys = set(cls.FILE_INPUT_KEYS) - set(cls.PARAM_KEYS)
        if unknown_file_keys:
            raise TypeError(
                f"`{cls.__name__}.FILE_INPUT_KEYS` must be a subset of `PARAM_KEYS`; "
                f"unknown keys: {sorted(unknown_file_keys)}."
            )

    def __init__(
        self,
        params: dict[str, Any],
        save_options: dict[str, Any],
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
        return self._params

    @params.setter
    def params(self, params: dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise TypeError(
                f"`params` must be a dictionary, got {type(params).__name__}"
            )
        _validate_key_set(params, self.PARAM_KEYS, "params")
        # ``check_params`` may inspect / partially consume the dict; isolate it.
        self.check_params(deepcopy(params))
        self._params = deepcopy(params)

    @property
    def save_options(self) -> dict[str, Any]:
        return self._save_options

    @save_options.setter
    def save_options(self, save_options: dict[str, Any]) -> None:
        if not isinstance(save_options, dict):
            raise TypeError(
                f"`save_options` must be a dictionary, got "
                f"{type(save_options).__name__}"
            )
        _validate_key_set(save_options, self.OUTPUT_KEYS, "save_options")
        for key, value in save_options.items():
            if value is not None and not isinstance(value, (Path, str)):
                raise TypeError(
                    f"`save_options['{key}']` must be a Path, str, or None; "
                    f"got {type(value).__name__}"
                )
        self._save_options = deepcopy(save_options)

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

    @abstractmethod
    def check_params(self, params: dict[str, Any]) -> None:
        """Validate the per-stage configuration ``params``.

        Called automatically by the :attr:`params` setter with a deep copy of the
        candidate ``params`` dict. The setter has already checked that the keys exactly
        match :attr:`PARAM_KEYS`; implementations should focus on validating values and
        raise ``ValueError`` / ``TypeError`` as appropriate.
        """
        ...

    @abstractmethod
    def load_params(self, key: str, value: Any) -> Any:
        """Lazily materialise a file-input ``value`` for parameter ``key``.

        Called from :meth:`run` (via :meth:`_load_params`) for every key in
        :attr:`FILE_INPUT_KEYS`, after any ``ctx.``-style reference has been resolved.
        Implementations may return the loaded object or pass through the path itself, as
        appropriate for the stage.
        """
        ...

    @abstractmethod
    def save_outputs(self, key: str, value: Any, output_path: Path) -> None:
        """Persist a single output ``value`` to ``output_path``.

        Called from :meth:`_save_outputs` for every key in :attr:`OUTPUT_KEYS` whose
        ``save_options`` entry is not ``None``. ``output_path`` is guaranteed to be an
        absolute :class:`pathlib.Path`.
        """
        ...

    @abstractmethod
    def forward(self, **params: Any) -> dict[str, Any]:
        """Run the stage's transformation.

        Receives the loaded parameters as keyword arguments and must return a
        ``dict`` whose keys exactly match :attr:`OUTPUT_KEYS`.
        """
        ...

    @abstractmethod
    def update_ctx(
        self, outputs: dict[str, Any], ctx: RuntimeContext
    ) -> RuntimeContext:
        """Return an updated :class:`RuntimeContext` reflecting this stage's outputs.

        Implementations MAY add or overwrite entries in ``ctx.metadata``,
        ``ctx.artifacts`` and ``ctx.state``, but MUST NOT remove keys that were present
        in the incoming context (enforced by :meth:`_update_ctx`).
        """
        ...

    def _load_params(
        self, params: dict[str, Any], ctx: RuntimeContext
    ) -> dict[str, Any]:
        """Resolve ``ctx.`` references and run :meth:`load_params` on file inputs.

        Returns a fresh dictionary; the input ``params`` is not mutated.
        """
        file_keys = set(self.FILE_INPUT_KEYS)
        resolved: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith(_CTX_PREFIX):
                value = _resolve_ctx_path(ctx, value)
            if key in file_keys:
                value = self.load_params(key, value)
            resolved[key] = value
        return resolved

    def _save_outputs(self, outputs: dict[str, Any]) -> None:
        """Dispatch each output to :meth:`save_outputs` when a target path is set."""
        for key, value in outputs.items():
            target = self.save_options[key]
            if target is None:
                continue
            self.save_outputs(key, value, resolve_path(target))

    def _update_ctx(
        self, outputs: dict[str, Any], ctx: RuntimeContext
    ) -> RuntimeContext:
        """Run :meth:`update_ctx` and reject removal of previously-set ctx keys."""
        prev_keys = {
            "metadata": set(ctx.metadata),
            "artifacts": set(ctx.artifacts),
            "state": set(ctx.state),
        }
        updated = self.update_ctx(outputs, ctx)
        if not isinstance(updated, RuntimeContext):
            raise TypeError(
                f"`update_ctx` must return a RuntimeContext, "
                f"got {type(updated).__name__}"
            )
        new_keys = {
            "metadata": set(updated.metadata),
            "artifacts": set(updated.artifacts),
            "state": set(updated.state),
        }
        for section, before in prev_keys.items():
            lost = before - new_keys[section]
            if lost:
                raise ValueError(
                    f"`update_ctx` removed previously-present {section} keys: "
                    f"{sorted(lost)}"
                )
        return updated

    def run(self, ctx: RuntimeContext | None = None) -> RuntimeContext:
        """Execute the stage end-to-end and return the resulting context.

        The execution flow is: load parameters (resolve ``ctx.`` references and
        run :meth:`load_params` on file inputs) -> :meth:`forward` ->
        :meth:`update_ctx` -> :meth:`save_outputs`.

        Args:
            ctx: Runtime context to thread through the stage. A fresh empty
                :class:`RuntimeContext` is created when omitted, so a stage can
                be invoked standalone.

        Returns:
            The :class:`RuntimeContext` produced by :meth:`update_ctx`.
        """
        if ctx is None:
            ctx = RuntimeContext()
        elif not isinstance(ctx, RuntimeContext):
            raise TypeError(
                f"`ctx` must be a RuntimeContext or None, got {type(ctx).__name__}"
            )

        name = type(self).__name__
        self.log(f"[Stage {name}] Running...")

        params = self._load_params(deepcopy(self.params), ctx)
        outputs = self.forward(**params)

        if not isinstance(outputs, dict):
            raise TypeError(
                f"`{name}.forward` must return a dict, got {type(outputs).__name__}"
            )
        _validate_key_set(outputs, self.OUTPUT_KEYS, f"{name}.forward outputs")

        ctx = self._update_ctx(outputs, ctx)

        self.log(f"[Stage {name}] Writing outputs...")
        self._save_outputs(outputs)
        self.log(f"[Stage {name}] Done.")

        return ctx

    __call__ = run


class DummyProcessingStage(ProcessingStage):
    """Minimal concrete stage for development and integration smoke tests.

    Applies a scalar ``scale_factor`` to a lazily loaded ``input_nii`` and stores
    the outcome on ``ctx.artifacts['output_nii']``. Mirrors the usual stage
    contract: file param, optional ``ctx.`` reference, and keyed ``save_options``.
    """

    PARAM_KEYS = ["input_nii", "scale_factor"]
    FILE_INPUT_KEYS = ["input_nii"]
    OUTPUT_KEYS = ["output_nii"]

    def check_params(self, params: dict[str, Any]) -> None:
        if not isinstance(params["scale_factor"], (int, float)):
            raise ValueError("`scale_factor` must be numeric")
        if not isinstance(params["input_nii"], (str, Path)):
            raise ValueError("`input_nii` must be a path-like string")

    def load_params(self, key: str, value: Any) -> Any:
        if key == "input_nii":
            path = Path(value)
            return path.read_bytes() if path.is_file() else b""
        return value

    def save_outputs(self, key: str, value: Any, output_path: Path) -> None:
        output_path.write_bytes(
            value if isinstance(value, bytes) else repr(value).encode()
        )

    def forward(self, *, input_nii: bytes, scale_factor: float) -> dict[str, Any]:
        scaled = (
            bytes(min(255, int(b * scale_factor)) for b in input_nii)
            if input_nii
            else b""
        )
        return {"output_nii": scaled}

    def update_ctx(
        self, outputs: dict[str, Any], ctx: RuntimeContext
    ) -> RuntimeContext:
        ctx.artifacts["output_nii"] = outputs["output_nii"]
        return ctx

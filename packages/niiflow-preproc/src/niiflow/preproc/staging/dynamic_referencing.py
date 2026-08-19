"""Dynamic reference replacement for staged params."""

from __future__ import annotations

__all__ = [
    "DynamicReferenceError",
    "resolve_dynamic_refs",
    "ResolveActiveReferences",
    "ResolveParamReferences",
    "add_reference_staging_bookends",
]

import re
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, TypeVar

from niiflow.preproc.utils.misc import get_by_dotted_path, split_dotted_path
from .stager import StagingContext, StagedEntry, Stager
from .validation import ensure_file

T = TypeVar("T")

_REF_PATTERN = re.compile(r"\{([^{}]+)\}")


class DynamicReferenceError(ValueError):
    """Raised when a dynamic reference cannot be resolved."""


def resolve_dynamic_refs(
    value: T,
    *,
    params: dict[str, Any],
    ctx: StagingContext,
    resolve_params: bool,
) -> T:
    """Recursively replace dynamic references.

    Supported references
    --------------------
    Active-file references, always resolved (``ctx.active`` is a
    :class:`~pathlib.Path`):

    - ``{active}`` — the active path itself
    - ``{active.name}``, ``{active.stem}``, ``{active.parent}``, … — any
      :class:`~pathlib.Path` attribute via :func:`getattr` (pathlib ``stem``
      drops only the final suffix; for ``.nii.gz`` prefer
      ``{active.name|strip:.nii.gz}``)

    Parameter references, resolved only when ``resolve_params=True``:

    - ``{params.some.dotted.path}`` — nested dict/list walk, returns any value
    - ``{params.some.object.attr}`` — if the full dotted path is not a dict/list
      walk, the last segment is taken as an attribute on the parent object

    Supported modifiers
    -------------------
    References may append ``|<modifier>`` or ``|<modifier>:<arg>``. Modifiers are
    registered callables from :mod:`niiflow.preproc.staging.modifiers` and
    validate their own input types. Built-in: ``strip:<suffix>`` (``str`` /
    ``Path`` only).

    Native type behavior
    --------------------
    A string that is exactly one reference returns the resolved object in its
    native type. Embedded references are interpolated as text.
    """
    if isinstance(value, str):
        return _replace_in_string(
            value,
            params=params,
            ctx=ctx,
            resolve_params=resolve_params,
        )  # type: ignore[return-value]

    if isinstance(value, list):
        return [
            resolve_dynamic_refs(
                item,
                params=params,
                ctx=ctx,
                resolve_params=resolve_params,
            )
            for item in value
        ]  # type: ignore[return-value]

    if isinstance(value, tuple):
        return tuple(
            resolve_dynamic_refs(
                item,
                params=params,
                ctx=ctx,
                resolve_params=resolve_params,
            )
            for item in value
        )  # type: ignore[return-value]

    if isinstance(value, dict):
        return {
            key: resolve_dynamic_refs(
                item,
                params=params,
                ctx=ctx,
                resolve_params=resolve_params,
            )
            for key, item in value.items()
        }  # type: ignore[return-value]

    return value


def _replace_in_string(
    text: str,
    *,
    params: dict[str, Any],
    ctx: StagingContext,
    resolve_params: bool,
) -> Any:
    matches = list(_REF_PATTERN.finditer(text))
    if not matches:
        return text

    # Preserve unresolved params refs during the first pass.
    if len(matches) == 1 and matches[0].span() == (0, len(text)):
        ref = matches[0].group(1).strip()

        if ref.startswith("params.") and not resolve_params:
            return text

        return _resolve_ref(
            ref,
            params=params,
            ctx=ctx,
            resolve_params=resolve_params,
        )

    out = text

    for match in reversed(matches):
        ref = match.group(1).strip()

        if ref.startswith("params.") and not resolve_params:
            continue

        resolved = _resolve_ref(
            ref,
            params=params,
            ctx=ctx,
            resolve_params=resolve_params,
        )

        if resolved is None:
            raise DynamicReferenceError(
                f"Dynamic reference {{{ref}}} resolved to None, but it was used "
                f"inside a larger string: {text!r}. A None value can only replace "
                "the entire string."
            )

        out = out[: match.start()] + str(resolved) + out[match.end() :]

    return out


def _resolve_ref(
    ref: str,
    *,
    params: dict[str, Any],
    ctx: StagingContext,
    resolve_params: bool,
) -> Any:
    original_ref = ref
    ref, modifier = _split_modifier(ref)

    if ref == "active" or ref.startswith("active."):
        if ref == "active":
            value: Any = ctx.active
        else:
            attr = ref.removeprefix("active.")
            if not attr or "." in attr:
                raise DynamicReferenceError(
                    f"Active reference must be {{active}} or {{active.<attr>}}, "
                    f"got {{{original_ref}}}."
                )
            value = _resolve_object_attr(ctx.active, attr, original_ref)
        return _apply_modifier(value, modifier, original_ref)

    if ref.startswith("params."):
        if not resolve_params:
            return "{" + original_ref + "}"

        body = ref.removeprefix("params.")
        if not body:
            raise DynamicReferenceError("Empty params reference: {params}.")

        try:
            # Prioritize fully resolved param values over attribute lookups.
            value = get_by_dotted_path(params, body)
        except (KeyError, TypeError, ValueError):
            parts = split_dotted_path(body)
            if len(parts) < 2 or not isinstance(parts[-1], str):
                raise DynamicReferenceError(
                    f"Failed to resolve params reference {{{original_ref}}}."
                )
            attr = parts[-1]
            parent_path = _join_dotted_parts(parts[:-1])
            try:
                parent = get_by_dotted_path(params, parent_path)
            except (KeyError, TypeError, ValueError) as exc:
                raise DynamicReferenceError(
                    f"Failed to resolve params reference {{{original_ref}}}."
                ) from exc
            value = _resolve_object_attr(parent, attr, original_ref)

        return _apply_modifier(value, modifier, original_ref)

    raise DynamicReferenceError(
        f"Unknown dynamic reference {{{original_ref}}}. Expected 'active', "
        "'active.<attr>', or 'params.<dotted-path>[.<attr>]'."
    )


def _join_dotted_parts(parts: list[str | int]) -> str:
    """Rebuild a dotted path from :func:`split_dotted_path` parts."""
    tokens: list[str] = []
    for part in parts:
        if isinstance(part, int):
            tokens.append(f"[{part}]")
        else:
            tokens.append(part)
    return ".".join(tokens)


def _resolve_object_attr(value: Any, attr: str, original_ref: str) -> Any:
    """Resolve ``.<attr>`` on an object via :func:`getattr`."""
    # Treat None as valid input; let downstream processing handle it.
    if value is None:
        return None

    if isinstance(value, str) and _REF_PATTERN.search(value):
        raise DynamicReferenceError(
            f"Cannot read attribute {attr!r} from unresolved dynamic reference "
            f"{value!r} in reference {{{original_ref}}}."
        )

    try:
        return getattr(value, attr)
    except AttributeError as exc:
        raise DynamicReferenceError(
            f"Object of type {type(value).__name__} has no attribute {attr!r} "
            f"in reference {{{original_ref}}}."
        ) from exc


def _split_modifier(ref: str) -> tuple[str, str | None]:
    """Split a dynamic reference into base reference and optional modifier."""
    if "|" not in ref:
        return ref.strip(), None

    base, modifier = ref.split("|", 1)
    base = base.strip()
    modifier = modifier.strip()

    if not base:
        raise DynamicReferenceError(f"Invalid dynamic reference {{{ref}}}.")

    if not modifier:
        raise DynamicReferenceError(f"Empty dynamic-reference modifier in {{{ref}}}.")

    return base, modifier


def _apply_modifier(value: Any, modifier: str | None, original_ref: str) -> Any:
    """Apply an optional exportable modifier; map ``ValueError`` to reference errors."""
    if modifier is None:
        return value

    from .modifiers import get_modifiers

    if ":" in modifier:
        name, arg = modifier.split(":", 1)
        name = name.strip()
    else:
        name, arg = modifier.strip(), None

    if not name:
        raise DynamicReferenceError(
            f"Empty dynamic-reference modifier in {{{original_ref}}}."
        )

    modifiers = get_modifiers()
    try:
        fn = modifiers[name]
    except KeyError as exc:
        supported = ", ".join(sorted(modifiers)) or "(none)"
        raise DynamicReferenceError(
            f"Unsupported dynamic-reference modifier {modifier!r} in reference "
            f"{{{original_ref}}}. Supported modifiers: {supported}."
        ) from exc

    try:
        return fn(value, arg=arg, ref=original_ref)
    except ValueError as exc:
        raise DynamicReferenceError(str(exc)) from exc


def add_reference_staging_bookends(
    stagers: Sequence[Stager],
) -> list[Stager]:
    """Wrap ``stagers`` with active-ref then param-ref resolution stagers.

    ``stagers`` is an ordered chain (list/tuple). Leading
    :class:`ResolveActiveReferences` and trailing :class:`ResolveParamReferences`
    instances already present are dropped so bookending is idempotent.
    """
    middle = list(stagers)
    while middle and type(middle[0]) is ResolveActiveReferences:
        middle.pop(0)
    while middle and type(middle[-1]) is ResolveParamReferences:
        middle.pop()
    return [ResolveActiveReferences(), *middle, ResolveParamReferences()]


class ResolveActiveReferences(Stager):
    """Expand ``{active.*}`` references in each entry's ``params`` tree.

    ``{params.*}`` references are left untouched for later stagers. The active path is
    used as a path-attribute anchor and need not exist on disk.
    """

    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        if entry.errors:
            return entry
        active = ensure_file(entry.active, must_exist=False)
        ctx = StagingContext(active=active)
        params = deepcopy(entry.params)
        params = resolve_dynamic_refs(
            params,
            params=params,
            ctx=ctx,
            resolve_params=False,
        )
        return StagedEntry(active=entry.active, params=params, errors=entry.errors)


class ResolveParamReferences(Stager):
    """Expand ``{params.*}`` (and any remaining ``{active.*}``) in ``params``."""

    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        if entry.errors:
            return entry
        active = ensure_file(entry.active, must_exist=False)
        ctx = StagingContext(active=active)
        params = deepcopy(entry.params)
        params = resolve_dynamic_refs(
            params,
            params=params,
            ctx=ctx,
            resolve_params=True,
        )
        return StagedEntry(active=entry.active, params=params, errors=entry.errors)

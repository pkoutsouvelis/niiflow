"""Dynamic reference replacement for staged params."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypeVar

from niiflow.preproc.utils.misc import get_by_dotted_path
from niiflow.preproc.utils.file import get_ext
from .stager import StageContext

T = TypeVar("T")

_REF_PATTERN = re.compile(r"\{([^{}]+)\}")
_PATH_ATTRS = {"path", "name", "stem", "parent"}


class DynamicReferenceError(ValueError):
    """Raised when a dynamic reference cannot be resolved."""


def resolve_dynamic_refs(
    value: T,
    *,
    params: dict[str, Any],
    ctx: StageContext,
    resolve_params: bool,
) -> T:
    """Recursively replace dynamic references.

    Supported references
    --------------------
    Active-file references, always resolved:

    - ``{active.path}``
    - ``{active.name}``
    - ``{active.stem}``
    - ``{active.parent}``

    Parameter references, resolved only when ``resolve_params=True``:

    - ``{params.some.dotted.path}``
    - ``{params.some.dotted.path.path}``
    - ``{params.some.dotted.path.name}``
    - ``{params.some.dotted.path.stem}``
    - ``{params.some.dotted.path.parent}``

    Supported modifiers
    -------------------
    References may use ``|strip:<suffix>`` to remove a suffix from the resolved
    string value:

    - ``{active.stem|strip:_bet}``
    - ``{params.inputs.brain.stem|strip:_bet}``

    ``strip`` uses ``str.removesuffix`` semantics. It removes the suffix only if
    the resolved value ends with that suffix.

    Native type behavior
    --------------------
    If the whole string is exactly one reference, the resolved object is returned
    in its native type. For example, ``"{params.inputs.t1w.path}"`` may return a
    ``Path``.

    If the reference is embedded inside a larger string, it is interpolated as
    text.

    If ``|strip:...`` is used, the resolved value is converted to ``str`` and the
    returned value is therefore a string.
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
    ctx: StageContext,
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
    ctx: StageContext,
    resolve_params: bool,
) -> Any:
    original_ref = ref
    ref, modifier = _split_modifier(ref)

    if ref.startswith("active."):
        value = _extract_path_attr(
            ctx.active,
            ref.removeprefix("active."),
            original_ref,
        )
        return _apply_modifier(value, modifier, original_ref)

    if ref.startswith("params."):
        if not resolve_params:
            return "{" + original_ref + "}"

        body = ref.removeprefix("params.")
        if not body:
            raise DynamicReferenceError("Empty params reference: {params}.")

        parts = body.split(".")
        attr = parts[-1] if parts[-1] in _PATH_ATTRS else None
        dotted_path = ".".join(parts[:-1]) if attr is not None else body

        if not dotted_path:
            raise DynamicReferenceError(f"Invalid params reference {{{original_ref}}}.")

        target = get_by_dotted_path(params, dotted_path)
        value = (
            target if attr is None else _extract_path_attr(target, attr, original_ref)
        )

        return _apply_modifier(value, modifier, original_ref)

    raise DynamicReferenceError(
        f"Unknown dynamic reference {{{original_ref}}}. Expected 'active.<attr>' "
        "or 'params.<dotted-path>[.<attr>]'."
    )


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
    """Apply an optional modifier to a resolved reference value."""
    if modifier is None:
        return value

    if modifier.startswith("strip:"):
        suffix = modifier.removeprefix("strip:")

        if suffix == "":
            raise DynamicReferenceError(
                f"Empty strip suffix in reference {{{original_ref}}}."
            )

        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            raise DynamicReferenceError(
                f"Cannot apply modifier {modifier!r} to sequence value in "
                f"reference {{{original_ref}}}."
            )

        return str(value).removesuffix(suffix)

    raise DynamicReferenceError(
        f"Unsupported dynamic-reference modifier {modifier!r} in reference "
        f"{{{original_ref}}}. Supported modifiers: strip:<suffix>."
    )


def _extract_path_attr(value: Any, attr: str, original_ref: str) -> Any:
    """Extract a supported path-like attribute from a value."""
    if attr not in _PATH_ATTRS:
        raise DynamicReferenceError(
            f"Unsupported path attribute {attr!r} in reference {{{original_ref}}}. "
            "Expected one of: path, name, stem, parent."
        )

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        raise DynamicReferenceError(
            f"Cannot extract path attribute {attr!r} from sequence in reference "
            f"{{{original_ref}}}."
        )

    path = value if isinstance(value, Path) else Path(str(value))

    if attr == "path":
        return path

    if attr == "name":
        return path.name

    if attr == "stem":
        ext = get_ext(path)
        if ext:
            return path.name[: -len(ext)]
        return path.name

    if attr == "parent":
        return path.parent

    raise AssertionError("unreachable")

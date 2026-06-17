"""Factory for instantiating a data explorer with user-provided settings."""

from __future__ import annotations

__all__ = [
    "get_data_explorer",
]

import importlib
from typing import Any, Sequence, cast

from nifti_finder.explorers import AllPurposeFileExplorer
from nifti_finder.filters import Filter

from .types import FilterConfig, ComposeFilterKwargs, DataExplorer

_FILTER_MODULES: tuple[str, ...] = ("nifti_finder.filters",)


def _get_filter_name_and_kwargs(entry: FilterConfig) -> dict[str, Any]:
    if "name" not in entry:
        raise ValueError("`name` key is required in filter entry")
    if "kwargs" not in entry:
        raise ValueError("`kwargs` key is required in filter entry")
    return {"filter_name": entry["name"], "kwargs": entry["kwargs"]}


def _get_filter_obj(filter_name: str, kwargs: dict[str, Any]) -> Filter:
    for module_name in _FILTER_MODULES:
        try:
            module = importlib.import_module(module_name)
            filter_cls = getattr(module, filter_name)
            return filter_cls(**kwargs)
        except (ModuleNotFoundError, AttributeError):
            continue
    raise ImportError(
        f"Filter '{filter_name}' not found in any known modules: {list(_FILTER_MODULES)}"
    )


def _build_filter(filters: FilterConfig | None) -> Filter | None:
    """Recursively instantiate filters from a nested mapping.

    Accepts the same shape your existing factory uses:
      - ``None``                                       -> no filter
      - ``{name: "<ClassName>", kwargs: {...}}``       -> single filter
      - ``{name: "ComposeFilter", kwargs: {filters: <single|list>, logic: "AND"|"OR"}}``
    """
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise ValueError(f"Invalid type for `filters`: {type(filters).__name__}")

    name_and_kwargs = _get_filter_name_and_kwargs(filters)

    if name_and_kwargs["filter_name"] == "ComposeFilter":
        inner = cast(ComposeFilterKwargs, name_and_kwargs["kwargs"])["filters"]
        if isinstance(inner, dict):
            return _build_filter(inner)
        if isinstance(inner, list):
            new_kwargs = dict(name_and_kwargs["kwargs"])
            composed: list[Filter] = []
            for entry in inner:
                if entry is None:
                    continue
                composed.append(cast(Filter, _build_filter(entry)))
            new_kwargs["filters"] = composed
            return _get_filter_obj("ComposeFilter", new_kwargs)
        raise ValueError(
            f"Invalid type for `filters` in `ComposeFilter`: {type(inner).__name__}"
        )

    return _get_filter_obj(**name_and_kwargs)


def get_data_explorer(
    pattern: str | Sequence[str] | None = None,
    filter_kwargs: FilterConfig | None = None,
) -> DataExplorer:
    """Instantiate nifti-finder's `AllPurposeFileExplorer` with user-provided settings.

    Args:
        pattern: A string or list of strings representing the pattern to match.
        filter_kwargs: A dictionary of keyword arguments to pass to the filter.

    Returns:
        An `AllPurposeFileExplorer` instance.
    """
    pattern = pattern or "*"
    flt = _build_filter(filter_kwargs) if filter_kwargs else None
    if not isinstance(pattern, (str, list)):
        raise ValueError(
            f"`pattern` must be a string or list of strings, got {type(pattern).__name__}"
        )
    if isinstance(pattern, list) and any(not isinstance(p, str) for p in pattern):
        raise ValueError(
            f"`pattern` must be a list of strings, got {type(pattern).__name__}"
        )

    return AllPurposeFileExplorer(pattern, filters=flt)

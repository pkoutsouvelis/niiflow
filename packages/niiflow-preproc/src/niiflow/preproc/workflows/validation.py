"""Validation helpers for workflow execution inputs."""

from __future__ import annotations

from collections.abc import Sequence

from niiflow.preproc.staging import StagedEntry


def validate_staged_entries(
    entries: Sequence[StagedEntry],
) -> tuple[StagedEntry, ...]:
    seen_ids: set[str] = set()
    validated: list[StagedEntry] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, StagedEntry):
            raise TypeError(
                "`entries` must contain only `StagedEntry` objects, got "
                f"{type(entry).__name__} at index {index}"
            )

        if not isinstance(entry.id, str) or not entry.id:
            raise ValueError(
                f"StagedEntry for `{entry.active}` must have a non-empty string `id`"
            )

        if entry.id in seen_ids:
            raise ValueError(
                f"StagedEntry IDs must be unique; duplicate `{entry.id}` "
                f"at index {index}"
            )

        seen_ids.add(entry.id)
        validated.append(entry)

    return tuple(validated)

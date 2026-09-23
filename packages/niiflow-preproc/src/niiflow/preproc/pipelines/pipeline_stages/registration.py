"""Pipeline stages for registration."""

from __future__ import annotations

__all__ = [
    "ANTsApplyTransforms",
    "ANTsRegistration",
]

import shutil
from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.registration import (
    ants_apply_transforms,
    ants_registration,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import (
    ArtifactPathRecord,
    PipelineStage,
)
from niiflow.preproc.utils.file import (
    ants_image_read,
    ants_image_write,
    get_ext,
    read_json,
    write_json,
)

_ALLOWED_TRANSFORM_EXT = frozenset({".mat", ".nii", ".nii.gz", ".h5"})
_TRANSFORM_MANIFEST_EXT = ".json"


def _load_transform_manifest(
    manifest_path: str | Path,
) -> tuple[list[str], list[bool]]:
    """Load one ANTs transform-chain manifest.

    The manifest must contain a ``transforms`` list. Each entry must contain:

    * ``path``: transform path, relative to the manifest directory or absolute;
    * ``invert``: ANTs ``whichtoinvert`` flag for that transform.

    Entry order is the ANTs application order for that chain.
    """
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)

    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "ants_transform_chain"
    ):
        raise ValueError(
            f"Transform manifest {manifest_path!s} is not an "
            "ANTs transform-chain manifest."
        )

    entries = manifest.get("transforms")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Transform manifest {manifest_path!s} must contain a non-empty "
            "'transforms' list."
        )

    paths: list[str] = []
    invert_flags: list[bool] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(
                f"Transform manifest {manifest_path!s} entry {index} must "
                "be a dictionary."
            )

        if "path" not in entry:
            raise ValueError(
                f"Transform manifest {manifest_path!s} entry {index} is "
                "missing required key 'path'."
            )

        if "invert" not in entry:
            raise ValueError(
                f"Transform manifest {manifest_path!s} entry {index} is "
                "missing required key 'invert'."
            )

        raw_path = entry["path"]
        if not isinstance(raw_path, str):
            raise TypeError(
                f"Transform manifest {manifest_path!s} entry {index} has "
                f"non-string 'path': {type(raw_path).__name__}."
            )

        transform_path = Path(raw_path)
        if not transform_path.is_absolute():
            transform_path = manifest_path.parent / transform_path

        if not transform_path.is_file():
            raise FileNotFoundError(f"Transform file not found: {transform_path}")

        transform_ext = get_ext(transform_path).lower()
        if transform_ext not in _ALLOWED_TRANSFORM_EXT:
            raise ValueError(
                f"Unknown transform extension {transform_ext!r} in manifest "
                f"{manifest_path!s} entry {index}: {transform_path!s}."
            )

        invert = entry["invert"]
        if not isinstance(invert, bool):
            raise TypeError(
                f"Transform manifest {manifest_path!s} entry {index} has "
                f"non-boolean 'invert': {type(invert).__name__}."
            )

        paths.append(str(transform_path))
        invert_flags.append(invert)

    return paths, invert_flags


def _resolve_transformlist(
    value: Any,
) -> tuple[list[str], list[bool | None]]:
    """Materialise ``transformlist`` from transform files and/or manifests.

    Accepted forms:

    * a path to one ``.json`` manifest;
    * a non-empty list of transform file paths;
    * a non-empty list of manifests, transform files, or both.

    List entries are expanded in order. A manifest contributes its stored
    transforms at that position. The flattened sequence is the ANTs
    ``transformlist``.

    The return value is ``(paths, invert_flags)``. Flags come from manifests;
    an individual transform file contributes ``None`` so a caller-supplied
    ``whichtoinvert`` can fill that position without conflicting with manifest
    metadata.
    """
    if isinstance(value, (str, Path)):
        if get_ext(value) != _TRANSFORM_MANIFEST_EXT:
            raise ValueError(
                "`transformlist` as a single path must be a .json manifest; "
                f"pass a list to supply transform files. Got {value!s}."
            )
        paths, manifest_flags = _load_transform_manifest(value)
        resolved_flags: list[bool | None] = list(manifest_flags)
        return paths, resolved_flags

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("`transformlist` must be a non-empty list.")

        paths: list[str] = []
        invert_flags: list[bool | None] = []

        for item in value:
            if (
                isinstance(item, (str, Path))
                and get_ext(item) == _TRANSFORM_MANIFEST_EXT
            ):
                item_paths, item_flags = _load_transform_manifest(item)
                paths.extend(item_paths)
                invert_flags.extend(item_flags)
                continue

            transform_path = Path(item)
            if not transform_path.is_file():
                raise FileNotFoundError(f"Transform file not found: {transform_path}")

            transform_ext = get_ext(transform_path).lower()
            if transform_ext not in _ALLOWED_TRANSFORM_EXT:
                raise ValueError(
                    f"Unknown transform extension {transform_ext!r}: "
                    f"{transform_path!s}."
                )

            paths.append(str(transform_path))
            invert_flags.append(None)

        return paths, invert_flags

    raise ValueError(
        "`transformlist` must be a list of transform paths and/or .json "
        "manifests, or a path to a single .json manifest, "
        f"got {type(value).__name__}."
    )


def _save_transformlist(
    key: str,
    value: Any,
    output_path: Path,
) -> ArtifactPathRecord:
    """Persist an ordered ANTs transform chain as a JSON manifest plus payload files.

    ``output_path`` is the canonical artifact path and must end with ``.json``.
    Transform files are copied next to the manifest. The manifest stores only the
    relative payload paths and the ANTs inversion flags needed to reload the chain.
    """
    if get_ext(output_path) != _TRANSFORM_MANIFEST_EXT:
        raise ValueError(
            f"Output path for {key!r} must be a transform manifest ending "
            f"with .json, got {output_path!s}."
        )

    if key not in {"fwdtransforms", "invtransforms"}:
        raise ValueError(
            f"`key` must be 'fwdtransforms' or 'invtransforms', got {key!r}."
        )

    if not isinstance(value, list) or not value:
        raise ValueError(f"{key!r} must be a non-empty list of transform paths.")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_stem = output_path.name[: -len(_TRANSFORM_MANIFEST_EXT)]

    written: list[Path] = []
    entries: list[dict[str, Any]] = []

    for index, transform_path in enumerate(value):
        source = Path(transform_path)
        if not source.is_file():
            raise FileNotFoundError(f"Transform file not found: {source}")

        source_ext = get_ext(source).lower()
        if source_ext not in _ALLOWED_TRANSFORM_EXT:
            raise ValueError(
                f"Unknown transform extension {source_ext!r} at index {index} "
                f"for {key!r}: {source!s}."
            )

        # Use the manifest stem as the naming anchor to avoid collisions between
        # forward/inverse chains saved in the same directory.
        dest = output_path.parent / f"{manifest_stem}_{index:02d}{source_ext}"
        shutil.copy2(source, dest)

        written.append(dest.resolve())

        entries.append(
            {
                "path": dest.name,  # relative to the manifest directory
                "invert": key == "invtransforms" and source_ext == ".mat",
            }
        )

    manifest = {
        "format": "ants_transform_chain",
        "transforms": entries,
    }
    write_json(manifest, output_path)

    return ArtifactPathRecord(
        path=output_path,
        payload=tuple(written),
    )


class ANTsRegistration(PipelineStage):
    """Register a moving image to a fixed target with ANTs.

    Wraps :func:`~niiflow.preproc.functional.image.registration.ants_registration`.
    The moving ``image`` is warped into the space of the fixed ``target`` (ANTs
    moving/fixed terminology).

    **Parameters** (``params``):

    * ``image`` — moving :class:`ants.core.ANTsImage` or path to load.
    * ``target`` — fixed :class:`ants.core.ANTsImage` or path to load.
    * Additional kwargs are forwarded to ``ants_registration`` (e.g. ``mode``,
      ``apply_forward``, ``interpolation``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — warped moving image when ``apply_forward`` is true.
    * ``fwdtransforms`` — ordered list of forward transform file paths from ANTs.
    * ``invtransforms`` — ordered list of inverse transform file paths from ANTs.

    When ``apply_forward`` is false, only the transform lists are returned.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    * ``fwdtransforms`` / ``invtransforms`` — JSON manifest path. Each chain is
      copied beside the manifest in ANTs order (see :func:`_save_transformlist`).
    """

    REQUIRED_PARAMS = frozenset({"image", "target"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "target":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read target image from {value}") from e

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        result = ants_registration(**params)
        if isinstance(result, dict):
            return result
        out_image, transforms = result
        return {"out_image": out_image, **transforms}

    def save_output(
        self, key: str, value: Any, output_path: Path
    ) -> Path | ArtifactPathRecord:
        """Persist one registration output to ``output_path``.

        **``out_image``** — NIfTI write via :func:`~niiflow.preproc.utils.file.ants_image_write`.

        **``fwdtransforms`` / ``invtransforms``** — compound artifact via
        :func:`_save_transformlist` (``.json`` manifest plus copied transforms).
        """
        if key == "out_image":
            return ants_image_write(value, output_path)

        if key in ("fwdtransforms", "invtransforms"):
            return _save_transformlist(key, value, output_path)

        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class ANTsApplyTransforms(PipelineStage):
    """Apply an ANTs transform chain to map an image into a target space.

    Wraps
    :func:`~niiflow.preproc.functional.image.registration.ants_apply_transforms`.
    The moving ``image`` is resampled onto the grid of the fixed ``target`` using
    an ordered ``transformlist``.

    **Parameters** (``params``):

    * ``image`` — moving :class:`ants.core.ANTsImage` or path to load.
    * ``target`` — fixed :class:`ants.core.ANTsImage` or path to load.
    * ``transformlist`` — an ordered ANTs chain, in any of these forms:

      * a path to one ``.json`` manifest written by :func:`_save_transformlist`;
      * a list of transform file paths;
      * a list whose entries are manifests, transform files, or both.

      :func:`_save_transformlist` persists each registration chain as one
      manifest. The copied transform files are payload beside that manifest,
      not separate outputs, so composing saved chains means listing the
      manifests. List entries are expanded in order; each manifest contributes
      its stored transforms at that position. The flattened sequence is passed
      to ANTs as ``transformlist``.

      Inversion flags come from manifests, while individual transform files
      contribute no inversion constraint. An optional ``whichtoinvert`` is
      merged element-wise with those flags after manifest expansion: unspecified
      values are filled from the other source, matching booleans are accepted,
      and conflicting booleans raise an error. Thus ``whichtoinvert`` describes
      the flattened transform list, not the unexpanded input list.

    * Additional kwargs are forwarded to ``ants_apply_transforms``; for example
      ``interpolation``, ``whichtoinvert``, and ``defaultvalue``.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — the transformed image on ``target``'s grid.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``). No other artifact is
      written.
    """

    REQUIRED_PARAMS = frozenset({"image", "target", "transformlist"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "target":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read target image from {value}") from e

        if key == "transformlist":
            return _resolve_transformlist(value)

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        transformlist, stored_flags = params.pop("transformlist")
        requested_flags = params.pop("whichtoinvert", None)

        if requested_flags is None:
            merged_flags = stored_flags
        else:
            if not isinstance(requested_flags, (list, tuple)):
                raise TypeError(
                    "`whichtoinvert` must be a list or tuple containing booleans "
                    f"or None, got {type(requested_flags).__name__}."
                )
            if len(requested_flags) != len(transformlist):
                raise ValueError(
                    "`whichtoinvert` must have one entry per flattened transform: "
                    f"got {len(requested_flags)} flags for "
                    f"{len(transformlist)} transforms."
                )

            merged_flags: list[bool | None] = []
            for index, (stored, requested) in enumerate(
                zip(stored_flags, requested_flags, strict=True)
            ):
                if requested is not None and not isinstance(requested, bool):
                    raise TypeError(
                        f"`whichtoinvert` entry {index} must be a boolean or None, "
                        f"got {type(requested).__name__}."
                    )
                if stored is not None and requested is not None and stored != requested:
                    raise ValueError(
                        f"`whichtoinvert` entry {index} conflicts with the "
                        f"transform manifest: requested {requested}, "
                        f"manifest requires {stored}."
                    )
                merged_flags.append(stored if stored is not None else requested)

        params["whichtoinvert"] = merged_flags

        return {
            "out_image": ants_apply_transforms(
                **params,
                transformlist=transformlist,
            )
        }

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key != "out_image":
            raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}.")

        return ants_image_write(value, output_path)

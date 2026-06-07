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


def _resolve_transformlist(
    value: Any,
) -> list[str] | tuple[list[str], list[bool]]:
    """Materialise ``transformlist`` from paths or an ANTs transform manifest.

    A plain list/tuple of paths is returned as a list of strings.

    A ``.json`` manifest must contain a ``transforms`` list. Each entry must
    contain:
    - ``path``: transform path, relative to the manifest directory or absolute;
    - ``invert``: ANTs ``whichtoinvert`` flag for that transform.

    The order of entries in the manifest is the ANTs application order.
    """
    if isinstance(value, (str, Path)) and get_ext(value) == ".json":
        manifest_path = Path(value)
        manifest = read_json(manifest_path)

        if manifest.get("format") != "ants_transform_chain":
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
                    f"be a dictionary."
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

            invert = entry["invert"]
            if not isinstance(invert, bool):
                raise TypeError(
                    f"Transform manifest {manifest_path!s} entry {index} has "
                    f"non-boolean 'invert': {type(invert).__name__}."
                )

            paths.append(str(transform_path))
            invert_flags.append(invert)

        return paths, invert_flags

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("`transformlist` must be a non-empty list of paths.")

        paths = [str(Path(transform)) for transform in value]
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Transform file not found: {path}")

        return paths

    raise ValueError(
        "`transformlist` must be a list of transform paths or a path to a "
        f".json manifest, got {type(value).__name__}."
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
    if get_ext(output_path) != ".json":
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

    manifest_stem = output_path.name[: -len(".json")]

    written: list[Path] = []
    entries: list[dict[str, Any]] = []

    for index, transform_path in enumerate(value):
        source = Path(transform_path)
        if not source.is_file():
            raise FileNotFoundError(f"Transform file not found: {source}")

        source_ext = get_ext(source).lower()
        if source_ext not in {".mat", ".nii", ".nii.gz", ".h5"}:
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

    **Persistence** (``save_options``):

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
            if get_ext(output_path) not in (".nii.gz", ".nii"):
                raise ValueError(
                    f"Output path for {key!r} must end with .nii.gz or .nii, "
                    f"got {output_path!s}"
                )
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
    * ``transformlist`` — either a list of transform file paths, or a ``.json``
      ANTs transform-chain manifest written by :func:`_save_transformlist`.

      A manifest is a complete transform-chain artifact: it must contain the
      ordered transform paths and explicit ``invert`` flags. When a manifest is
      used, ``whichtoinvert`` must not be supplied separately in ``params``.

      Plain path lists do not carry inversion metadata. For plain lists,
      ``whichtoinvert`` may be supplied explicitly in ``params`` or omitted to
      rely on the underlying ANTs defaults.

    * Additional kwargs are forwarded to ``ants_apply_transforms``; for example
      ``interpolation``, ``whichtoinvert``, and ``defaultvalue``.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — the transformed image on ``target``'s grid.

    **Persistence** (``save_options``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``). No other artifact is
      written.
    """

    REQUIRED_PARAMS = frozenset({"image", "target", "transformlist"})

    def check_params(self, params: dict[str, Any]) -> None:
        if params.get("whichtoinvert") is None:
            return

        transformlist = params["transformlist"]
        if isinstance(transformlist, (str, Path)) and get_ext(transformlist) == ".json":
            raise ValueError(
                "`whichtoinvert` cannot be set in `params` when "
                "`transformlist` points to an ANTs transform-chain manifest. "
                "The manifest is the canonical source of inversion flags."
            )

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
        loaded = params.pop("transformlist")

        if isinstance(loaded, tuple):
            transformlist, whichtoinvert = loaded
            params["whichtoinvert"] = whichtoinvert
        else:
            transformlist = loaded

        return {
            "out_image": ants_apply_transforms(
                **params,
                transformlist=transformlist,
            )
        }

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key != "out_image":
            raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}.")

        if get_ext(output_path) not in (".nii.gz", ".nii"):
            raise ValueError(
                f"Output path for {key!r} must end with .nii.gz or .nii, "
                f"got {output_path!s}."
            )

        ants_image_write(value, output_path)
        return output_path

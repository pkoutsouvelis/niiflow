"""Array MONAI-based transforms."""

from __future__ import annotations

__all__ = [
    "SlidingWindowPatch",
]

import logging
from typing import Literal, Sequence
from typing import cast
import math
import warnings

import torch
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.data.utils import dense_patch_slices

from niiflow.utils.parsers.sequences import ensure_tuple
from niiflow.utils.data.spatial import pad_to_size
from niiflow.utils._validators import validate_type, validate_literal_str

logger = logging.getLogger(__name__)


class SlidingWindowPatch(Transform):
    """
    Extracts sliding window patches from a tensor of shape (C, *spatial_dims),
    returning a tensor of shape (N_patches, C, *patch_size).

    Stride of the sliding window is calculated as either by `overlap`
    or automatically to match `n_patches`.

    If `n_patches` is provided, `overlap` is ignored.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        patch_size: int | Sequence[int],
        overlap: float | Sequence[float] | None = 0.5,
        n_patches: int | Sequence[int] | None = None,
        padding_mode: str = "constant",
        padding_side: Literal["right", "left", "both"] = "both",
        spatial_dims: int = 3,
    ):
        """
        Args:
            patch_size: Target patch size.
            overlap: Overlap between patches.
            n_patches: Number of patches to extract.
            padding_mode: Padding mode; see `torch.nn.functional.pad`.
            padding_side: Side to pad; can be 'right', 'left', or 'both'.
            spatial_dims: Number of spatial dimensions.

        Warnings:
            UserWarning: If both `n_patches` and `overlap` are provided.

        Raises:
            ValueError: If neither `n_patches` nor `overlap` are provided.
            TypeError: If the `patch_size`, `overlap`, `n_patches`, `padding_mode`, 
                `padding_side`, or `spatial_dims` are not of the correct type.
            ValueError: If the `patch_size`, `overlap`, and `n_patches` do not 
                have the expected length.
        """
        super().__init__()
        if n_patches is None and overlap is None:
            raise ValueError(
                f"{self.__class__.__name__} - Expected either `n_patches` "
                f"or `overlap` to be provided, "
                f"got both `n_patches`={n_patches!r} and `overlap`={overlap!r}."
            )

        if n_patches is not None and overlap is not None:
            warnings.warn(
                f"{self.__class__.__name__} - Provided both `n_patches` "
                f"and `overlap`; will use `n_patches`.",
                UserWarning,
            )

        self.patch_size: tuple[int, ...] = ensure_tuple(
            patch_size,
            n=spatial_dims,
            allowed_types=int,
            msg_id=f"{self.__class__.__name__} - patch_size",
        )
        self.overlap: tuple[float, ...] | None = (
            ensure_tuple(
                overlap,
                n=spatial_dims,
                allowed_types=float,
                msg_id=f"{self.__class__.__name__} - overlap",
            )
            if overlap is not None
            else None
        )
        self.n_patches: tuple[int, ...] | None = (
            ensure_tuple(
                n_patches,
                n=spatial_dims,
                allowed_types=int,
                msg_id=f"{self.__class__.__name__} - n_patches",
            )
            if n_patches is not None
            else None
        )

        validate_type(
            padding_mode, str, msg_id=f"{self.__class__.__name__} - padding_mode"
        )
        validate_literal_str(
            padding_side,
            ("right", "left", "both"),
            msg_id=f"{self.__class__.__name__} - padding_side",
        )
        validate_type(
            spatial_dims, int, msg_id=f"{self.__class__.__name__} - spatial_dims"
        )

        self.padding_mode: str = padding_mode
        self.padding_side: Literal["right", "left", "both"] = padding_side
        self.spatial_dims: int = spatial_dims
        self.slices: list[tuple[slice, ...]] = []  # will be populated by __call__()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input tensor of shape (..., C, *spatial_dims).

        Returns:
            Tensor of shape (..., N_patches, C, *patch_size).

        Raises:
            ValueError: If the input tensor has less than `spatial_dims + 1` dimensions.
            Exception: If `pad_to_size` raises an exception, possibly by `torch.nn.functional.pad`
                through the use of an invalid `padding_mode`.
        """
        if img.ndim < 1 + self.spatial_dims:
            msg = (
                f"{self.__class__.__name__} - Expected at least "
                f"{self.spatial_dims + 1}D tensor "
                f"(C, *spatial), got shape {img.shape}"
            )
            raise ValueError(msg)

        C, *spatial = img.shape[-self.spatial_dims - 1 :]

        # minimum target dim to pad
        min_target_dims = [max(S, P) for S, P in zip(spatial, self.patch_size)]

        # Calculate stride
        if self.n_patches is not None:
            stride = []
            target_dims = []
            for S, P, N in zip(min_target_dims, self.patch_size, self.n_patches):
                if N <= 1:
                    st = max(1, S)
                    target_dim = S
                else:
                    st = max(1, math.ceil(max(0, S - P) / (N - 1)))
                    target_dim = (
                        N - 1
                    ) * st + P  # possibly extend padding to force N patches
                stride.append(st)
                target_dims.append(target_dim)
        else:
            stride = tuple(
                max(1, int(round(p * (1.0 - o))))
                for p, o in zip(self.patch_size, cast(tuple[float, ...], self.overlap))
            )
            target_dims = min_target_dims

        # Pad to target dims
        img = pad_to_size(
            img,
            target=target_dims,
            spatial_dims=self.spatial_dims,
            mode=self.padding_mode,
            side=self.padding_side,
        )
        spatial = img.shape[-self.spatial_dims :]  # updated spatial dims after padding

        # Get slices for patches
        self.slices = dense_patch_slices(spatial, self.patch_size, stride)

        prefix = (slice(None),) * (
            img.ndim - self.spatial_dims - 1
        )  # leading dims before C
        patches = [img[prefix + (slice(None),) + slc] for slc in self.slices]

        return torch.stack(
            patches, dim=len(prefix)
        )  # shape: (..., N_patches, C, *patch_size)

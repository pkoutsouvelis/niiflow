"""Utility functions for spatial transformations."""

from __future__ import annotations

__all__ = [
    "pad_to_size",
]

import logging
from typing import Literal, Sequence

import torch
import torch.nn.functional as F


def pad_to_size(
    img: torch.Tensor,
    target: Sequence[int],
    spatial_dims: int,
    mode: str = "constant",
    side: Literal["right", "left", "both"] = "right",
) -> torch.Tensor:
    """
    Pad spatial dims to `target` size.

    Args:
        img: Tensor to pad.
        target: Target size of the image.
        spatial_dims: Number of spatial dimensions.
        mode: Padding mode; see `torch.nn.functional.pad`.
        side: Side to pad; can be 'right', 'left', or 'both'.

    Returns:
        Padded tensor.

    Raises:
        ValueError: If the entry for `side` is invalid.
    """
    if not isinstance(spatial_dims, int):
        raise TypeError(
            f"Expected an integer for `spatial_dims`, "
            f"got {type(spatial_dims).__name__!r}."
        )

    current = img.shape[-spatial_dims:]
    pads: list[int] = []
    for S, T in zip(reversed(current), reversed(target)):
        need = max(T - S, 0)
        if side == "right":
            pads.extend((0, need))  # F.pad expects (left, right) for last dim first
        elif side == "left":
            pads.extend((need, 0))
        elif side == "both":
            l = need // 2
            r = need - l
            pads.extend((l, r))
        else:
            msg = f"{parse_msg_id(msg_id)}Invalid entry for `side`={side!r}"
            raise ValueError(msg)
    if any(pads):
        img = F.pad(img, pad=pads, mode=mode)
    return img

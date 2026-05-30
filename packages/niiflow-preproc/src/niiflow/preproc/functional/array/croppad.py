"""Cropping and padding functions for numpy arrays.

All functions in this module produce fresh arrays; inputs are never modified
in place. Where a function accepts a ``mask``, the same encodings as the
intensity-normalization helpers are accepted: a boolean array or a numeric
array whose unique values are ``0`` and ``1``.

The functions are written to work on N-dimensional arrays, but the typical
use case is 3D medical-image volumes.

Functions that crop or pad to a target report what they did in the same
vocabulary they accept, so the plan can be replayed directly:

* :func:`crop_to_mask` and :func:`center_crop` return ``(result, crop_ranges)``
  where ``crop_ranges`` is a per-axis tuple of ``(start, stop)`` pairs,
  replayable via :func:`crop_to_range`.
* :func:`center_pad` returns ``(result, pad_ranges)`` where ``pad_ranges`` is
  a per-axis tuple of ``(before, after)`` width pairs, replayable via
  :func:`pad_to_range`.
* :func:`crop_to_range` and :func:`pad_to_range` return just the array.
* :func:`bbox_from_mask` returns only a tuple of :class:`slice` objects -- a
  bounding box is idiomatically an index tuple. Convert each slice to
  ``(s.start, s.stop)`` to feed :func:`crop_to_range`.

There is no combined crop-and-pad helper: to reach an arbitrary target shape,
chain :func:`center_crop` then :func:`center_pad`. Each leaves any axis that
already satisfies its part of the target untouched (no crop when the target is
larger than the array, no pad when the target is smaller).
"""

from __future__ import annotations

__all__ = [
    "bbox_from_mask",
    "crop_to_range",
    "crop_to_mask",
    "center_crop",
    "pad_to_range",
    "center_pad",
]

from collections.abc import Sequence

import numpy as np

from .utils import resolve_mask, validate_numeric_array


def _validate_shape_tuple(
    shape: Sequence[int], ndim: int, name: str = "shape"
) -> tuple[int, ...]:
    """Coerce a shape-like sequence to a length-`ndim` tuple of positive ints."""
    try:
        shape_tuple = tuple(int(s) for s in shape)
    except TypeError as exc:
        raise ValueError(
            f"`{name}` must be a sequence of integers, got {type(shape).__name__}"
        ) from exc

    if len(shape_tuple) != ndim:
        raise ValueError(
            f"`{name}` must have length {ndim} to match the array, got "
            f"length {len(shape_tuple)}"
        )
    if any(s <= 0 for s in shape_tuple):
        raise ValueError(
            f"`{name}` must contain only positive integers, got {shape_tuple}"
        )
    return shape_tuple


def _validate_pair_ranges(
    ranges: Sequence[Sequence[int | None]],
    ndim: int,
    name: str,
    *,
    allow_none: bool,
    non_negative: bool,
) -> list[tuple[int | None, int | None]]:
    """Coerce a per-axis range specification into a uniform list of pairs."""
    try:
        ranges_list = list(ranges)
    except TypeError as exc:
        raise ValueError(
            f"`{name}` must be a sequence of (start, stop) pairs, got "
            f"{type(ranges).__name__}"
        ) from exc

    if len(ranges_list) != ndim:
        raise ValueError(
            f"`{name}` must have length {ndim} to match the array, got "
            f"length {len(ranges_list)}"
        )

    cleaned: list[tuple[int | None, int | None]] = []
    for axis, pair in enumerate(ranges_list):
        try:
            pair_tuple = tuple(pair)
        except TypeError as exc:
            raise ValueError(
                f"`{name}[{axis}]` must be a length-2 sequence, got "
                f"{type(pair).__name__}"
            ) from exc
        if len(pair_tuple) != 2:
            raise ValueError(
                f"`{name}[{axis}]` must be a length-2 sequence, got length "
                f"{len(pair_tuple)}"
            )

        coerced: list[int | None] = []
        for value in pair_tuple:
            if value is None:
                if not allow_none:
                    raise ValueError(f"`{name}[{axis}]` may not contain ``None``")
                coerced.append(None)
                continue
            try:
                int_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"`{name}[{axis}]` must contain integers (or ``None``), "
                    f"got {value!r}"
                ) from exc
            if non_negative and int_value < 0:
                raise ValueError(
                    f"`{name}[{axis}]` must be non-negative, got {int_value}"
                )
            coerced.append(int_value)

        cleaned.append((coerced[0], coerced[1]))
    return cleaned


def _resolve_center(
    array_shape: tuple[int, ...], mask: np.ndarray | None
) -> np.ndarray:
    """Return the centering reference, either the array center or a mask bbox center."""
    if mask is None:
        return np.asarray(array_shape, dtype=np.float64) / 2.0

    if not isinstance(mask, np.ndarray):
        raise ValueError(f"`mask` must be a numpy array, got {type(mask).__name__}")
    if mask.shape != array_shape:
        raise ValueError(f"`mask` must have shape {array_shape}, got {mask.shape}")
    bbox = bbox_from_mask(mask)
    return np.array([(sl.start + sl.stop) / 2.0 for sl in bbox], dtype=np.float64)


def _centered_crop_ranges(
    array_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    center: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Per-axis ``(start, stop)`` pairs for a centered crop to `target_shape`.

    Axes where the target is not smaller than the array are left intact as
    ``(0, n)``.
    """
    ranges: list[tuple[int, int]] = []
    for axis, (n, t) in enumerate(zip(array_shape, target_shape)):
        if t >= n:
            ranges.append((0, n))
            continue
        off = int(np.rint(float(center[axis]) - t / 2.0))
        off = max(0, min(off, n - t))
        ranges.append((off, off + t))
    return tuple(ranges)


def _centered_pad_ranges(
    array_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    center: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Per-axis ``(before, after)`` pairs for a centered pad to `target_shape`.

    Axes where the target is not larger than the array are left unpadded as
    ``(0, 0)``.
    """
    ranges: list[tuple[int, int]] = []
    for axis, (n, t) in enumerate(zip(array_shape, target_shape)):
        if t <= n:
            ranges.append((0, 0))
            continue
        off = int(np.rint(float(center[axis]) - t / 2.0))
        off = max(-(t - n), min(off, 0))
        pad_before = -off
        pad_after = (t - n) - pad_before
        ranges.append((pad_before, pad_after))
    return tuple(ranges)


def _resolve_pad_kwargs(mode: str, constant_values) -> dict:
    """Build the keyword arguments forwarded to :func:`numpy.pad`."""
    if mode == "constant":
        return {"mode": "constant", "constant_values": constant_values}
    return {"mode": mode}


# ---------------------------------------------------------------------------
# Bounding box.
# ---------------------------------------------------------------------------


def bbox_from_mask(mask: np.ndarray, pad: int = 0) -> tuple[slice, ...]:
    """Return the axis-aligned bounding box of the non-zero voxels in `mask`.

    The bounding box is returned as a tuple of :class:`slice` objects (one
    per axis), suitable for direct use as a fancy-index into a same-shaped
    array or, after converting each slice to ``(s.start, s.stop)``, as input
    to :func:`crop_to_range`. An optional integer `pad` is added to every
    side, then the result is clipped to the image bounds.

    Args:
        mask:
            A boolean array, or a numeric array containing only ``0`` and
            ``1``.
        pad:
            Number of voxels to expand the bounding box by on every side.
            Must be non-negative. Defaults to ``0``.

    Returns:
        A tuple of length ``mask.ndim`` of slices that index the padded
        bounding box of the masked region.

    Raises:
        ValueError: If `mask` fails validation (see
            :func:`niiflow.preproc.functional.array.utils.resolve_mask`),
            or `pad` is negative.
    """
    bool_mask = resolve_mask(mask, name="mask")
    if pad < 0:
        raise ValueError(f"`pad` must be non-negative, got {pad}")

    nz = np.argwhere(bool_mask)
    mins = np.maximum(nz.min(axis=0) - pad, 0)
    maxs = np.minimum(nz.max(axis=0) + 1 + pad, bool_mask.shape)
    return tuple(slice(int(mi), int(ma)) for mi, ma in zip(mins, maxs))


# ---------------------------------------------------------------------------
# Cropping.
# ---------------------------------------------------------------------------


def crop_to_range(
    array: np.ndarray,
    ranges: Sequence[Sequence[int | None]],
) -> np.ndarray:
    """Crop `array` to per-axis ``(start, stop)`` ranges.

    Each entry of `ranges` is a length-2 sequence ``(start, stop)`` with
    Python-slice semantics: ``start`` is inclusive, ``stop`` is exclusive,
    and either may be ``None`` to mean "from the beginning" or "to the
    end" respectively. Negative indices are not supported.

    Args:
        array:
            Numeric input array.
        ranges:
            A sequence of length ``array.ndim`` of ``(start, stop)``
            pairs.

    Returns:
        A freshly-allocated cropped copy of `array`.

    Raises:
        ValueError: If `array` is not a numeric numpy array, `ranges` has
            the wrong length, a start/stop is out of bounds for its axis,
            or a stop is not strictly greater than its start.
    """
    validate_numeric_array(array)
    pairs = _validate_pair_ranges(
        ranges, array.ndim, name="ranges", allow_none=True, non_negative=True
    )

    slices: list[slice] = []
    for axis, (start, stop) in enumerate(pairs):
        axis_len = array.shape[axis]
        actual_start = 0 if start is None else start
        actual_stop = axis_len if stop is None else stop
        if actual_start > axis_len:
            raise ValueError(
                f"`ranges[{axis}]` start={actual_start} exceeds axis length "
                f"{axis_len}"
            )
        if actual_stop > axis_len:
            raise ValueError(
                f"`ranges[{axis}]` stop={actual_stop} exceeds axis length "
                f"{axis_len}"
            )
        if actual_stop <= actual_start:
            raise ValueError(
                f"`ranges[{axis}]` stop must be greater than start, got "
                f"start={actual_start}, stop={actual_stop}"
            )
        slices.append(slice(actual_start, actual_stop))

    return array[tuple(slices)].copy()


def crop_to_mask(
    array: np.ndarray,
    mask: np.ndarray | None = None,
    pad: int = 0,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Crop `array` to the bounding box of `mask` (or of the array itself).

    If `mask` is provided, the crop is the bounding box of its non-zero
    voxels (optionally grown by `pad` voxels per side, clipped to the
    image bounds). If `mask` is ``None``, the array's own non-zero voxels
    are used to derive the bounding box.

    Args:
        array:
            Numeric input array.
        mask:
            Optional mask whose shape must match `array`. Accepts a
            boolean array or a numeric array containing only ``0`` and
            ``1``. Defaults to ``None``.
        pad:
            Number of voxels to expand the bounding box by on every side,
            clipped to the image bounds. Must be non-negative. Defaults
            to ``0``.

    Returns:
        A pair ``(result, crop_ranges)`` where `result` is a
        freshly-allocated cropped copy of `array` and `crop_ranges` is a
        per-axis tuple of ``(start, stop)`` pairs indexing the input array
        (replayable with :func:`crop_to_range`).

    Raises:
        ValueError: If `array` is not a numeric numpy array, `mask`
            (when given) does not match `array.shape`, the derived or
            given mask is empty, or `pad` is negative.
    """
    validate_numeric_array(array)

    if mask is None:
        derived_mask = array != 0
        if not derived_mask.any():
            raise ValueError(
                "Cannot derive bounding box from `array`: all voxels are zero"
            )
        bbox = bbox_from_mask(derived_mask, pad=pad)
    else:
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"`mask` must be a numpy array, got {type(mask).__name__}")
        if mask.shape != array.shape:
            raise ValueError(f"`mask` must have shape {array.shape}, got {mask.shape}")
        bbox = bbox_from_mask(mask, pad=pad)

    crop_ranges = tuple((int(sl.start), int(sl.stop)) for sl in bbox)
    return array[bbox].copy(), crop_ranges


def center_crop(
    array: np.ndarray,
    shape: Sequence[int],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Crop `array` toward `shape`, centered on the array or on a mask.

    When `mask` is ``None`` the crop is centered on the array's geometric
    center. When `mask` is given, the crop is centered on the center of
    the mask's bounding box; if the requested window would fall outside
    the array, the window is clamped (translated minimally) to remain
    fully inside the array.

    Per axis, cropping is applied only when the target is smaller than the
    array; otherwise that axis is left intact. This makes it safe to chain
    with :func:`center_pad` toward a mixed crop/pad target shape.

    Args:
        array:
            Numeric input array.
        shape:
            Target shape, of length ``array.ndim``. Each entry must be a
            positive integer.
        mask:
            Optional mask whose shape must match `array`. Defaults to
            ``None`` (center on the array).

    Returns:
        A pair ``(result, crop_ranges)`` where `crop_ranges` is a per-axis
        tuple of ``(start, stop)`` pairs indexing the input array
        (replayable with :func:`crop_to_range`).

    Raises:
        ValueError: If `array` is not a numeric numpy array, `shape` is
            malformed, or `mask` (when given) does not match `array.shape`.
    """
    validate_numeric_array(array)
    target_shape = _validate_shape_tuple(shape, array.ndim, name="shape")
    center = _resolve_center(array.shape, mask)
    crop_ranges = _centered_crop_ranges(array.shape, target_shape, center)
    return crop_to_range(array, crop_ranges), crop_ranges


# ---------------------------------------------------------------------------
# Padding.
# ---------------------------------------------------------------------------


def pad_to_range(
    array: np.ndarray,
    ranges: Sequence[Sequence[int]],
    mode: str = "constant",
    constant_values: float | int = 0,
) -> np.ndarray:
    """Pad `array` by per-axis ``(before, after)`` widths.

    The semantics follow :func:`numpy.pad`: each axis is padded with
    ``before`` voxels prepended and ``after`` voxels appended.
    ``constant_values`` is only used when ``mode == "constant"``.

    Args:
        array:
            Numeric input array.
        ranges:
            A sequence of length ``array.ndim`` of ``(before, after)``
            pairs of non-negative integers.
        mode:
            Padding mode, forwarded to :func:`numpy.pad`. Defaults to
            ``"constant"``.
        constant_values:
            Fill value used when ``mode == "constant"``. Defaults to
            ``0``.

    Returns:
        A freshly-allocated padded copy of `array`.

    Raises:
        ValueError: If `array` is not a numeric numpy array, `ranges` is
            malformed, or any width is negative.
    """
    validate_numeric_array(array)
    pairs = _validate_pair_ranges(
        ranges, array.ndim, name="ranges", allow_none=False, non_negative=True
    )

    pad_width: tuple[tuple[int, int], ...] = tuple(
        (int(before), int(after)) for before, after in pairs  # type: ignore[arg-type]
    )
    return np.pad(array, pad_width, **_resolve_pad_kwargs(mode, constant_values))


def center_pad(
    array: np.ndarray,
    shape: Sequence[int],
    mask: np.ndarray | None = None,
    mode: str = "constant",
    constant_values: float | int = 0,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Pad `array` toward `shape`, centered on the array or on a mask.

    When `mask` is ``None`` the array is placed at the geometric center
    of the padded output. When `mask` is given, the array is placed so
    that the mask's bounding-box center coincides with the output
    center; if this would require negative padding (i.e. the requested
    placement would crop the array), the placement is clamped so all of
    the array is preserved.

    Per axis, padding is applied only when the target is larger than the
    array; otherwise that axis is left intact. This makes it safe to chain
    with :func:`center_crop` toward a mixed crop/pad target shape.

    Args:
        array:
            Numeric input array.
        shape:
            Target shape, of length ``array.ndim``. Each entry must be a
            positive integer.
        mask:
            Optional mask whose shape must match `array`. Defaults to
            ``None`` (center the array).
        mode:
            Padding mode, forwarded to :func:`numpy.pad`. Defaults to
            ``"constant"``.
        constant_values:
            Fill value used when ``mode == "constant"``. Defaults to
            ``0``.

    Returns:
        A pair ``(result, pad_ranges)`` where `pad_ranges` is a per-axis
        tuple of ``(before, after)`` width pairs (replayable with
        :func:`pad_to_range`).

    Raises:
        ValueError: If `array` is not a numeric numpy array, `shape` is
            malformed, or `mask` (when given) does not match `array.shape`.
    """
    validate_numeric_array(array)
    target_shape = _validate_shape_tuple(shape, array.ndim, name="shape")
    center = _resolve_center(array.shape, mask)
    pad_ranges = _centered_pad_ranges(array.shape, target_shape, center)
    return (
        pad_to_range(array, pad_ranges, mode=mode, constant_values=constant_values),
        pad_ranges,
    )

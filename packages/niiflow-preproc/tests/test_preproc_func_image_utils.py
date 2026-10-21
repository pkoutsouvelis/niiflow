"""Tests for shared ANTsImage conversion utilities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from niiflow.preproc.functional.image.utils import (
    ants_to_numpy_with_metadata,
    mm_to_voxels,
    numpy_to_ants_with_metadata,
)


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


def test_metadata_contains_only_serializable_geometry(ants_mod) -> None:
    image = ants_mod.from_numpy(
        np.zeros((2, 3, 4), dtype=np.float32),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.5, 1.0, 2.0),
    )

    _, metadata = ants_to_numpy_with_metadata(image)

    assert metadata == {
        "shape": [2, 3, 4],
        "origin": [1.0, 2.0, 3.0],
        "spacing": [0.5, 1.0, 2.0],
        "direction": image.direction.tolist(),
    }
    json.dumps(metadata)


def test_numpy_round_trip_preserves_geometry(ants_mod) -> None:
    direction = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    image = ants_mod.from_numpy(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.5, 1.0, 2.0),
        direction=direction,
    )

    array, metadata = ants_to_numpy_with_metadata(image)
    out = numpy_to_ants_with_metadata(array, metadata)

    np.testing.assert_array_equal(out.numpy(), image.numpy())
    assert out.origin == image.origin
    assert out.spacing == image.spacing
    np.testing.assert_array_equal(out.direction, image.direction)


def test_mm_to_voxels_broadcasts_scalar() -> None:
    assert mm_to_voxels(2.0, (2.0, 1.0, 0.5)) == (1.0, 2.0, 4.0)


def test_mm_to_voxels_converts_tuple_per_axis() -> None:
    assert mm_to_voxels((2.0, 1.0, 1.0), (2.0, 1.0, 0.5)) == (1.0, 1.0, 2.0)


def test_mm_to_voxels_rejects_tuple_length_mismatch() -> None:
    with pytest.raises(ValueError, match="tuple length"):
        mm_to_voxels((1.0, 1.0), (1.0, 1.0, 1.0))


def test_mm_to_voxels_rejects_non_positive_spacing() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        mm_to_voxels(1.0, (1.0, 0.0, 1.0))


def test_mm_to_voxels_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError, match="float or a tuple"):
        mm_to_voxels([1.0, 2.0], (1.0, 1.0))  # type: ignore[arg-type]


def test_numpy_to_ants_rejects_shape_mismatch(ants_mod) -> None:
    image = ants_mod.from_numpy(np.zeros((2, 3), dtype=np.float32))
    _, metadata = ants_to_numpy_with_metadata(image)

    with pytest.raises(ValueError, match="shape recorded in metadata"):
        numpy_to_ants_with_metadata(np.zeros((3, 2), dtype=np.float32), metadata)

"""Tests for :mod:`niiflow.preproc.functional.image.metadata`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.metadata import sync_ants_metadata


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


def _volume(ants_mod, data, *, origin, spacing, direction=None):
    kwargs: dict = {"origin": origin, "spacing": spacing}
    if direction is not None:
        kwargs["direction"] = direction
    return ants_mod.from_numpy(np.asarray(data, dtype=np.float64), **kwargs)


class TestSyncAntsMetadata:
    def test_copies_reference_metadata_without_changing_voxels(self, ants_mod) -> None:
        voxels = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
        origin = (10.0, 20.0, 30.0)
        spacing = (0.5, 0.6, 0.7)
        direction = np.diag([-1.0, 1.0, 1.0])
        image = _volume(
            ants_mod,
            voxels,
            origin=origin,
            spacing=spacing,
            direction=direction,
        )
        reference = _volume(
            ants_mod,
            np.zeros_like(voxels),
            origin=origin,
            spacing=spacing,
            direction=direction,
        )

        out = sync_ants_metadata(image, reference)

        np.testing.assert_array_equal(out.numpy(), voxels)
        assert out.origin == reference.origin
        assert out.spacing == reference.spacing
        np.testing.assert_array_equal(out.direction, reference.direction)
        assert out is not image
        np.testing.assert_array_equal(image.numpy(), voxels)

    def test_syncs_when_differences_are_within_tolerance(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((3, 3, 3)),
            origin=(1.0 + 1e-8, 2.0, 3.0),
            spacing=(1.0, 1.0 + 1e-8, 1.0),
            direction=np.eye(3) + 1e-8,
        )
        reference = _volume(
            ants_mod,
            np.zeros((3, 3, 3)),
            origin=(1.0, 2.0, 3.0),
            spacing=(1.0, 1.0, 1.0),
            direction=np.eye(3),
        )

        out = sync_ants_metadata(image, reference)

        assert out.origin == reference.origin
        assert out.spacing == reference.spacing
        np.testing.assert_array_equal(out.direction, reference.direction)
        np.testing.assert_array_equal(out.numpy(), np.ones((3, 3, 3)))

    def test_rtol_allows_scaled_origin_drift(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((2, 2, 2)),
            origin=(100.005, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        reference = _volume(
            ants_mod,
            np.zeros((2, 2, 2)),
            origin=(100.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        out = sync_ants_metadata(image, reference, origin_atol=0.0, rtol=1e-4)
        assert out.origin == reference.origin

    def test_raises_when_origin_exceeds_tolerance(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((2, 2, 2)),
            origin=(1.1, 2.0, 3.0),
            spacing=(1.0, 1.0, 1.0),
        )
        reference = _volume(
            ants_mod,
            np.zeros((2, 2, 2)),
            origin=(1.0, 2.0, 3.0),
            spacing=(1.0, 1.0, 1.0),
        )
        with pytest.raises(ValueError, match="origin"):
            sync_ants_metadata(image, reference)

    def test_raises_when_spacing_exceeds_tolerance(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((2, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.2, 1.0, 1.0),
        )
        reference = _volume(
            ants_mod,
            np.zeros((2, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        with pytest.raises(ValueError, match="spacing"):
            sync_ants_metadata(image, reference)

    def test_raises_when_direction_exceeds_tolerance(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((2, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=np.diag([-1.0, 1.0, 1.0]),
        )
        reference = _volume(
            ants_mod,
            np.zeros((2, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=np.eye(3),
        )
        with pytest.raises(ValueError, match="direction"):
            sync_ants_metadata(image, reference)

    def test_custom_tolerance_allows_larger_origin_drift(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((2, 2, 2)),
            origin=(1.1, 2.0, 3.0),
            spacing=(1.0, 1.0, 1.0),
        )
        reference = _volume(
            ants_mod,
            np.zeros((2, 2, 2)),
            origin=(1.0, 2.0, 3.0),
            spacing=(1.0, 1.0, 1.0),
        )
        out = sync_ants_metadata(image, reference, origin_atol=0.2)
        assert out.origin == reference.origin

    def test_raises_on_shape_mismatch(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((2, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        reference = _volume(
            ants_mod,
            np.zeros((3, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        with pytest.raises(ValueError, match="same shape"):
            sync_ants_metadata(image, reference)

    def test_rejects_non_image_inputs(self, ants_mod) -> None:
        reference = _volume(
            ants_mod,
            np.zeros((2, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        with pytest.raises(ValueError, match="ANTsImage"):
            sync_ants_metadata(np.zeros((2, 2, 2)), reference)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ANTsImage"):
            sync_ants_metadata(reference, np.zeros((2, 2, 2)))  # type: ignore[arg-type]

    def test_rejects_invalid_tolerances(self, ants_mod) -> None:
        image = _volume(
            ants_mod,
            np.ones((2, 2, 2)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        with pytest.raises(TypeError, match="atol"):
            sync_ants_metadata(image, image, origin_atol="tight")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="atol"):
            sync_ants_metadata(image, image, spacing_atol=-1.0)
        with pytest.raises(ValueError, match="rtol"):
            sync_ants_metadata(image, image, rtol=np.inf)

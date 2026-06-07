"""Wrapper tests for :mod:`niiflow.preproc.functional.image.reorientation`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.reorientation import ants_reorient


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.zeros((4, 5, 6), dtype=np.float64))


class TestAntsReorient:
    def test_itk_convention_forwards_orientation_unchanged(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple] = []

        def fake_reorient(image, orientation):
            calls.append((image, orientation))
            return image

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.reorientation.ants.reorient_image2",
            fake_reorient,
        )
        out = ants_reorient(ants_image, "RPI", convention="itk")

        assert calls == [(ants_image, "RPI")]
        assert out is ants_image

    def test_nibabel_convention_converts_before_calling_ants(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple] = []

        def fake_reorient(image, orientation):
            calls.append((image, orientation))
            return image

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.reorientation.ants.reorient_image2",
            fake_reorient,
        )
        out = ants_reorient(ants_image, "RAS", convention="nibabel")

        assert calls == [(ants_image, "RAI")]
        assert out is ants_image

    def test_rejects_non_image_input(self) -> None:
        with pytest.raises(ValueError, match="ANTsImage"):
            ants_reorient(np.zeros((4, 5, 6)), "RPI")  # type: ignore[arg-type]

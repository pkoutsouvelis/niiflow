"""Wrapper tests for :mod:`niiflow.preproc.functional.image.resampling`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.resampling import (
    _RESAMPLE_INTERP_CODES,
    ants_resample,
    ants_resample_to_target,
)


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.zeros((4, 5), dtype=np.float64))


class TestResampleAnts:
    @pytest.mark.parametrize(
        ("interpolation", "interp_type"),
        list(_RESAMPLE_INTERP_CODES.items()),
    )
    def test_forwards_interpolation_as_interp_type_code(
        self,
        ants_image,
        interpolation: str,
        interp_type: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict] = []

        def fake_resample_image(**kwargs):
            calls.append(kwargs)
            return ants_image

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.resampling.resample_image",
            fake_resample_image,
        )
        out = ants_resample(
            ants_image,
            resample_params=(1.0, 1.0),
            use_voxels=True,
            interpolation=interpolation,  # type: ignore[arg-type]
        )

        assert calls == [
            {
                "image": ants_image,
                "resample_params": [1.0, 1.0],
                "use_voxels": True,
                "interp_type": interp_type,
            }
        ]
        assert out is ants_image

    def test_rejects_invalid_interpolation(self, ants_image) -> None:
        with pytest.raises(ValueError, match="Invalid interpolation"):
            ants_resample(ants_image, resample_params=(1.0, 1.0), interpolation="lanczos")  # type: ignore[arg-type]

    def test_rejects_non_image_input(self) -> None:
        with pytest.raises(ValueError, match="ANTsImage"):
            ants_resample(np.zeros((4, 5)), resample_params=(1.0, 1.0))  # type: ignore[arg-type]


class TestResampleToTargetAnts:
    def test_forwards_to_resample_image_to_target(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        calls: list[dict] = []

        def fake_resample_image_to_target(**kwargs):
            calls.append(kwargs)
            return target

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.resampling.resample_image_to_target",
            fake_resample_image_to_target,
        )
        out = ants_resample_to_target(
            ants_image,
            target,
            interpolation="nearestNeighbor",
            verbose=True,
        )

        assert calls == [
            {
                "image": ants_image,
                "target": target,
                "interp_type": "nearestNeighbor",
                "verbose": True,
            }
        ]
        assert out is target

    def test_rejects_reserved_kwargs(self, ants_image) -> None:
        target = ants_image.clone()
        with pytest.raises(TypeError, match="does not accept .* via `kwargs`"):
            ants_resample_to_target(ants_image, target, interp_type="linear")

    def test_rejects_non_image_target(self, ants_image) -> None:
        with pytest.raises(ValueError, match="`target` must be an ANTsImage"):
            ants_resample_to_target(ants_image, np.zeros((4, 5)))  # type: ignore[arg-type]

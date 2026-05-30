"""Wrapper tests for :mod:`niiflow.preproc.functional.image.registration`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.registration import (
    apply_transforms_ants,
    registration_ants,
)


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.zeros((4, 5), dtype=np.float64))


class TestApplyTransformsAnts:
    def test_maps_image_and_target_to_moving_and_fixed(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        calls: list[dict] = []

        def fake_apply_transforms(**kwargs):
            calls.append(kwargs)
            return target

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.apply_transforms",
            fake_apply_transforms,
        )
        out = apply_transforms_ants(
            ants_image,
            target,
            interpolation="linear",
            transformlist=["/tmp/fwd.mat"],
        )

        assert calls == [
            {
                "fixed": target,
                "moving": ants_image,
                "interpolator": "linear",
                "transformlist": ["/tmp/fwd.mat"],
            }
        ]
        assert out is target

    def test_rejects_reserved_kwargs(self, ants_image) -> None:
        target = ants_image.clone()
        with pytest.raises(TypeError, match="does not accept .* via `kwargs`"):
            apply_transforms_ants(ants_image, target, fixed=target)


class TestRegistrationAnts:
    def test_apply_forward_false_returns_transform_dict(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        reg_result = {
            "fwdtransforms": ["/tmp/fwd.mat"],
            "invtransforms": ["/tmp/inv.mat"],
        }
        registration_calls: list[dict] = []
        apply_calls: list[dict] = []

        def fake_registration(**kwargs):
            registration_calls.append(kwargs)
            return reg_result

        def fake_apply_transforms(**kwargs):
            apply_calls.append(kwargs)
            raise AssertionError("apply_transforms should not be called")

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.registration",
            fake_registration,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.apply_transforms",
            fake_apply_transforms,
        )
        out = registration_ants(
            ants_image,
            target,
            mode="Rigid",
            apply_forward=False,
            random_seed=0,
        )

        assert registration_calls == [
            {
                "fixed": target,
                "moving": ants_image,
                "type_of_transform": "Rigid",
                "random_seed": 0,
            }
        ]
        assert apply_calls == []
        assert out == {
            "fwdtransforms": ["/tmp/fwd.mat"],
            "invtransforms": ["/tmp/inv.mat"],
        }

    def test_apply_forward_true_warps_with_forward_transforms(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        warped = target.clone()
        reg_result = {
            "fwdtransforms": ["/tmp/fwd.mat", "/tmp/fwd.nii.gz"],
            "invtransforms": ["/tmp/inv.nii.gz", "/tmp/inv.mat"],
        }
        apply_calls: list[dict] = []

        def fake_registration(**kwargs):
            return reg_result

        def fake_apply_transforms(**kwargs):
            apply_calls.append(kwargs)
            return warped

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.registration",
            fake_registration,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.apply_transforms",
            fake_apply_transforms,
        )
        result = registration_ants(
            ants_image,
            target,
            mode="Affine",
            interpolation="nearestNeighbor",
        )
        assert isinstance(result, tuple)
        out_warped, transforms = result
        assert apply_calls == [
            {
                "fixed": target,
                "moving": ants_image,
                "transformlist": ["/tmp/fwd.mat", "/tmp/fwd.nii.gz"],
                "interpolator": "nearestNeighbor",
            }
        ]
        assert out_warped is warped
        assert transforms["fwdtransforms"] == reg_result["fwdtransforms"]
        assert transforms["invtransforms"] == reg_result["invtransforms"]

    def test_rejects_reserved_kwargs(self, ants_image) -> None:
        target = ants_image.clone()
        with pytest.raises(TypeError, match="does not accept .* via `kwargs`"):
            registration_ants(ants_image, target, type_of_transform="SyN")

"""Wrapper tests for :mod:`niiflow.preproc.functional.image.registration`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.registration import (
    ants_apply_transforms,
    ants_registration,
    ants_similarity_metrics,
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
        out = ants_apply_transforms(
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
            ants_apply_transforms(ants_image, target, fixed=target)


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
        out = ants_registration(
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
        result = ants_registration(
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
            ants_registration(ants_image, target, type_of_transform="SyN")


class TestAntsSimilarityMetrics:
    @pytest.fixture
    def mask(self, ants_mod, ants_image):
        return ants_mod.from_numpy(np.ones(ants_image.shape, dtype=np.float64))

    def test_negates_ants_costs_and_masks(
        self, ants_image, mask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        calls: list[dict] = []

        def fake_similarity(fixed, moving, **kwargs):
            calls.append({"fixed": fixed, "moving": moving, **kwargs})
            return {"Correlation": -0.8, "MattesMutualInformation": -1.5}[
                kwargs["metric_type"]
            ]

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.image_similarity",
            fake_similarity,
        )
        scores = ants_similarity_metrics(
            ants_image,
            target,
            mask,
            metrics=("correlation", "mattes_mutual_information"),
        )
        assert scores == {
            "correlation": 0.8,
            "mattes_mutual_information": 1.5,
        }
        assert [c["metric_type"] for c in calls] == [
            "Correlation",
            "MattesMutualInformation",
        ]
        assert all(c["fixed_mask"] is mask and c["moving_mask"] is mask for c in calls)
        assert all(c["fixed"] is target and c["moving"] is ants_image for c in calls)

    def test_mask_optional(self, ants_image, monkeypatch: pytest.MonkeyPatch) -> None:
        target = ants_image.clone()
        calls: list[dict] = []

        def fake_similarity(fixed, moving, **kwargs):
            calls.append(kwargs)
            return -1.0

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.image_similarity",
            fake_similarity,
        )
        scores = ants_similarity_metrics(ants_image, target, metrics=("correlation",))
        assert scores == {"correlation": 1.0}
        assert calls == [
            {
                "metric_type": "Correlation",
                "fixed_mask": None,
                "moving_mask": None,
            }
        ]

    def test_defaults_to_all_supported_metrics(
        self, ants_image, mask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        metric_types: list[str] = []

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.image_similarity",
            lambda *a, **kw: (metric_types.append(kw["metric_type"]) or -1.0),
        )
        scores = ants_similarity_metrics(ants_image, target, mask)
        assert set(scores) == {
            "correlation",
            "mattes_mutual_information",
            "neighborhood_correlation",
        }
        assert metric_types == [
            "Correlation",
            "MattesMutualInformation",
            "ANTSNeighborhoodCorrelation",
        ]
        assert all(score == 1.0 for score in scores.values())

    def test_rejects_unknown_and_empty_metrics(self, ants_image, mask) -> None:
        target = ants_image.clone()
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            ants_similarity_metrics(ants_image, target, mask, metrics=("nope",))
        with pytest.raises(ValueError, match="at least one metric"):
            ants_similarity_metrics(ants_image, target, mask, metrics=())

    def test_rejects_physical_space_mismatch(
        self, ants_mod, ants_image, mask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_mod.from_numpy(np.zeros(ants_image.shape, dtype=np.float64))
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.registration.image_physical_space_consistency",
            lambda *_a, **_k: False,
        )
        with pytest.raises(ValueError, match="same physical space"):
            ants_similarity_metrics(ants_image, target, mask)

    def test_rejects_shape_mismatch(self, ants_mod, ants_image, mask) -> None:
        target = ants_mod.from_numpy(np.zeros((3, 3), dtype=np.float64))
        with pytest.raises(ValueError, match="shape"):
            ants_similarity_metrics(ants_image, target, mask)

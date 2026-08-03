"""Wrapper tests for :mod:`niiflow.preproc.functional.image.skull_stripping`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.skull_stripping import ants_brain_extraction


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.ones((4, 5), dtype=np.float64))


class _MorphResult:
    def __init__(self, mask):
        self._mask = mask

    def iMath_fill_holes(self):
        return self._mask


class TestBrainExtractionAnts:
    def test_rejects_non_image_input(self) -> None:
        with pytest.raises(ValueError, match="ANTsImage"):
            ants_brain_extraction(np.zeros((4, 5)))  # type: ignore[arg-type]

    @pytest.mark.parametrize("modality", ["t1", "t2", "flair"])
    def test_default_modality_thresholds_and_morphs_probability_map(
        self,
        ants_image,
        modality: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bet = object()
        mask = ants_image.clone()
        bet_calls: list[tuple] = []
        threshold_calls: list[tuple] = []
        morphology_calls: list[tuple] = []

        def fake_brain_extraction(image, *, modality, verbose):
            bet_calls.append((image, modality, verbose))
            return bet

        def fake_threshold_image(image, low, high, inval, outval):
            threshold_calls.append((image, low, high, inval, outval))
            return mask

        def fake_morphology(image, op, radius):
            morphology_calls.append((image, op, radius))
            return _MorphResult(mask)

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.brain_extraction",
            fake_brain_extraction,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.threshold_image",
            fake_threshold_image,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.morphology",
            fake_morphology,
        )
        result = ants_brain_extraction(
            ants_image,
            modality=modality,
            verbose=True,
        )
        assert isinstance(result, tuple)
        out_brain, out_mask = result
        assert bet_calls == [(ants_image, modality, True)]
        assert threshold_calls == [(bet, 0.5, 1, 1, 0)]
        assert morphology_calls == [(mask, "close", 6)]
        assert out_mask is mask
        assert out_brain is not None

    def test_t1threetissue_uses_segmentation_label(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        segmentation = object()
        bet = {"segmentation_image": segmentation}
        mask = ants_image.clone()
        threshold_calls: list[tuple] = []
        morphology_calls: list[tuple] = []

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.brain_extraction",
            lambda image, modality, verbose: bet,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.threshold_image",
            lambda image, low, high, inval, outval: threshold_calls.append(
                (image, low, high, inval, outval)
            )
            or mask,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.morphology",
            lambda *args, **kwargs: morphology_calls.append((args, kwargs)),
        )
        out = ants_brain_extraction(
            ants_image, modality="t1threetissue", apply_mask=False
        )

        assert threshold_calls == [(segmentation, 1, 1, 1, 0)]
        assert morphology_calls == []
        assert out is mask

    def test_t1combined_thresholds_combined_labels(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bet = object()
        mask = ants_image.clone()
        threshold_calls: list[tuple] = []
        morphology_calls: list[tuple] = []

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.brain_extraction",
            lambda image, modality, verbose: bet,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.threshold_image",
            lambda image, low, high, inval, outval: threshold_calls.append(
                (image, low, high, inval, outval)
            )
            or mask,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.morphology",
            lambda *args, **kwargs: morphology_calls.append((args, kwargs)),
        )
        out = ants_brain_extraction(ants_image, modality="t1combined", apply_mask=False)

        assert threshold_calls == [(bet, 2, 3, 1, 0)]
        assert morphology_calls == []
        assert out is mask

    def test_apply_mask_false_returns_mask_only(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bet = object()
        mask = ants_image.clone()

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.brain_extraction",
            lambda image, modality, verbose: bet,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.threshold_image",
            lambda *args, **kwargs: mask,
        )
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.skull_stripping.morphology",
            lambda image, op, radius: _MorphResult(mask),
        )
        out = ants_brain_extraction(ants_image, apply_mask=False)

        assert out is mask

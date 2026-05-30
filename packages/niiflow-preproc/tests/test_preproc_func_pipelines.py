"""Wrapper tests for :mod:`niiflow.preproc.functional.image.pipelines`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.pipelines import preprocessing_pipeline_ants


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.zeros((4, 5), dtype=np.float64))


def test_forwards_kwargs_to_preprocess_brain_image(
    ants_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    mask = ants_image.clone()

    def fake_preprocess_brain_image(**kwargs):
        calls.append(kwargs)
        return {
            "preprocessed_image": ants_image,
            "brain_mask": mask,
        }

    monkeypatch.setattr(
        "niiflow.preproc.functional.image.pipelines.preprocess_brain_image",
        fake_preprocess_brain_image,
    )
    result = preprocessing_pipeline_ants(
        image=ants_image,
        brain_extraction_modality="t1",
        do_denoising=False,
        verbose=False,
    )
    assert isinstance(result, tuple)
    preprocessed, metadata = result
    assert calls == [
        {
            "image": ants_image,
            "brain_extraction_modality": "t1",
            "do_denoising": False,
            "verbose": False,
        }
    ]
    assert preprocessed is ants_image
    assert metadata == {"brain_mask": mask}


def test_return_metadata_false_returns_image_only(
    ants_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "niiflow.preproc.functional.image.pipelines.preprocess_brain_image",
        lambda **kwargs: {"preprocessed_image": ants_image, "brain_mask": kwargs["image"]},
    )
    out = preprocessing_pipeline_ants(image=ants_image, return_metadata=False)

    assert out is ants_image


def test_return_metadata_true_splits_preprocessed_image_from_auxiliary_outputs(
    ants_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    mask = ants_image.clone()
    transforms = {"fwdtransforms": ["/tmp/fwd.mat"], "invtransforms": ["/tmp/inv.mat"]}

    monkeypatch.setattr(
        "niiflow.preproc.functional.image.pipelines.preprocess_brain_image",
        lambda **kwargs: {
            "preprocessed_image": ants_image,
            "brain_mask": mask,
            "template_transforms": transforms,
        },
    )
    result = preprocessing_pipeline_ants(image=ants_image)

    assert isinstance(result, tuple)
    preprocessed, metadata = result
    assert preprocessed is ants_image
    assert metadata == {
        "brain_mask": mask,
        "template_transforms": transforms,
    }
    assert "preprocessed_image" not in metadata


def test_raises_when_preprocessed_image_key_missing(
    ants_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "niiflow.preproc.functional.image.pipelines.preprocess_brain_image",
        lambda **kwargs: {"brain_mask": ants_image},
    )
    with pytest.raises(KeyError, match='preprocess_brain_image must return a "preprocessed_image" key'):
        preprocessing_pipeline_ants(image=ants_image)

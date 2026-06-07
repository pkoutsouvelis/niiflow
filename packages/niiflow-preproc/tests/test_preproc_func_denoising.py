"""Wrapper tests for :mod:`niiflow.preproc.functional.image.denoising`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.denoising import ants_denoise


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.zeros((4, 5), dtype=np.float64))


def test_forwards_kwargs_to_denoise_image(
    ants_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    mask = ants_image.clone()
    calls: list[dict] = []

    def fake_denoise_image(**kwargs):
        calls.append(kwargs)
        return ants_image

    monkeypatch.setattr(
        "niiflow.preproc.functional.image.denoising.denoise_image",
        fake_denoise_image,
    )
    out = ants_denoise(
        image=ants_image,
        mask=mask,
        noise_model="Rician",
        shrink_factor=2,
    )

    assert calls == [
        {
            "image": ants_image,
            "mask": mask,
            "noise_model": "Rician",
            "shrink_factor": 2,
        }
    ]
    assert out is ants_image

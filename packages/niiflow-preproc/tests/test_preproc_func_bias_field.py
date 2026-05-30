"""Wrapper tests for :mod:`niiflow.preproc.functional.image.bias_field`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.image.bias_field import bias_field_correction_ants


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.zeros((4, 5), dtype=np.float64))


def test_forwards_kwargs_to_n4_bias_field_correction(
    ants_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_n4_bias_field_correction(**kwargs):
        calls.append(kwargs)
        return ants_image

    monkeypatch.setattr(
        "niiflow.preproc.functional.image.bias_field.n4_bias_field_correction",
        fake_n4_bias_field_correction,
    )
    out = bias_field_correction_ants(
        image=ants_image,
        shrink_factor=2,
        return_bias_field=False,
    )

    assert calls == [
        {
            "image": ants_image,
            "shrink_factor": 2,
            "return_bias_field": False,
        }
    ]
    assert out is ants_image

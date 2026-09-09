"""Tests for :mod:`niiflow.preproc.utils.misc`."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.utils.misc import numeric_mismatch


class TestNumericMismatch:
    def test_returns_none_when_values_are_close(self) -> None:
        assert (
            numeric_mismatch(
                (1.0, 2.0),
                (1.0 + 1e-8, 2.0),
                atol=1e-5,
                name="origin",
            )
            is None
        )

    def test_rtol_allows_scaled_difference(self) -> None:
        assert (
            numeric_mismatch(
                100.005,
                100.0,
                atol=0.0,
                rtol=1e-4,
                name="origin",
            )
            is None
        )

    def test_rtol_zero_rejects_scaled_difference(self) -> None:
        message = numeric_mismatch(
            100.005,
            100.0,
            atol=0.0,
            rtol=0.0,
            name="origin",
        )
        assert message is not None
        assert "origin" in message

    def test_reports_magnitude_mismatch_with_labels(self) -> None:
        message = numeric_mismatch(
            (1.1, 2.0),
            (1.0, 2.0),
            atol=1e-5,
            name="origin",
            actual_label="image",
            expected_label="reference",
        )
        assert message is not None
        assert "origin" in message
        assert "atol=" in message
        assert "rtol=" in message
        assert "image=" in message
        assert "reference=" in message

    def test_reports_shape_mismatch(self) -> None:
        message = numeric_mismatch(
            (1.0, 2.0),
            (1.0, 2.0, 3.0),
            atol=1e-5,
            name="origin",
        )
        assert message is not None
        assert "shape" in message
        assert "actual=" in message
        assert "expected=" in message

    def test_rejects_non_numeric_atol(self) -> None:
        with pytest.raises(TypeError, match="atol"):
            numeric_mismatch(1.0, 1.0, atol="tight", name="origin")  # type: ignore[arg-type]

    def test_rejects_boolean_atol(self) -> None:
        with pytest.raises(TypeError, match="atol"):
            numeric_mismatch(1.0, 1.0, atol=True, name="origin")  # type: ignore[arg-type]

    def test_rejects_negative_rtol(self) -> None:
        with pytest.raises(ValueError, match="rtol"):
            numeric_mismatch(1.0, 1.0, atol=1e-5, rtol=-0.1, name="origin")

    def test_rejects_non_finite_atol(self) -> None:
        with pytest.raises(ValueError, match="atol"):
            numeric_mismatch(1.0, 1.0, atol=np.inf, name="origin")

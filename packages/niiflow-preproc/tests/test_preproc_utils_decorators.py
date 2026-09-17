"""Tests for :mod:`niiflow.preproc.utils.decorators`."""

from __future__ import annotations

import warnings

import pytest

from niiflow.preproc.utils.decorators import deprecate


class TestDeprecate:
    def test_warns_with_generated_message(self) -> None:
        @deprecate(remove_in="1.0.0", alternative="new_fn")
        def old_fn(value: int) -> int:
            return value + 1

        with pytest.warns(
            DeprecationWarning,
            match=(
                r"`old_fn` is deprecated and will be removed in v1.0.0. "
                r"Use `new_fn` instead."
            ),
        ):
            assert old_fn(1) == 2

    def test_custom_message_replaces_generated_text(self) -> None:
        @deprecate("stop using this")
        def old_fn() -> str:
            return "ok"

        with pytest.warns(DeprecationWarning, match="^stop using this$"):
            assert old_fn() == "ok"

    def test_does_not_warn_until_called(self) -> None:
        @deprecate(remove_in="1.0.0")
        def old_fn() -> None:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert old_fn.__name__ == "old_fn"

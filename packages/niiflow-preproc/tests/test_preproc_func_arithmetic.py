"""Tests for array and image :mod:`...functional.*.arithmetic` helpers."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.array.arithmetic import (
    pointwise_arithmetic as array_pointwise_arithmetic,
)


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


class TestArrayPointwiseArithmetic:
    def test_scalar_mul_and_add(self) -> None:
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = array_pointwise_arithmetic(field, {"mul": 2.0}, {"add": -1.0})
        np.testing.assert_allclose(out, np.array([1.0, 3.0, 5.0]))

    def test_scalar_sub_and_div(self) -> None:
        field = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        out = array_pointwise_arithmetic(field, {"sub": 4.0}, {"div": 2.0})
        np.testing.assert_allclose(out, np.array([3.0, 8.0, 13.0]))

    def test_array_operands(self) -> None:
        field = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        mul = np.array([[2.0, 0.5], [1.0, 3.0]], dtype=np.float64)
        add = np.array([[1.0, -1.0], [0.0, 2.0]], dtype=np.float64)
        out = array_pointwise_arithmetic(field, {"mul": mul}, {"add": add})
        np.testing.assert_allclose(
            out, np.array([[3.0, 0.0], [3.0, 14.0]], dtype=np.float64)
        )

    def test_repeated_operations_are_applied_sequentially(self) -> None:
        field = np.array([2.0, 4.0], dtype=np.float64)
        out = array_pointwise_arithmetic(
            field,
            {"mul": 2.0},
            {"mul": 3.0},
            {"add": 1.0},
            {"add": 5.0},
        )
        np.testing.assert_allclose(out, np.array([18.0, 30.0]))

    def test_operation_order_matters(self) -> None:
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mul_then_add = array_pointwise_arithmetic(field, {"mul": 2.0}, {"add": 1.0})
        add_then_mul = array_pointwise_arithmetic(field, {"add": 1.0}, {"mul": 2.0})
        np.testing.assert_allclose(mul_then_add, np.array([3.0, 5.0, 7.0]))
        np.testing.assert_allclose(add_then_mul, np.array([4.0, 6.0, 8.0]))
        assert not np.allclose(mul_then_add, add_then_mul)

    def test_example_smooth_mask_blend(self) -> None:
        field = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        smooth_mask = np.array([0.0, 1.0, 0.5], dtype=np.float64)
        out = array_pointwise_arithmetic(
            field,
            {"sub": 1},
            {"mul": smooth_mask},
            {"add": 1},
        )
        np.testing.assert_allclose(out, np.array([1.0, 1.0, 1.5]))

    def test_broadcasting(self) -> None:
        field = np.ones((2, 3), dtype=np.float64)
        col = np.array([[2.0], [3.0]], dtype=np.float64)
        row = np.array([1.0, 10.0, 100.0], dtype=np.float64)
        out = array_pointwise_arithmetic(field, {"mul": col}, {"add": row})
        np.testing.assert_allclose(
            out, np.array([[3.0, 12.0, 102.0], [4.0, 13.0, 103.0]])
        )

    def test_does_not_modify_input(self) -> None:
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        original = field.copy()
        _ = array_pointwise_arithmetic(field, {"mul": 2.0}, {"add": 1.0})
        np.testing.assert_array_equal(field, original)

    def test_preserves_numpy_dtype_promotion(self) -> None:
        field = np.array([1, 2, 3], dtype=np.int32)
        out = array_pointwise_arithmetic(field, {"mul": 2.5})
        assert out.dtype == np.result_type(field, 2.5)
        np.testing.assert_allclose(out, np.array([2.5, 5.0, 7.5]))

    def test_rejects_no_operations(self) -> None:
        field = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one operation"):
            array_pointwise_arithmetic(field)

    def test_rejects_non_mapping_operation(self) -> None:
        field = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="must be a mapping"):
            array_pointwise_arithmetic(field, ("mul", 2.0))  # type: ignore[arg-type]

    def test_rejects_empty_operation_mapping(self) -> None:
        field = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="exactly one key"):
            array_pointwise_arithmetic(field, {})

    def test_rejects_multi_key_operation_mapping(self) -> None:
        field = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="exactly one key"):
            array_pointwise_arithmetic(field, {"mul": 2.0, "add": 1.0})

    def test_rejects_unsupported_operation_key(self) -> None:
        field = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="unsupported operation"):
            array_pointwise_arithmetic(field, {"pow": 2.0})

    def test_rejects_non_array_input(self) -> None:
        with pytest.raises(ValueError, match="numpy array"):
            array_pointwise_arithmetic(
                [1.0, 2.0], {"mul": 2.0}  # type: ignore[arg-type]
            )

    def test_wraps_broadcast_mismatch(self) -> None:
        field = np.ones((2, 3), dtype=np.float64)
        bad = np.ones((4,), dtype=np.float64)
        with pytest.raises(
            ValueError,
            match=(
                r"'mul' operation at index 1 failed: could not broadcast "
                r"running array of shape \(2, 3\) with operand of shape \(4,\)"
            ),
        ):
            array_pointwise_arithmetic(field, {"add": 1.0}, {"mul": bad})


class TestImagePointwiseArithmetic:
    def test_scalar_ops_preserve_geometry(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.arithmetic import (
            pointwise_arithmetic as image_pointwise_arithmetic,
        )

        data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
        image = ants_mod.from_numpy(
            data,
            origin=(1.0, 2.0, 3.0),
            spacing=(0.5, 1.0, 1.5),
        )
        out = image_pointwise_arithmetic(image, {"mul": 2.0}, {"sub": 1.0})
        assert out.origin == image.origin
        assert out.spacing == image.spacing
        np.testing.assert_array_equal(out.direction, image.direction)
        np.testing.assert_allclose(out.numpy(), np.array([[[1.0, 3.0], [5.0, 7.0]]]))

    def test_image_operands(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.arithmetic import (
            pointwise_arithmetic as image_pointwise_arithmetic,
        )

        field = ants_mod.from_numpy(
            np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float64),
            origin=(4.0, 5.0, 6.0),
            spacing=(1.0, 1.0, 2.0),
        )
        mask = ants_mod.from_numpy(
            np.array([[[0.0, 1.0], [0.5, 1.0]]], dtype=np.float64),
            origin=(4.0, 5.0, 6.0),
            spacing=(1.0, 1.0, 2.0),
        )
        out = image_pointwise_arithmetic(
            field,
            {"sub": 1},
            {"mul": mask},
            {"add": 1},
        )
        assert out.origin == field.origin
        assert out.spacing == field.spacing
        np.testing.assert_allclose(out.numpy(), np.array([[[1.0, 1.0], [1.5, 3.0]]]))

    def test_rejects_image_operand_metadata_mismatch(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.arithmetic import (
            pointwise_arithmetic as image_pointwise_arithmetic,
        )

        field = ants_mod.from_numpy(
            np.ones((2, 2, 2), dtype=np.float64),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        other = ants_mod.from_numpy(
            np.ones((2, 2, 2), dtype=np.float64),
            origin=(1.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        with pytest.raises(
            ValueError,
            match=r"ANTsImage operand for 'mul' at operation index 0.*voxel grid",
        ):
            image_pointwise_arithmetic(field, {"mul": other})

    def test_rejects_non_ants_input(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.arithmetic import (
            pointwise_arithmetic as image_pointwise_arithmetic,
        )

        with pytest.raises(ValueError, match="ANTsImage"):
            image_pointwise_arithmetic(
                np.ones((2, 2), dtype=np.float64), {"mul": 2.0}  # type: ignore[arg-type]
            )

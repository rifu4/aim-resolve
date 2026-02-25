"""Tests for aim_resolve.model.util — utility functions."""

import numpy as np
import pytest

from aim_resolve.model.util import (
    check_type,
    extend_shape,
    flatten_list,
    is_val,
    to_shape,
)

# ---------- check_type ----------


class TestCheckType:
    def test_valid_single_type(self):
        check_type(42, int)  # should not raise

    def test_valid_multiple_types(self):
        check_type(42, (int, float))

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            check_type("hello", int)

    def test_nested_iterable(self):
        check_type((1, 2, 3), tuple, int)  # tuple of ints

    def test_nested_invalid_raises(self):
        with pytest.raises(TypeError):
            check_type((1, "x"), tuple, int)


# ---------- flatten_list ----------


class TestFlattenList:
    def test_flat(self):
        assert flatten_list([1, 2, 3]) == [1, 2, 3]

    def test_nested(self):
        assert flatten_list([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]

    def test_empty(self):
        assert flatten_list([]) == []

    def test_strings_not_expanded(self):
        assert flatten_list(["ab", "cd"]) == ["ab", "cd"]


# ---------- to_shape ----------


class TestToShape:
    def test_scalar_broadcast(self):
        arr = to_shape(5.0, (3,), "float64")
        np.testing.assert_array_equal(arr, [5.0, 5.0, 5.0])
        assert arr.shape == (3,)

    def test_list_reshape(self):
        arr = to_shape([1, 2, 3, 4], (2, 2), "int64")
        assert arr.shape == (2, 2)
        assert arr.dtype == np.int64

    def test_single_value(self):
        arr = to_shape(7, (), "float64")
        assert arr.shape == ()
        assert float(arr) == 7.0


# ---------- is_val ----------


class TestIsVal:
    def test_nonzero(self):
        assert is_val(np.array([0.0, 1.0])) is True

    def test_all_zero(self):
        assert is_val(np.array([0.0, 0.0])) is False

    def test_with_nan(self):
        assert is_val(np.array([np.nan, 1.0])) is True

    def test_all_nan(self):
        assert is_val(np.array([np.nan, np.nan])) is False


# ---------- extend_shape ----------


class TestExtendShape:
    def test_single_freq_single_copy(self):
        assert extend_shape(1, np.ones(1), (32, 32)) == (32, 32)

    def test_multi_freq(self):
        assert extend_shape(1, np.ones(4), (32, 32)) == (4, 32, 32)

    def test_multi_copies(self):
        assert extend_shape(3, np.ones(1), (32, 32)) == (3, 32, 32)

    def test_multi_freq_and_copies(self):
        assert extend_shape(3, np.ones(4), (32, 32)) == (3, 4, 32, 32)

    def test_offset_mode(self):
        # offset=True uses (1,) instead of (len(freq),)
        assert extend_shape(1, np.ones(4), (32, 32), offset=True) == (1, 32, 32)

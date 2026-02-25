"""Tests for aim_resolve.plot.util — rows_and_cols and to_shape."""

import numpy as np

from aim_resolve.plot.util import rows_and_cols, to_shape


class TestRowsAndCols:
    def test_square_4(self):
        r, c = rows_and_cols(4)
        assert r * c >= 4
        assert r == 2 and c == 2

    def test_square_9(self):
        r, c = rows_and_cols(9)
        assert r == 3 and c == 3

    def test_with_rows(self):
        r, c = rows_and_cols(6, rows=2)
        assert r == 2 and c == 3

    def test_with_cols(self):
        r, c = rows_and_cols(6, cols=3)
        assert r == 2 and c == 3

    def test_single(self):
        r, c = rows_and_cols(1)
        assert r == 1 and c == 1

    def test_rows_capped_at_nums(self):
        r, c = rows_and_cols(2, rows=5)
        assert r == 2


class TestToShape:
    def test_scalar_to_grid(self):
        arr = to_shape(42, (2, 3))
        assert arr.shape == (2, 3)

    def test_list_to_grid(self):
        arr = to_shape([1, 2, 3, 4], None, 2, 2)
        assert arr.shape[:2] == (2, 2)

    def test_return_nums(self):
        arr, nums = to_shape([1, 2, 3], None, 1, 3, return_nums=True)
        assert nums == 3

    def test_transpose(self):
        arr = to_shape([1, 2, 3, 4], (2, 2), transpose=True)
        assert arr.shape[:2] == (2, 2)

    def test_2d_input(self):
        data = np.ones((4, 8, 8))
        arr = to_shape(data, (2, 2))
        assert arr.shape == (2, 2, 8, 8)

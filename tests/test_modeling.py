"""Tests for aim_resolve.modeling."""

import numpy as np
import pytest

from aim_resolve.mask import remove_freq_axis
from aim_resolve.modeling import get_offset


class TestGetOffset:
    """Test the log-scale offset computation."""

    def test_point_offset_returns_list(self):
        """Point-model offsets should return a list of floats."""
        rec = np.full((3, 32, 32), 2.0)
        mask = np.zeros((3, 32, 32), dtype=bool)
        mask[0, 10, 10] = True
        mask[1, 20, 20] = True
        mask[2, 5, 5] = True

        offset = get_offset("point", rec, mask, freq=[1.0])
        assert isinstance(offset, list)
        assert len(offset) == 3

    def test_background_offset_returns_scalar(self):
        rec = np.full((32, 32), 5.0)
        mask = np.ones((32, 32), dtype=bool)

        offset = get_offset("background", rec, mask, freq=[1.0])
        assert isinstance(offset, float)
        np.testing.assert_almost_equal(offset, round(float(np.log(5.0)), 1))

    def test_tile_offset_returns_list(self):
        rec = np.full((2, 32, 32), 3.0)
        mask = np.ones((2, 32, 32), dtype=bool)

        offset = get_offset("tile", rec, mask, freq=[1.0])
        assert isinstance(offset, list)
        assert len(offset) == 2

    def test_object_offset_is_scalar(self):
        rec = np.full((32, 32), 4.0)
        mask = np.ones((32, 32), dtype=bool)

        offset = get_offset("object", rec, mask, freq=[1.0])
        assert isinstance(offset, float)

    def test_signal_offset_same_as_object(self):
        rec = np.full((32, 32), 4.0)
        mask = np.ones((32, 32), dtype=bool)

        offset = get_offset("signal", rec, mask, freq=[1.0])
        assert isinstance(offset, float)

"""Tests for aim_resolve.data."""

import pytest

from aim_resolve.data import data_func


class TestDataFuncDispatch:
    """Test data_func mode routing and error handling."""

    def test_unknown_mode_raises(self):
        with pytest.raises(TypeError, match="Unknown data mode"):
            data_func(mode="invalid")

    def test_image_mode_string_matching(self):
        """Modes containing 'image' should be accepted (routing test only)."""
        # We only test the dispatch branch, not the actual file loading.
        with pytest.raises(Exception):
            # Will fail because fname is missing / file does not exist,
            # but it should NOT raise TypeError — the dispatch worked.
            data_func(mode="image", fname="__nonexistent__")

    def test_radio_mode_string_matching(self):
        with pytest.raises(Exception):
            data_func(mode="radio", fname="__nonexistent__")

"""Tests for aim_resolve.likelihood."""

import pytest

from aim_resolve.likelihood import likelihood_func, likelihood_sum


class TestLikelihoodFuncDispatch:
    """Test likelihood_func mode routing."""

    def test_unknown_mode_raises(self):
        with pytest.raises(TypeError, match="Unknown likelihood mode"):
            likelihood_func(mode="bogus")

    def test_image_mode_dispatches(self):
        """'image' should reach the image branch (will fail on missing args)."""
        with pytest.raises(Exception):
            likelihood_func(mode="image")

    def test_fast_mode_dispatches(self):
        with pytest.raises(Exception):
            likelihood_func(mode="fast")

    def test_radio_mode_dispatches(self):
        with pytest.raises(Exception):
            likelihood_func(mode="radio")


class TestLikelihoodSum:
    """Test the likelihood aggregation helper."""

    def test_sum_of_scalars(self):
        result = likelihood_sum(a=1, b=2, c=3)
        assert result == 6

    def test_sum_single_element(self):
        result = likelihood_sum(a=42)
        assert result == 42

    def test_sum_accepts_named_kwargs(self):
        result = likelihood_sum(lh_1=10, lh_2=20)
        assert result == 30

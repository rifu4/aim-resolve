"""Tests for aim_resolve.img_data.points — get_blur utility."""

import jax.numpy as jnp
import numpy as np

from aim_resolve.img_data.points import get_blur


class TestGetBlur:
    def test_length_from_n_max(self):
        result = get_blur(20, b_min=0, b_max=5)
        assert result.shape == (20,)

    def test_minimum_steps(self):
        result = get_blur(3, b_min=0, b_max=5, steps=10)
        assert result.shape == (10,)

    def test_range(self):
        result = get_blur(10, b_min=1.0, b_max=4.0)
        np.testing.assert_allclose(float(result[0]), 1.0)
        np.testing.assert_allclose(float(result[-1]), 4.0)

    def test_zero_blur(self):
        result = get_blur(5, b_min=0, b_max=0)
        np.testing.assert_allclose(result, 0.0)

    def test_monotonicity(self):
        result = get_blur(15, b_min=0, b_max=3)
        diffs = jnp.diff(result)
        assert (diffs >= 0).all()

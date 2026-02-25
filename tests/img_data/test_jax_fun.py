"""Tests for aim_resolve.img_data.jax_fun — JAX image transformations."""

import jax.numpy as jnp
import numpy as np
import pytest

from aim_resolve.img_data.jax_fun import (
    flip_data,
    gaussian_filter2d,
    gaussian_kernel2d,
    rotate_data,
)


# ---------------------------------------------------------------------------
# gaussian_kernel2d
# ---------------------------------------------------------------------------

class TestGaussianKernel2D:
    def test_shape(self):
        k = gaussian_kernel2d(1.0, 3)
        assert k.shape == (7, 7)

    def test_centre_is_max(self):
        k = gaussian_kernel2d(1.0, 4)
        assert k[4, 4] == k.max()

    def test_symmetric(self):
        k = gaussian_kernel2d(2.0, 5)
        np.testing.assert_allclose(k, k[::-1, :], atol=1e-12)
        np.testing.assert_allclose(k, k[:, ::-1], atol=1e-12)

    def test_non_negative(self):
        k = gaussian_kernel2d(1.5, 3)
        assert float(k.min()) >= 0.0


# ---------------------------------------------------------------------------
# gaussian_filter2d
# ---------------------------------------------------------------------------

class TestGaussianFilter2D:
    def test_zero_sigma_identity(self):
        x = jnp.arange(25, dtype=float).reshape(5, 5)
        result = gaussian_filter2d(x, sigma=0.0)
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_output_shape(self):
        x = jnp.ones((16, 16))
        result = gaussian_filter2d(x, sigma=1.0, radius=3)
        assert result.shape == x.shape

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(0)
        x = jnp.array(rng.random((32, 32)))
        result = gaussian_filter2d(x, sigma=2.0, radius=5)
        assert float(result.var()) <= float(x.var())


# ---------------------------------------------------------------------------
# rotate_data
# ---------------------------------------------------------------------------

class TestRotateData:
    def test_k0_identity(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = rotate_data(m, k=0)
        np.testing.assert_array_equal(result, m)

    def test_k4_identity(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = rotate_data(m, k=4)
        np.testing.assert_array_equal(result, m)

    def test_k1_matches_numpy_rot90(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = rotate_data(m, k=1)
        expected = jnp.rot90(m, k=1)
        np.testing.assert_array_equal(result, expected)

    def test_k2(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = rotate_data(m, k=2)
        expected = jnp.rot90(m, k=2)
        np.testing.assert_array_equal(result, expected)

    def test_k3(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = rotate_data(m, k=3)
        expected = jnp.rot90(m, k=3)
        np.testing.assert_array_equal(result, expected)

    def test_shape_preserved(self):
        m = jnp.ones((8, 8))
        for k in range(4):
            assert rotate_data(m, k=k).shape == (8, 8)


# ---------------------------------------------------------------------------
# flip_data
# ---------------------------------------------------------------------------

class TestFlipData:
    def test_axis0_identity(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = flip_data(m, axis=0)
        np.testing.assert_array_equal(result, m)

    def test_axis1_flip_rows(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = flip_data(m, axis=1)
        expected = jnp.flip(m, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_axis2_flip_cols(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = flip_data(m, axis=2)
        expected = jnp.flip(m, axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_axis3_flip_both(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = flip_data(m, axis=3)
        expected = jnp.flip(m, axis=(0, 1))
        np.testing.assert_array_equal(result, expected)

    def test_axis4_wraps_to_identity(self):
        m = jnp.arange(9).reshape(3, 3).astype(float)
        result = flip_data(m, axis=4)
        np.testing.assert_array_equal(result, m)

    def test_shape_preserved(self):
        m = jnp.ones((8, 8))
        for a in range(4):
            assert flip_data(m, axis=a).shape == (8, 8)

"""Tests for aim_resolve.model.map — downsample and upsample."""

import jax.numpy as jnp
import numpy as np
import pytest

from aim_resolve.model.map import downsample, upsample


class TestDownsample:
    def test_basic_2d(self):
        arr = jnp.ones((8, 8))
        result = downsample(arr, 2)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result, 1.0)

    def test_averages(self):
        arr = jnp.arange(16).reshape(4, 4).astype(float)
        result = downsample(arr, 2)
        assert result.shape == (2, 2)
        # Top-left 2x2 block: mean of 0,1,4,5 = 2.5
        np.testing.assert_allclose(result[0, 0], 2.5)

    def test_batched(self):
        arr = jnp.ones((3, 8, 8))
        result = downsample(arr, 4)
        assert result.shape == (3, 2, 2)


class TestUpsample:
    def test_basic_2d(self):
        arr = jnp.ones((4, 4))
        result = upsample(arr, 2)
        assert result.shape == (8, 8)
        np.testing.assert_allclose(result, 1.0)

    def test_repeats_values(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = upsample(arr, 2)
        assert result.shape == (4, 4)
        assert result[0, 0] == 1.0
        assert result[0, 2] == 2.0
        assert result[2, 0] == 3.0

    def test_batched(self):
        arr = jnp.ones((3, 2, 2))
        result = upsample(arr, 3)
        assert result.shape == (3, 6, 6)


class TestDownsampleUpsampleRoundtrip:
    def test_constant_roundtrip(self):
        arr = jnp.ones((8, 8)) * 5.0
        result = upsample(downsample(arr, 2), 2)
        np.testing.assert_allclose(result, 5.0)
        assert result.shape == (8, 8)

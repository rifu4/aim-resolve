"""Tests for aim_resolve.fast_resolve.convolve — FFT convolution helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from aim_resolve.fast_resolve.convolve import (
    build_fft_kernel,
    build_padder,
    build_slicer,
    build_split_kernel,
    downsample,
    fft_convolve,
    fft_convolve_2d,
    split_fft_convolve,
    upsample,
)

# ---------------------------------------------------------------------------
# downsample / upsample
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_shape_2d(self):
        arr = np.ones((16, 16))
        result = downsample(arr, 2)
        assert result.shape == (8, 8)

    def test_shape_3d(self):
        arr = np.ones((3, 16, 16))
        result = downsample(arr, 4)
        assert result.shape == (3, 4, 4)

    def test_constant_array_value_preserved(self):
        arr = np.full((8, 8), 5.0)
        result = downsample(arr, 2)
        np.testing.assert_allclose(result, 5.0)

    def test_averaging(self):
        arr = np.zeros((4, 4))
        arr[0, 0] = 4.0
        result = downsample(arr, 2)
        assert result.shape == (2, 2)
        assert result[0, 0] == 1.0  # 4/4
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert result[1, 1] == 0.0

    def test_non_divisible_raises(self):
        arr = np.ones((7, 7))
        with pytest.raises(AssertionError):
            downsample(arr, 2)


class TestUpsample:
    def test_shape_2d(self):
        arr = np.ones((4, 4))
        result = upsample(arr, 3)
        assert result.shape == (12, 12)

    def test_shape_3d(self):
        arr = np.ones((2, 4, 4))
        result = upsample(arr, 2)
        assert result.shape == (2, 8, 8)

    def test_values_repeated(self):
        arr = np.arange(4).reshape(2, 2).astype(float)
        result = upsample(arr, 2)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[0, 2] == 1.0
        assert result[0, 3] == 1.0
        assert result[2, 0] == 2.0

    def test_roundtrip_constant(self):
        """Upsample then downsample a constant array should be identity."""
        arr = np.full((4, 4), 3.0)
        result = downsample(upsample(arr, 2), 2)
        np.testing.assert_allclose(result, arr)


# ---------------------------------------------------------------------------
# build_fft_kernel
# ---------------------------------------------------------------------------


class TestBuildFFTKernel:
    def test_output_shape(self):
        kernel = np.random.default_rng(0).random((2, 32, 32))
        result = build_fft_kernel(kernel, kernel.shape)
        assert result.shape == kernel.shape

    def test_dvol_scaling(self):
        kernel = np.ones((1, 8, 8))
        k1 = build_fft_kernel(kernel, kernel.shape, dvol=1.0)
        k2 = build_fft_kernel(kernel, kernel.shape, dvol=2.0)
        np.testing.assert_allclose(np.abs(k2), 2.0 * np.abs(k1))


# ---------------------------------------------------------------------------
# build_padder / build_slicer
# ---------------------------------------------------------------------------


class TestBuildPadder:
    def test_padding_shape(self):
        padder = build_padder((1, 8, 8), (1, 16, 16))
        x = jnp.ones((1, 8, 8))
        result = padder(x)
        assert result.shape == (1, 16, 16)

    def test_padding_values(self):
        padder = build_padder((4, 4), (8, 8))
        x = jnp.ones((4, 4))
        result = padder(x)
        assert float(result[0, 0]) == 1.0
        assert float(result[7, 7]) == 0.0


class TestBuildSlicer:
    def test_slice_shape(self):
        slicer = build_slicer((2, 2), (1, 4, 4))
        x = jnp.ones((1, 8, 8))
        result = slicer(x)
        assert result.shape == (1, 4, 4)

    def test_slice_values(self):
        x = jnp.arange(64).reshape(1, 8, 8).astype(float)
        slicer = build_slicer((1, 1), (1, 4, 4))
        result = slicer(x)
        expected = x[0:1, 1:5, 1:5]
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# fft_convolve_2d / fft_convolve
# ---------------------------------------------------------------------------


class TestFFTConvolve:
    def test_identity_kernel(self):
        """Convolving with a delta should return approximately the input."""
        x = jnp.ones((8, 8))
        kernel = jnp.zeros((8, 8))
        kernel = kernel.at[0, 0].set(1.0)
        result = fft_convolve_2d(x, jnp.fft.fftn(kernel))
        np.testing.assert_allclose(result, x, atol=1e-12)

    def test_batch_convolve_shape(self):
        x = jnp.ones((3, 8, 8))
        kernel = jnp.zeros((3, 8, 8))
        kernel = kernel.at[:, 0, 0].set(1.0)
        fft_k = jnp.fft.fftn(kernel, axes=(-2, -1))
        result = fft_convolve(x, fft_k)
        assert result.shape == (3, 8, 8)

    def test_batch_convolve_values(self):
        """Each batch element convolved with delta should be identity."""
        rng = np.random.default_rng(42)
        x = jnp.array(rng.random((2, 16, 16)))
        kernel = jnp.zeros((2, 16, 16))
        kernel = kernel.at[:, 0, 0].set(1.0)
        fft_k = jnp.fft.fftn(kernel, axes=(-2, -1))
        result = fft_convolve(x, fft_k)
        np.testing.assert_allclose(result, x, atol=1e-12)


# ---------------------------------------------------------------------------
# build_split_kernel
# ---------------------------------------------------------------------------


class TestBuildSplitKernel:
    def test_output_types(self):
        kernel = np.random.default_rng(1).random((1, 32, 32))
        k_high, k_low = build_split_kernel(kernel, (16, 16), size=8, factor=2)
        assert isinstance(k_high, jnp.ndarray)
        assert isinstance(k_low, jnp.ndarray)

    def test_high_kernel_shape(self):
        kernel = np.random.default_rng(1).random((1, 32, 32))
        k_high, _ = build_split_kernel(kernel, (16, 16), size=8, factor=2)
        assert k_high.shape == (1, 24, 24)

    def test_low_kernel_shape(self):
        kernel = np.random.default_rng(1).random((1, 32, 32))
        _, k_low = build_split_kernel(kernel, (16, 16), size=8, factor=2)
        assert k_low.shape == (1, 16, 16)

    def test_2d_kernel_promoted(self):
        """A 2D kernel (no freq axis) should be promoted to 3D internally."""
        kernel = np.random.default_rng(1).random((32, 32))
        k_high, k_low = build_split_kernel(kernel, (16, 16), size=8, factor=2)
        assert k_high.ndim == 3
        assert k_low.ndim == 3


# ---------------------------------------------------------------------------
# split_fft_convolve (integration-like test)
# ---------------------------------------------------------------------------


class TestSplitFFTConvolve:
    def test_output_shape_3d(self):
        """Output should match the input spatial shape."""
        shape = (16, 16)
        kernel = np.random.default_rng(0).random((1, 32, 32))
        k_high, k_low = build_split_kernel(kernel, shape, size=8, factor=2)

        shape_high = (1,) + shape
        padder_high = build_padder(shape_high, k_high.shape)
        slicer_high = build_slicer((4, 4), shape_high)

        shape_low = (1,) + tuple(s // 2 for s in shape)
        padder_low = build_padder(shape_low, k_low.shape)
        slicer_low = build_slicer((3, 3), shape_low)

        x = jnp.ones((1, 16, 16))
        result = split_fft_convolve(
            x,
            k_high,
            k_low,
            padder_high,
            padder_low,
            slicer_high,
            slicer_low,
            factor=2,
        )
        assert result.shape == (16, 16)  # squeezed

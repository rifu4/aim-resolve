"""Tests for aim_resolve.train.dataset — pure utility functions.

The Dataset / TensorDataset classes depend on torch DataLoader and
ImageDataGenerator which require complex setups.  We test the standalone
helper functions that operate on plain JAX / numpy arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aim_resolve.train.dataset import (
    rotate_array,
    flip_array,
    build_facet_array,
    merge_facet_array,
    add_coordinates,
    split_data,
    transform_data,
    TensorDataset,
)


# ---------------------------------------------------------------------------
# rotate_array
# ---------------------------------------------------------------------------


class TestRotateArray:
    """Test 90-degree rotation helper."""

    def test_identity(self):
        arr = jnp.arange(16).reshape(4, 4).astype(float)
        result = rotate_array(arr, 0, axes=(0, 1))
        np.testing.assert_array_equal(result, arr)

    def test_four_rotations_roundtrip(self):
        arr = jnp.arange(16).reshape(4, 4).astype(float)
        rotated = arr
        for _ in range(4):
            rotated = rotate_array(rotated, 1, axes=(0, 1))
        np.testing.assert_array_equal(rotated, arr)

    def test_single_90deg(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = rotate_array(arr, 1, axes=(0, 1))
        expected = jnp.rot90(arr, 1, axes=(0, 1))
        np.testing.assert_array_equal(result, expected)

    def test_mod4(self):
        arr = jnp.ones((3, 3))
        r1 = rotate_array(arr, 1, axes=(0, 1))
        r5 = rotate_array(arr, 5, axes=(0, 1))
        np.testing.assert_array_equal(r1, r5)


# ---------------------------------------------------------------------------
# flip_array
# ---------------------------------------------------------------------------


class TestFlipArray:
    """Test conditional flipping helper."""

    def test_no_flip(self):
        arr = jnp.arange(24).reshape(2, 3, 4).astype(float)
        result = flip_array(arr, axis=0)
        np.testing.assert_array_equal(result, arr)

    def test_flip_axis1(self):
        arr = jnp.arange(24).reshape(2, 3, 4).astype(float)
        result = flip_array(arr, axis=1)
        expected = jnp.flip(arr, axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_flip_axis2(self):
        arr = jnp.arange(24).reshape(2, 3, 4).astype(float)
        result = flip_array(arr, axis=2)
        expected = jnp.flip(arr, axis=2)
        np.testing.assert_array_equal(result, expected)

    def test_mod3(self):
        arr = jnp.ones((2, 3, 4))
        f0 = flip_array(arr, axis=0)
        f3 = flip_array(arr, axis=3)
        np.testing.assert_array_equal(f0, f3)


# ---------------------------------------------------------------------------
# build_facet_array / merge_facet_array
# ---------------------------------------------------------------------------


class TestFacetArrays:
    """Test faceting (splitting) and merging of 4-d arrays."""

    def test_build_facet_shape(self):
        arr = np.arange(4 * 1 * 8 * 8).reshape(4, 1, 8, 8).astype(float)
        faceted = build_facet_array(arr, factor=2)
        # n * factor^2, l, h/factor, w/factor
        assert faceted.shape == (4 * 4, 1, 4, 4)

    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        arr = rng.random((2, 1, 16, 16))
        factor = 4
        faceted = build_facet_array(arr, factor)
        merged = merge_facet_array(faceted, factor)
        np.testing.assert_allclose(merged, arr)

    def test_non_4d_raises(self):
        with pytest.raises(ValueError, match="4-dimensional"):
            build_facet_array(np.zeros((3, 3)), 2)

    def test_merge_non_4d_raises(self):
        with pytest.raises(ValueError, match="4-dimensional"):
            merge_facet_array(np.zeros((3, 3)), 2)


# ---------------------------------------------------------------------------
# add_coordinates
# ---------------------------------------------------------------------------


class TestAddCoordinates:
    """Test coordinate channel concatenation."""

    def test_channels_increase(self):
        images = np.random.default_rng(0).random((3, 1, 8, 8))
        labels = np.random.default_rng(0).random((3, 1, 8, 8))
        coos = [np.random.default_rng(0).random((8, 8)),
                np.random.default_rng(1).random((8, 8))]
        result_images, result_labels = add_coordinates((images, labels), coos)
        # 1 original channel + 2 coordinate channels
        assert result_images.shape == (3, 3, 8, 8)
        # labels unchanged
        np.testing.assert_array_equal(result_labels, labels)

    def test_coordinates_broadcast(self):
        images = np.ones((2, 1, 4, 4))
        labels = np.zeros((2, 1, 4, 4))
        coos = [np.full((4, 4), 0.5)]
        result_images, _ = add_coordinates((images, labels), coos)
        assert result_images.shape == (2, 2, 4, 4)
        np.testing.assert_allclose(result_images[:, 1], 0.5)


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------


class TestSplitData:
    """Test random train/valid splitting."""

    def test_default_split(self):
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100).reshape(100, 1)
        train, valid = split_data((x, y), split=0.8)
        assert train[0].shape[0] == 80
        assert valid[0].shape[0] == 20

    def test_all_elements_present(self):
        x = np.arange(10)
        train, valid = split_data((x,), split=0.5)
        all_vals = np.sort(np.concatenate([train[0], valid[0]]))
        np.testing.assert_array_equal(all_vals, x)


# ---------------------------------------------------------------------------
# transform_data
# ---------------------------------------------------------------------------


class TestTransformData:
    """Test the transform_data pipeline."""

    def test_normalize_and_standardize_raises(self):
        data = (np.ones((2, 1, 4, 4)), np.zeros((2, 1, 4, 4)))
        with pytest.raises(ValueError, match="normalize and standardize"):
            transform_data(data, normalize=True, standardize=True)

    def test_basic_transform(self):
        rng = np.random.default_rng(0)
        images = rng.random((4, 1, 8, 8)).astype(np.float32) + 0.1
        labels = np.zeros((4, 1, 8, 8), dtype=np.float32)
        result_images, result_labels = transform_data(
            (images, labels),
            log=True,
            normalize=True,
            rotate=False,
            flip=False,
        )
        # normalized images should be in [0, 1]
        assert np.all(result_images >= -1e-6)
        assert np.all(result_images <= 1.0 + 1e-6)

    def test_faceting(self):
        rng = np.random.default_rng(0)
        images = rng.random((2, 1, 16, 16)).astype(np.float32) + 0.1
        labels = np.zeros((2, 1, 16, 16), dtype=np.float32)
        result_images, result_labels = transform_data(
            (images, labels),
            log=False,
            normalize=False,
            rotate=False,
            flip=False,
            facet_size=8,
        )
        # factor = 16 // 8 = 2, so n * factor^2 = 2 * 4 = 8
        assert result_images.shape == (8, 1, 8, 8)


# ---------------------------------------------------------------------------
# TensorDataset
# ---------------------------------------------------------------------------


class TestTensorDataset:
    """Test the minimal TensorDataset wrapper."""

    def test_len(self):
        x = np.random.default_rng(0).random((10, 1, 4, 4))
        y = np.zeros((10, 1, 4, 4))
        ds = TensorDataset((x, y))
        assert len(ds) == 10

    def test_getitem(self):
        x = np.random.default_rng(0).random((5, 1, 4, 4))
        y = np.ones((5, 1, 4, 4))
        ds = TensorDataset((x, y))
        item = ds[2]
        assert "x" in item and "y" in item
        np.testing.assert_array_equal(item["x"], x[2])
        np.testing.assert_array_equal(item["y"], y[2])

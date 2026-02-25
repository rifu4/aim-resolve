"""Tests for aim_resolve.img_data.data — ImageData container."""

import numpy as np
import os
import pytest

from aim_resolve.model.grid import SignalGrid
from aim_resolve.img_data.data import ImageData


@pytest.fixture
def grid():
    return SignalGrid.build(space=(32, 32))


@pytest.fixture
def img_data(grid):
    val = np.random.default_rng(0).random(grid.shape)
    return ImageData(val, grid, prefix="test")


class TestImageData:
    def test_constructor(self, grid):
        val = np.ones(grid.shape)
        data = ImageData(val, grid, prefix="my_img")
        assert data.prefix == "my_img"
        assert data.val.shape == grid.shape
        assert data.noisy_val is None

    def test_repr(self, img_data):
        r = repr(img_data)
        assert "ImageData" in r
        assert "test" in r

    def test_add_noise(self, img_data):
        img_data.add_noise(key=42, max_std=0.01)
        assert img_data.noisy_val is not None
        assert img_data.noisy_val.shape == img_data.val.shape
        # noisy_val should differ from val
        assert not np.allclose(img_data.noisy_val, img_data.val)

    def test_add_noise_deterministic(self, grid):
        val = np.ones(grid.shape) * 10.0
        d1 = ImageData(val.copy(), grid)
        d2 = ImageData(val.copy(), grid)
        d1.add_noise(key=7, max_std=0.01)
        d2.add_noise(key=7, max_std=0.01)
        np.testing.assert_array_equal(d1.noisy_val, d2.noisy_val)

    def test_save_load_roundtrip(self, img_data, tmp_path):
        fname = "roundtrip_test"
        img_data.save(fname, odir=str(tmp_path))
        loaded = ImageData.load(fname, odir=str(tmp_path))
        np.testing.assert_allclose(loaded.val, img_data.val)
        assert loaded.prefix == img_data.prefix
        assert loaded.grid.shape == img_data.grid.shape

    def test_save_adds_extension(self, img_data, tmp_path):
        img_data.save("no_ext", odir=str(tmp_path))
        assert os.path.isfile(os.path.join(str(tmp_path), "no_ext.pkl"))

    def test_maps_default_zeros(self, grid):
        val = np.ones(grid.shape)
        data = ImageData(val, grid)
        np.testing.assert_array_equal(data.maps, np.zeros_like(val))

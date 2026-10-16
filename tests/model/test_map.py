"""Tests for aim_resolve.model.map — resampling and placement between grids."""

import jax.numpy as jnp
import numpy as np

import pytest

from aim_resolve.model.grid import PointGrid, SignalGrid
from aim_resolve.model.map import downsample, map_signal, upsample


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


def _mapped_coords(in_grid, out_grid, coords):
    """Map ``coords`` (copies, 2, x, y) of the input pixels onto ``out_grid``;
    return the mapped coordinates and the coverage count per output pixel."""
    f = map_signal(in_grid, out_grid)
    ones = np.ones(coords.shape[:-3] + (1,) + coords.shape[-2:])
    return np.asarray(f(coords)), np.asarray(f(ones))[0]


class TestSignalPlacement:
    """Every input pixel must land on the output pixel with the same coordinates."""

    @pytest.mark.parametrize("factor", [1, 2, 3])
    def test_cutout_lands_on_matching_coords(self, factor):
        out = SignalGrid.build(space=(64, 64), factor=factor)
        cut = SignalGrid.build(space=(12, 8), center=(-9, 13), factor=factor)
        mapped, cov = _mapped_coords(cut, out, np.asarray(cut.coords))
        assert cov.sum() == np.prod(cut.shape)
        np.testing.assert_allclose(mapped[:, cov == 1], np.asarray(out.coords)[:, cov == 1])

    def test_tiles_land_on_matching_coords(self):
        out = SignalGrid.build(space=(64, 64), factor=3)
        tiles = SignalGrid.build(space=(8, 8), center=((-20, 5), (17, -11), (0, 24)), factor=3, n_copies=3)
        mapped, cov = _mapped_coords(tiles, out, np.asarray(tiles.coords))
        assert cov.sum() == 3 * np.prod(tiles.shape)
        np.testing.assert_allclose(mapped[:, cov == 1], np.asarray(out.coords)[:, cov == 1])

    def test_half_pixel_offset_raises(self):
        out = SignalGrid.build(space=(64, 64))
        cut = SignalGrid.build(space=(11, 8))  # odd size difference -> half-pixel offset
        with pytest.raises(ValueError, match="pixel-aligned"):
            map_signal(cut, out)

    def test_non_integer_factor_ratio_raises(self):
        out = SignalGrid.build(space=(16, 16), factor=2)
        cut = SignalGrid.build(space=(4, 4), factor=3)
        with pytest.raises(ValueError, match="integer multiples"):
            map_signal(cut, out)(np.ones(cut.shape))


class TestPointPlacement:
    """Refined point sources must each land on their own sub-pixel."""

    def test_refined_points_land_on_their_pixels(self):
        # coordinates as in a real run, where (coos - lims) * factor is not exact in floating point
        coarse = PointGrid.build(coordinates=[[-457.5, 103.5], [-311.5, -55.5], [12.5, 399.5]], n_copies=3)
        pts = coarse.refine(3)
        out = SignalGrid.build(space=(1024, 1024), factor=3)
        coos = np.asarray(pts.coos).reshape(-1, 2, 1, 1)
        mapped, cov = _mapped_coords(pts, out, coos)
        assert cov.max() == 1 and cov.sum() == pts.n_copies  # no collisions
        np.testing.assert_allclose(mapped[:, cov == 1], np.asarray(out.coords)[:, cov == 1])

    def test_flux_conserved_on_coarser_grid(self):
        pts = PointGrid.build(coordinates=[[0.5, 0.5], [3.5, -2.5]], n_copies=2).refine(2)
        out = SignalGrid.build(space=(16, 16), factor=1)
        res = np.asarray(map_signal(pts, out)(np.ones((pts.n_copies, 1, 1, 1))))
        # unit values on factor-2 sub-pixels become 1/4 per coarse pixel each
        np.testing.assert_allclose(res.sum(), pts.n_copies / 4)

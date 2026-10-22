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


# ---------------------------------------------------------------------------
# Placement of copies without a sequential loop (scatter-add / vmap) against the
# copy-by-copy reference (map_array_2d in a Python loop, as the former fori_loop)
# ---------------------------------------------------------------------------
def _reference(in_array, in_copies, out_copies, in_shape, out_shape, in_start, out_start):
    """Copy-by-copy placement with map_array_2d (zoom = 1)."""
    from aim_resolve.model.map import map_array_2d

    a, s_in, s_out = jnp.asarray(in_array), jnp.asarray(in_start), jnp.asarray(out_start)
    zeros = lambda *lead: jnp.zeros(lead + out_shape, a.dtype)  # noqa: E731
    put = lambda x, o, si, so: map_array_2d(x, o, si, so, in_shape)  # noqa: E731
    per_f = lambda x, o, si, so: jnp.stack([put(x[j], o[j], si, so) for j in range(x.shape[0])])  # noqa: E731
    match (a.ndim, in_copies, out_copies):
        case (2, 1, 1):
            return put(a, zeros(), s_in, s_out)
        case (2, 1, n):
            return jnp.stack([put(a, zeros(), s_in[i], s_out[i]) for i in range(n)])
        case (3, 1, 1):
            return per_f(a, zeros(a.shape[0]), s_in, s_out)
        case (3, n, 1):
            o = zeros()
            for i in range(n):
                o = put(a[i], o, s_in[i], s_out[i])
            return o
        case (3, n, m) if n == m:
            return jnp.stack([put(a[i], zeros(), s_in[i], s_out[i]) for i in range(n)])
        case (3, 1, n):
            return jnp.stack([per_f(a, zeros(a.shape[0]), s_in[i], s_out[i]) for i in range(n)])
        case (4, 1, 1):
            return jnp.stack([per_f(a[i], zeros(a.shape[1]), s_in, s_out) for i in range(a.shape[0])])
        case (4, n, 1):
            o = zeros(a.shape[1])
            for i in range(n):
                o = per_f(a[i], o, s_in[i], s_out[i])
            return o
        case (4, n, m) if n == m:
            return jnp.stack([per_f(a[i], zeros(a.shape[1]), s_in[i], s_out[i]) for i in range(n)])


_N, _F, _IN, _OUT, _SLICE = 7, 3, (6, 5), (20, 17), (4, 3)
_CASES = {  # (in_array shape, in_copies, out_copies)
    "2d": ((6, 5), 1, 1),
    "2d_to_copies": ((6, 5), 1, _N),
    "freq": ((_F, 6, 5), 1, 1),
    "copies_to_one": ((_N, 6, 5), _N, 1),
    "copies_to_copies": ((_N, 6, 5), _N, _N),
    "freq_to_copies": ((_F, 6, 5), 1, _N),
    "batch_freq": ((2, _F, 6, 5), 1, 1),
    "copies_freq_to_one": ((_N, _F, 6, 5), _N, 1),
    "copies_freq_to_copies": ((_N, _F, 6, 5), _N, _N),
}


def _random_case(shape, in_copies, out_copies, seed=0):
    rng = np.random.default_rng(seed)
    n = max(in_copies, out_copies)
    per_copy = n > 1 and not (len(shape) == 4 and in_copies == out_copies == 1)
    start_shape = (n, 2) if per_copy else (2,)
    # starts partly outside the valid range (clamped like dynamic_slice) and overlapping copies
    in_start = rng.integers(-2, 5, start_shape)
    out_start = rng.integers(-3, 19, start_shape)
    return rng.normal(size=shape), in_start, out_start


class TestCopyPlacement:
    @pytest.mark.parametrize("case", list(_CASES))
    def test_matches_copy_by_copy_reference(self, case):
        from aim_resolve.model.map import map_array

        shape, n_in, n_out = _CASES[case]
        x, s_in, s_out = _random_case(shape, n_in, n_out)
        kw = dict(in_copies=n_in, out_copies=n_out, in_shape=_SLICE, out_shape=_OUT,
                  in_start=s_in, out_start=s_out)
        res = map_array(jnp.asarray(x), zoom=1, **kw)
        ref = _reference(x, n_in, n_out, _SLICE, _OUT, s_in, s_out)
        assert res.shape == ref.shape
        np.testing.assert_allclose(np.asarray(res), np.asarray(ref), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("case", list(_CASES))
    def test_gradient_matches_reference(self, case):
        import jax

        from aim_resolve.model.map import map_array

        shape, n_in, n_out = _CASES[case]
        x, s_in, s_out = _random_case(shape, n_in, n_out, seed=1)
        kw = dict(in_copies=n_in, out_copies=n_out, in_shape=_SLICE, out_shape=_OUT,
                  in_start=s_in, out_start=s_out)
        w = jnp.asarray(np.random.default_rng(2).normal(size=np.asarray(map_array(jnp.asarray(x), zoom=1, **kw)).shape))
        g = jax.grad(lambda a: jnp.sum(w * map_array(a, zoom=1, **kw)))(jnp.asarray(x))
        g_ref = jax.grad(lambda a: jnp.sum(w * _reference(a, n_in, n_out, _SLICE, _OUT, s_in, s_out)))(jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-12, atol=1e-12)

    def test_numpy_input_returns_numpy(self):
        from aim_resolve.model.map import map_array

        shape, n_in, n_out = _CASES["copies_to_one"]
        x, s_in, s_out = _random_case(shape, n_in, n_out)
        res = map_array(x, in_copies=n_in, out_copies=n_out, in_shape=_SLICE, out_shape=_OUT,
                        in_start=s_in, out_start=s_out, zoom=1)
        assert isinstance(res, np.ndarray)

    def test_real_points_and_tiles(self):
        """Refined points and tiles on a factor-3 grid: same result as the reference."""
        out = SignalGrid.build(space=(128, 128), factor=3)
        pts = PointGrid.build(coordinates=[[-40.5, 10.5], [20.5, -30.5], [0.5, 0.5]], n_copies=3).refine(3)
        tiles = SignalGrid.build(space=(8, 8), center=((-20, 5), (17, -11), (0, 24)), factor=3, n_copies=3)
        rng = np.random.default_rng(3)
        for grid, x in [(pts, rng.normal(size=(pts.n_copies, 1, 1))),
                        (tiles, rng.normal(size=(3,) + tuple(tiles.shape[-2:])))]:
            f = map_signal(grid, out)
            kw = f.keywords
            ref = _reference(x, kw["in_copies"], kw["out_copies"], kw["in_shape"], kw["out_shape"],
                             kw["in_start"], kw["out_start"])
            np.testing.assert_allclose(np.asarray(f(jnp.asarray(x))), np.asarray(ref), rtol=1e-12, atol=1e-12)

"""Tests for the log versions of the sky components (``log`` methods without the exponential)."""

import jax
import numpy as np
import pytest

from aim_resolve.model.points import PointModel
from aim_resolve.model.signal import SignalModel
from aim_resolve.model.tiles import TileModel

jax.config.update("jax_enable_x64", True)

GRID = dict(space=[16, 16], fov=[0.01, 0.01])
UBIK = dict(
    zero_mode=[1, 1],
    spatial_amplitude=dict(fluctuations=[5, 1], loglogavgslope=[-2, 0.5], flexibility=[1.2, 0.4], asperity=[0.2, 0.2]),
)
OLD_SF = dict(i0=dict(offset_std=[1, 1], fluctuations=[5, 1], loglogavgslope=[-2, 0.5], flexibility=[1.2, 0.4],
                      asperity=[0.2, 0.2]))
UBIK_MF = dict(UBIK, ref_freq_index=1, spectral_index=dict(mean=[-1, 0.5], fluctuations=[3, 1]),
               deviations=dict(process="wiener", sigma=[0.2, 0.08]),
               spectral_curvature=dict(mean=[0.0, 0.5], fluctuations=[1.0, 0.5]))
GAUSS = dict(cov_x=[0.1, 0.01], cov_y=[0.1, 0.01])
FREQ = [0.99e9, 1.05e9, 1.12e9, 1.33e9]


def _check(model, x, own):
    log = np.asarray(model.log(x))
    ref = np.log(np.asarray(own))
    assert log.shape == ref.shape
    np.testing.assert_allclose(log, ref, rtol=0, atol=1e-10)


class TestSignalModelLog:
    @pytest.mark.parametrize("params", [UBIK, OLD_SF], ids=["ubik", "old_sf"])
    @pytest.mark.parametrize("gaussian", [None, GAUSS], ids=["plain", "gaussian"])
    def test_equals_log_of_signal(self, params, gaussian):
        sm = SignalModel.build(grid=GRID, params=params, prefix="bg", offset=8.0, gaussian=gaussian)
        assert sm.log_model is not None
        x = sm.init(jax.random.PRNGKey(1))
        _check(sm, x, sm(x))

    def test_multi_frequency_with_curvature(self):
        sm = SignalModel.build(grid=GRID, freq=FREQ, params=UBIK_MF, prefix="bg", offset=8.0)
        x = sm.init(jax.random.PRNGKey(2))
        _check(sm, x, sm(x))

    def test_finite_where_the_signal_underflows(self):
        sm = SignalModel.build(grid=GRID, params=UBIK, prefix="bg", offset=-800.0)  # exp(-800) == 0
        x = sm.init(jax.random.PRNGKey(3))
        assert np.all(np.asarray(sm(x)) == 0)
        log = np.asarray(sm.log(x))
        assert np.all(np.isfinite(log)) and log.max() < -700
        g = jax.grad(lambda y: sm.log(y).sum())(x)
        assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(g))

    def test_copy_keeps_the_log_twin(self):
        sm = SignalModel.build(grid=GRID, params=UBIK, prefix="bg", offset=8.0)
        assert sm.copy().log_model is sm.log_model

    def test_without_exp_falls_back_to_log(self):
        sm = SignalModel.build(grid=GRID, params=UBIK, prefix="bg", offset=8.0, nonlinearity="exp")
        sm.log_model = None
        x = sm.init(jax.random.PRNGKey(4))
        _check(sm, x, sm(x))


class TestTileAndPointLog:
    @pytest.mark.parametrize("params", [UBIK, OLD_SF], ids=["ubik", "old_sf"])
    def test_tiles_per_tile(self, params):
        tm = TileModel.build(
            grid=dict(space=[32, 32], fov=[0.02, 0.02]),
            tile_grid=dict(space=[8, 8], center=[[-8, -8], [8, 8]], n_copies=2, fov=[0.005, 0.005]),
            params=params, prefix="t0", offset=[9.0, 7.0], gaussian=GAUSS,
        )
        x = tm.init(jax.random.PRNGKey(5))
        own = tm(x, map=False)
        assert np.asarray(own).shape == (2, 8, 8)
        _check(tm, x, own)

    def test_points_per_point(self):
        pm = PointModel.build(
            grid=dict(space=[32, 32], fov=[0.02, 0.02]),
            point_grid=dict(coordinates=[[0.5, 0.5], [3.5, -2.5]], n_copies=2),
            params=dict(i0=dict(mean=0, std=3)), prefix="p0", offset=[10.0, 12.0],
        )
        x = pm.init(jax.random.PRNGKey(6))
        own = pm(x, map=False)
        assert np.asarray(own).shape[0] == 2
        _check(pm, x, own)

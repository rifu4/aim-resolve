"""Tests for the zoom transition with log-space MAP fits of every component in its own space."""

import jax
import numpy as np
import pytest
from nifty.re import Vector

from aim_resolve.model.components import ComponentModel
from aim_resolve.model.grid import PointGrid, SignalGrid
from aim_resolve.model.points import PointModel
from aim_resolve.model.signal import SignalModel
from aim_resolve.model.tiles import TileModel
from aim_resolve.optimize.samples import MySamples
from aim_resolve.transition import (
    HYPER_KEYS,
    _log_target,
    component_own,
    log_map_fit,
    own_kl_fit,
    own_to_new,
    start_level,
    transition_zoom,
)

jax.config.update("jax_enable_x64", True)

FOV = [0.02, 0.02]
UBIK = dict(
    zero_mode=[1, 1],
    spatial_amplitude=dict(fluctuations=[1, 0.5], loglogavgslope=[-3, 0.5], flexibility=[1.2, 0.4], asperity=[0.2, 0.2]),
)
OLD_SF = dict(i0=dict(offset_std=[1, 1], fluctuations=[1, 0.5], loglogavgslope=[-3, 0.5], flexibility=[1.2, 0.4],
                      asperity=[0.2, 0.2]))
PS = dict(i0=dict(mean=0, std=3))
GAUSS = dict(cov_x=[0.1, 0.01], cov_y=[0.1, 0.01])
COOS = [[0.5, 0.5], [5.5, -6.5]]
CENTERS = [[-8, -8], [8, 8]]


def _sky(zoom, params, prefix):
    """Background + 2 tiles + 2 point sources on a 32^2 field at resolution factor `zoom`."""
    grid = dict(space=[32, 32], fov=FOV, factor=zoom)
    bg = SignalModel.build(grid=grid, params=params, prefix=f"bg.{prefix}", offset=6.0)
    tiles = TileModel.build(
        grid=dict(space=[32, 32], fov=FOV),
        tile_grid=dict(space=[8, 8], center=CENTERS, n_copies=2, factor=zoom),
        params=params, prefix=f"t0.{prefix}", offset=[9.0, 7.0], gaussian=GAUSS,
    )
    pg = PointGrid.build(coordinates=COOS, n_copies=2)
    if zoom > 1:
        pg = pg.refine(zoom)
    points = PointModel.build(
        grid=dict(space=[32, 32], fov=FOV),
        point_grid=dict(coordinates=[list(c) for c in pg.coordinates], factor=pg.factor, n_copies=pg.n_copies),
        params=PS, prefix=f"p0.{prefix}", offset=[11.0] * pg.n_copies,
    )
    return ComponentModel(SignalGrid.build(**grid), bg, f"sky.{prefix}", points, tiles)


def _log_corr(a, b, rel=1e-3):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = a > rel * a.max()
    return np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1]


class TestOwnToNew:
    def test_tiles_are_zoomed_per_tile(self):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        t_old, t_new = old.tiles[0], new.tiles[0]
        vals = np.asarray(t_old(t_old.init(jax.random.PRNGKey(0)), map=False))
        res = own_to_new(t_old, t_new, vals)
        assert res.shape == (2, 16, 16)
        np.testing.assert_array_equal(res[:, ::2, ::2], vals)

    def test_points_take_the_value_of_their_coarse_point(self):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        p_old, p_new = old.points[0], new.points[0]
        vals = np.asarray(p_old(p_old.init(jax.random.PRNGKey(1)), map=False))
        res = own_to_new(p_old, p_new, vals)
        assert res.shape[0] == 8  # 2 points x 2^2 sub-points
        np.testing.assert_array_equal(res[:4], np.repeat(vals[:1], 4, axis=0))
        np.testing.assert_array_equal(res[4:], np.repeat(vals[1:], 4, axis=0))

    def test_signal_components_are_mapped(self):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        b_old, b_new = old.background, new.background
        vals = np.asarray(b_old(b_old.init(jax.random.PRNGKey(2))))
        res = own_to_new(b_old, b_new, vals)
        assert res.shape == (64, 64)
        np.testing.assert_allclose(res[::2, ::2], vals)


class TestLogMapFit:
    def test_recovers_a_target_of_the_same_model(self):
        sm = SignalModel.build(grid=dict(space=[16, 16], fov=[0.01, 0.01]), params=UBIK, prefix="bg", offset=8.0)
        target = np.asarray(sm(sm.init(jax.random.PRNGKey(3))))
        pos = log_map_fit(jax.random.PRNGKey(4), sm, target, maxiter=50)
        assert _log_corr(target, sm(pos)) > 0.99
        assert np.isclose(np.asarray(sm(pos)).sum() / target.sum(), 1.0, rtol=0.1)

    def test_shape_mismatch_raises(self):
        sm = SignalModel.build(grid=dict(space=[16, 16], fov=[0.01, 0.01]), params=UBIK, prefix="bg", offset=8.0)
        with pytest.raises(ValueError, match="do not match"):
            log_map_fit(jax.random.PRNGKey(5), sm, np.ones((8, 8)), maxiter=1)


class TestZoomTransitionLogMap:
    def test_every_component_is_reproduced_in_its_own_space(self):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        x_old = old.init(jax.random.PRNGKey(6))
        samples = MySamples(pos=x_old, samples=None, keys=None)
        res, _ = transition_zoom(
            jax.random.PRNGKey(7), samples, 1,
            lh_old=dict(sky_model=old, noise_model=None), lh_new=dict(sky_model=new, noise_model=None),
            opt_dct=None, fit="log_map", log_map=dict(maxiter=60),
        )
        for c_old, c_new in zip(old.models, new.models, strict=True):
            target = own_to_new(c_old, c_new, component_own(c_old, x_old))
            fitted = np.asarray(component_own(c_new, res.pos))
            assert fitted.shape == target.shape
            if isinstance(c_new, (TileModel, PointModel)):  # every copy on its own
                for t, f in zip(target, fitted, strict=True):
                    assert t.size == 1 or _log_corr(t, f) > 0.97
                    assert np.isclose(f.sum() / t.sum(), 1.0, rtol=0.15), c_new.prefix
            else:
                assert _log_corr(target, fitted) > 0.97
                assert np.isclose(fitted.sum() / target.sum(), 1.0, rtol=0.15)

    def test_unknown_fit_raises(self):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        samples = MySamples(pos=old.init(jax.random.PRNGKey(8)), samples=None, keys=None)
        with pytest.raises(ValueError, match="Unknown fit"):
            transition_zoom(jax.random.PRNGKey(9), samples, 1, lh_old=dict(sky_model=old, noise_model=None),
                            lh_new=dict(sky_model=new, noise_model=None), opt_dct=None, fit="nope")


class TestLogMapFitStopping:
    def test_large_component_is_not_stopped_after_the_first_step(self):
        """With NIFTy's default xtol (1e-5 per parameter) a large component stops after one
        Newton step; the default of log_map_fit must keep going (here: reach the target)."""
        sm = SignalModel.build(grid=dict(space=[128, 128], fov=[0.05, 0.05]), params=UBIK, prefix="bg", offset=8.0)
        target = np.asarray(sm(sm.init(jax.random.PRNGKey(10))))
        pos = log_map_fit(jax.random.PRNGKey(11), sm, target, maxiter=30)
        assert _log_corr(target, sm(pos)) > 0.99


class TestLogMapFitHyperPrior:
    def test_amplitude_hyperparameters_stay_near_the_prior_median(self):
        """A target with a large amplitude (fluctuations +3 sigma) is fitted with the excitations, not
        by moving the amplitude hyperparameters into the prior tails."""
        sm = SignalModel.build(grid=dict(space=[64, 64], fov=[0.03, 0.03]), params=UBIK, prefix="bg", offset=8.0)
        x = dict(sm.init(jax.random.PRNGKey(12)))
        x["bg i0 fluctuations"] = x["bg i0 fluctuations"] * 0 + 3.0
        target = np.asarray(sm(Vector(x)))
        fits = {s: log_map_fit(jax.random.PRNGKey(13), sm, target, maxiter=40, hyper_std=s) for s in (1.0, 0.3)}
        h = {s: max(abs(float(p.tree[k])) for k in p.tree if k.split()[-1] in HYPER_KEYS) for s, p in fits.items()}
        assert h[0.3] < 0.85 * h[1.0]
        assert _log_corr(target, sm(fits[0.3])) > 0.99


class TestZoomTransitionLogKL:
    def test_every_component_is_fitted_with_the_kl_in_log_space(self):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        x_old = old.init(jax.random.PRNGKey(14))
        samples = MySamples(pos=x_old, samples=None, keys=None)
        opt_dct = dict(
            n_total_iterations=3,
            n_samples=1,
            draw_linear_kwargs=dict(cg_name=None, cg_kwargs=dict(maxiter=50)),
            nonlinearly_update_kwargs=dict(minimize_kwargs=dict(name=None, maxiter=5)),
            kl_kwargs=dict(minimize_kwargs=dict(name=None, maxiter=20)),
            sample_mode="linear_resample",
        )
        res, _ = transition_zoom(
            jax.random.PRNGKey(15), samples, 1,
            lh_old=dict(sky_model=old, noise_model=None), lh_new=dict(sky_model=new, noise_model=None),
            opt_dct=opt_dct, fit="log_kl",
        )
        for c_old, c_new in zip(old.models, new.models, strict=True):
            target = own_to_new(c_old, c_new, component_own(c_old, x_old))
            fitted = np.asarray(component_own(c_new, res.pos))
            assert fitted.shape == target.shape
            if isinstance(c_new, (TileModel, PointModel)):  # every copy on its own
                for t, f in zip(target, fitted, strict=True):
                    assert t.size == 1 or _log_corr(t, f) > 0.9, c_new.prefix
            else:
                assert _log_corr(target, fitted) > 0.9, c_new.prefix


class TestLogTargetFloor:
    def test_floor_penalises_only_models_brighter_than_the_floor(self):
        sm = SignalModel.build(grid=dict(space=[16, 16], fov=[0.01, 0.01]), params=UBIK, prefix="bg", offset=8.0)
        x = sm.init(jax.random.PRNGKey(16))
        data = np.asarray(sm(x)) * 1.0
        data[:8] *= 1e-8  # far below the floor
        log_d, log_s = _log_target(sm, data, 1e-4, floor=True)
        f = 1e-4 * data.max()
        np.testing.assert_allclose(log_d, np.log(data + f))
        np.testing.assert_allclose(log_s(x), np.log(np.asarray(sm(x)) + f), atol=1e-10)
        res = np.abs(np.asarray(log_s(x)) - np.asarray(log_d))
        assert res[8:].max() < 1e-10  # data reproduced above the floor
        assert res[:8].min() > 0  # brighter than the data below the floor: penalised
        # a model below the floor costs (almost) nothing: residual of 1e-8 * data vs 1e-10 * data
        g = np.log(np.asarray(sm(x))[:8] * 1e-10 + f) - np.log(data[:8] + f)
        assert np.abs(g).max() < 1e-2


KL_SHORT = dict(
    n_total_iterations=6,
    n_samples=1,
    draw_linear_kwargs=dict(cg_name=None, cg_kwargs=dict(maxiter=50)),
    nonlinearly_update_kwargs=dict(minimize_kwargs=dict(name=None, maxiter=5)),
    kl_kwargs=dict(minimize_kwargs=dict(name=None, maxiter=40)),
    sample_mode="linear_resample",
)


class TestOwnKLFit:
    def test_faint_tile_is_weighted_like_a_bright_one(self):
        """Per-copy noise: a tile 1e4 times fainter than the other is fitted as well."""
        _, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        t = new.tiles[0]
        target = np.array(t(t.init(jax.random.PRNGKey(17)), map=False))
        target[1] *= 1e-4 * target[0].max() / target[1].max()
        pos = own_kl_fit(jax.random.PRNGKey(18), t, target, dict(KL_SHORT, n_total_iterations=15))
        fitted = np.asarray(t(pos, map=False))
        for i in range(2):  # short schedule: shape exact, flux roughly
            assert _log_corr(target[i], fitted[i]) > 0.99, i
            assert np.isclose(fitted[i].sum() / target[i].sum(), 1.0, rtol=0.25), i

    def test_rejects_other_models(self):
        old, _ = _sky(1, OLD_SF, "4"), None
        with pytest.raises(TypeError):
            own_kl_fit(jax.random.PRNGKey(19), old, np.ones((32, 32)), KL_SHORT)

    def test_signal_component_with_wide_taper_start(self):
        sm = SignalModel.build(grid=dict(space=[32, 32], fov=[0.02, 0.02]), params=UBIK, prefix="o0", offset=8.0,
                               gaussian=GAUSS)
        target = np.array(sm(sm.init(jax.random.PRNGKey(28))))
        pos = own_kl_fit(jax.random.PRNGKey(29), sm, target, dict(KL_SHORT, n_total_iterations=10), taper=True,
                         taper_rel=0.1)
        assert np.isclose(np.asarray(sm(pos)).sum() / target.sum(), 1.0, rtol=0.3)
        assert _log_corr(target, sm(pos)) > 0.9

    def test_transition_fits_tiles_and_points_in_their_own_space(self):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        x_old = old.init(jax.random.PRNGKey(20))
        samples = MySamples(pos=x_old, samples=None, keys=None)
        res, _ = transition_zoom(
            jax.random.PRNGKey(21), samples, 1,
            lh_old=dict(sky_model=old, noise_model=None), lh_new=dict(sky_model=new, noise_model=None),
            opt_dct=KL_SHORT, fit="own_kl",
        )
        for c_old, c_new in zip(old.models, new.models, strict=True):
            if not isinstance(c_new, (TileModel, PointModel)):
                continue
            target = own_to_new(c_old, c_new, component_own(c_old, x_old))
            fitted = np.asarray(component_own(c_new, res.pos))
            for t, f in zip(target, fitted, strict=True):
                # linear fit of a blocky (pixel-duplicated) target: faint wings only roughly
                assert t.size == 1 or _log_corr(t, f) > 0.9, c_new.prefix
                assert np.isclose(f.sum() / t.sum(), 1.0, rtol=0.4), c_new.prefix


class TestStartLevel:
    def test_flux_of_every_copy_matches_the_target(self):
        _, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        t = new.tiles[0]
        target = np.array(t(t.init(jax.random.PRNGKey(22)), map=False))
        target[0] *= 1e3
        target[1] *= 1e-3
        pos = start_level(jax.random.PRNGKey(23), t, target)
        flux = np.asarray(t(pos, map=False)).reshape(2, -1).sum(1)
        np.testing.assert_allclose(flux, target.reshape(2, -1).sum(1), rtol=1e-6)

    def test_points_without_zero_mode_are_unchanged(self):
        _, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        p = new.points[0]
        x = p.init(jax.random.PRNGKey(24))
        pos = start_level(jax.random.PRNGKey(25), p, np.ones((8, 1, 1)), pos=x)
        for k in x:
            np.testing.assert_array_equal(pos.tree[k], x[k])

    def test_bright_tile_is_only_found_with_the_start_level(self):
        """A tile far above the prior level: without the start level the linear fit does not find
        it (gradient ~ S), with the start level it does."""
        _, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        t = new.tiles[0]
        target = np.array(t(t.init(jax.random.PRNGKey(26)), map=False)) * 1e3
        ratio = {}
        for level in (False, True):
            pos = own_kl_fit(jax.random.PRNGKey(27), t, target, KL_SHORT, level=level)
            fitted = np.asarray(t(pos, map=False))
            ratio[level] = fitted.reshape(2, -1).sum(1) / target.reshape(2, -1).sum(1)
        assert np.all(ratio[False] < 0.01)
        np.testing.assert_allclose(ratio[True], 1.0, rtol=0.1)


class TestOwnKLStartHelp:
    def test_signals_and_tiles_get_the_start_help(self, capsys):
        old, new = _sky(1, OLD_SF, "4"), _sky(2, UBIK, "5")
        x_old = old.init(jax.random.PRNGKey(30))
        samples = MySamples(pos=x_old, samples=None, keys=None)
        res, _ = transition_zoom(
            jax.random.PRNGKey(31), samples, 1,
            lh_old=dict(sky_model=old, noise_model=None), lh_new=dict(sky_model=new, noise_model=None),
            opt_dct=KL_SHORT, fit="own_kl", own_kl=dict(taper=True, signals=["bg"]),
        )
        out = capsys.readouterr().out
        assert "KL fit of bg.5 in its own space" in out and "KL fit of t0.5 in its own space" in out
        for c_old, c_new in zip(old.models, new.models, strict=True):
            target = own_to_new(c_old, c_new, component_own(c_old, x_old))
            fitted = np.asarray(component_own(c_new, res.pos))
            for t, f in zip(target.reshape(-1, *target.shape[-2:]), fitted.reshape(-1, *fitted.shape[-2:]), strict=True):
                assert 1 / 3 < f.sum() / t.sum() < 3, c_new.prefix  # short schedule: nothing lost

"""Tests for the optional spectral curvature of the ubik spectral model (SpectralIndexCurvature)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aim_resolve.model.signal import SignalModel
from aim_resolve.multi_frequency.spectral_product_utils.spectral_behavior import (
    SpectralIndex,
    SpectralIndexCurvature,
)

FREQ = np.array([992.8, 1053.6, 1115.1, 1331.2, 1398.3, 1465.7]) * 1e6
REF = 1
PARAMS = dict(
    ref_freq_index=REF,
    zero_mode=[1, 1],
    spatial_amplitude=dict(fluctuations=[5, 1], loglogavgslope=[-2, 0.5], flexibility=[1.2, 0.4], asperity=[0.2, 0.2]),
    spectral_index=dict(mean=[-1, 0.5], fluctuations=[3, 1]),
    deviations=dict(process="wiener", sigma=[0.2, 0.08]),
)
CURVATURE = dict(mean=[-0.2, 0.5], fluctuations=[1, 0.5])


def _signal(curvature=None, freq=FREQ):
    params = dict(PARAMS, spectral_curvature=curvature) if curvature is not None else dict(PARAMS)
    return SignalModel.build(grid=dict(space=[16, 16], fov=[0.1, 0.1]), freq=list(freq), params=params, prefix="bg")


def _behaviour(sm):
    return sm.model.log_spectral_behavior


def _u():
    return np.log(FREQ / FREQ[REF])


class TestBasis:
    def test_curvature_basis_orthogonal_and_anchored(self):
        b = _behaviour(_signal(CURVATURE))
        u, q = np.asarray(b.relative_log_frequencies).reshape(-1), np.asarray(b.curvature_basis).reshape(-1)
        np.testing.assert_allclose(u, _u())
        assert abs(np.dot(u, q)) < 1e-12 * np.dot(q, q) ** 0.5 * np.dot(u, u) ** 0.5
        assert q[REF] == 0.0
        np.testing.assert_allclose(q, u**2 - np.sum(u**3) / np.sum(u**2) * u)

    def test_deviations_carry_no_slope_or_curvature(self):
        b = _behaviour(_signal(CURVATURE))
        rng = np.random.default_rng(0)
        dev = jnp.asarray(rng.standard_normal((FREQ.size, 16, 16)))
        dev = dev.at[REF].set(0.0)  # as the fixed-point Wiener process
        res = np.asarray(b.remove_degeneracy_of_spectral_deviations(dev))
        u, q = np.asarray(b.relative_log_frequencies), np.asarray(b.curvature_basis)
        np.testing.assert_allclose(np.sum(res * u, axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.sum(res * q, axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(res[REF], 0, atol=1e-12)

    def test_needs_three_frequencies(self):
        with pytest.raises(ValueError, match="three frequencies"):
            _signal(CURVATURE, freq=FREQ[:2])


class TestOptIn:
    def test_without_curvature_unchanged(self):
        sm = _signal()
        assert type(_behaviour(sm)) is SpectralIndex
        x = sm.init(jax.random.PRNGKey(1))
        assert not any("beta" in k for k in x)
        assert sm.model.spectral_curvature_distribution(x) is None

    def test_with_curvature_has_beta_parameters(self):
        sm = _signal(CURVATURE)
        assert type(_behaviour(sm)) is SpectralIndexCurvature
        keys = set(sm.init(jax.random.PRNGKey(1)))
        assert {"bg beta mean", "bg beta xi", "bg beta fluctuations"} <= keys


class TestModel:
    @pytest.fixture
    def evaluated(self):
        sm = _signal(CURVATURE)
        x = sm.init(jax.random.PRNGKey(3))
        ln_sky = np.log(np.asarray(sm(x)))
        alpha = np.asarray(sm.spectral_index(x))
        beta = np.asarray(sm.spectral_curvature(x))
        i0 = np.asarray(sm.ref_freq_model(x))
        return sm, ln_sky, alpha, beta, i0

    def test_shapes(self, evaluated):
        _, ln_sky, alpha, beta, i0 = evaluated
        assert ln_sky.shape == (FREQ.size, 16, 16)
        assert alpha.shape == beta.shape == i0.shape == (16, 16)

    def test_reference_map_unchanged(self, evaluated):
        _, ln_sky, _, _, i0 = evaluated
        np.testing.assert_allclose(ln_sky[REF], np.log(i0), rtol=1e-10, atol=1e-10)

    def test_alpha_beta_recovered_by_anchored_fit(self, evaluated):
        """The deviations are orthogonal to u and q: an anchored least-squares fit of
        ln S - ln S(ref) on (u, q) returns exactly the model's alpha and beta."""
        sm, ln_sky, alpha, beta, _ = evaluated
        b = _behaviour(sm)
        u, q = np.asarray(b.relative_log_frequencies).reshape(-1), np.asarray(b.curvature_basis).reshape(-1)
        y = (ln_sky - ln_sky[REF]).reshape(FREQ.size, -1)
        coef = np.linalg.lstsq(np.stack([u, q], 1), y, rcond=None)[0]
        np.testing.assert_allclose(coef[0].reshape(16, 16), alpha, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(coef[1].reshape(16, 16), beta, rtol=1e-8, atol=1e-8)

    def test_beta_is_u2_coefficient(self, evaluated):
        """beta is the coefficient of u^2 of the parabola through the reference point."""
        _, ln_sky, _, beta, _ = evaluated
        u = _u()
        y = (ln_sky - ln_sky[REF]).reshape(FREQ.size, -1)
        coef = np.linalg.lstsq(np.stack([u, u**2], 1), y, rcond=None)[0]
        np.testing.assert_allclose(coef[1].reshape(16, 16), beta, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Point-source / prior model (MultiFrequencyModel): optional beta and the
# degeneracy-free deviations.
# ---------------------------------------------------------------------------
MF_PARAMS = dict(
    ref_freq_index=REF,
    i0=dict(mean=0.0, std=3.0),
    alpha=dict(mean=-1.0, std=0.5),
    deviations=dict(process="wiener", sigma=[0.2, 0.08]),
)
MF_BETA = dict(mean=0.0, std=0.5)


def _mf_signal(beta=None, deviations=True):
    params = dict(MF_PARAMS)
    if beta is not None:
        params["beta"] = beta
    if not deviations:
        params.pop("deviations")
    return SignalModel.build(grid=dict(space=[8, 8]), freq=list(FREQ), params=params, prefix="p")


def _fit(ln_sky, basis):
    y = (ln_sky - ln_sky[REF]).reshape(FREQ.size, -1)
    return np.linalg.lstsq(np.stack(basis, 1), y, rcond=None)[0]


class TestMultiFrequencyModelCurvature:
    def test_alpha_beta_recovered(self):
        sm = _mf_signal(MF_BETA)
        assert sm.has_spectral_curvature
        x = sm.init(jax.random.PRNGKey(5))
        ln_sky = np.log(np.asarray(sm(x)))
        u = _u()
        q = u**2 - np.sum(u**3) / np.sum(u**2) * u
        coef = _fit(ln_sky, [u, q])
        np.testing.assert_allclose(coef[0].reshape(8, 8), np.asarray(sm.spectral_index(x)), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(coef[1].reshape(8, 8), np.asarray(sm.spectral_curvature(x)), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(ln_sky[REF], np.log(np.asarray(sm.ref_freq_model(x))), rtol=1e-10)

    def test_deviations_without_slope_and_curvature(self):
        sm = _mf_signal(MF_BETA)
        x = sm.init(jax.random.PRNGKey(6))
        dev = np.asarray(sm.spectral_deviations(x))
        u = _u()[:, None, None]
        q = u**2 - np.sum(u**3) / np.sum(u**2) * u
        np.testing.assert_allclose(np.sum(dev * u, axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.sum(dev * q, axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(dev[REF], 0, atol=1e-12)

    def test_degeneracy_fixed_without_beta(self):
        """Without a curvature the deviations carry no slope: alpha is the slope of
        the line through the reference point (as in the ubik model)."""
        sm = _mf_signal()
        assert not sm.has_spectral_curvature
        x = sm.init(jax.random.PRNGKey(7))
        ln_sky = np.log(np.asarray(sm(x)))
        coef = _fit(ln_sky, [_u()])
        np.testing.assert_allclose(coef[0].reshape(8, 8), np.asarray(sm.spectral_index(x)), rtol=1e-8, atol=1e-8)

    def test_unchanged_without_deviations_and_beta(self):
        sm = _mf_signal(deviations=False)
        x = sm.init(jax.random.PRNGKey(8))
        expected = np.asarray(sm.ref_freq_model(x))[None] * np.exp(_u()[:, None, None] * np.asarray(sm.spectral_index(x))[None])
        np.testing.assert_allclose(np.asarray(sm(x)), expected, rtol=1e-10)

    def test_curvature_needs_three_frequencies(self):
        with pytest.raises(ValueError, match="three frequencies"):
            SignalModel.build(grid=dict(space=[8, 8]), freq=list(FREQ[:2]),
                              params=dict(MF_PARAMS, beta=MF_BETA, ref_freq_index=0), prefix="p")

    def test_no_curvature_accessor_raises(self):
        sm = _mf_signal()
        with pytest.raises(ValueError, match="no spectral curvature"):
            sm.spectral_curvature  # noqa: B018


class TestComponentAccessors:
    def test_point_model(self):
        from aim_resolve.model.points import PointModel

        pm = PointModel.build(
            grid=dict(space=[16, 16]),
            point_grid=dict(coordinates=[[-3.5, 2.5], [4.5, -1.5]], n_copies=2),
            freq=list(FREQ),
            params=dict(MF_PARAMS, beta=MF_BETA),
            prefix="p0",
        )
        assert pm.has_spectral_curvature
        x = pm.init(jax.random.PRNGKey(9))
        assert np.asarray(pm.spectral_curvature(x)).shape == np.asarray(pm.spectral_index(x)).shape

    def test_tile_model(self):
        from aim_resolve.model.tiles import TileModel

        tm = TileModel.build(
            grid=dict(space=(32, 32)),
            tile_grid=dict(space=(8, 8), n_copies=2, center=[(0, 0), (4, 4)]),
            freq=list(FREQ),
            params=dict(PARAMS, spectral_curvature=CURVATURE),
            prefix="t0",
        )
        assert tm.has_spectral_curvature
        x = tm.init(jax.random.PRNGKey(10))
        assert np.asarray(tm.spectral_curvature(x)).shape == np.asarray(tm.spectral_index(x)).shape

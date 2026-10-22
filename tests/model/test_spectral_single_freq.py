"""Tests for the single-frequency ubik model (same parametrisation as the multi-frequency model)."""

import jax
import numpy as np

from aim_resolve.model.components import ComponentModel
from aim_resolve.model.grid import SignalGrid
from aim_resolve.model.signal import SignalModel
from aim_resolve.model.spectral import MultiFrequencyModel
from aim_resolve.optimize.samples import MySamples
from aim_resolve.transition import transition_freq

FREQ = np.array([992.8, 1053.6, 1115.1, 1331.2, 1398.3, 1465.7]) * 1e6
GRID = dict(space=[16, 16], fov=[0.01, 0.01])
SPATIAL = dict(
    zero_mode=[1, 1],
    spatial_amplitude=dict(fluctuations=[5, 1], loglogavgslope=[-2, 0.5], flexibility=[1.2, 0.4], asperity=[0.2, 0.2]),
)
SPECTRAL = dict(
    ref_freq_index=1,
    spectral_index=dict(mean=[-1, 0.5], fluctuations=[3, 1]),
    deviations=dict(process="wiener", sigma=[0.2, 0.08]),
    spectral_curvature=dict(mean=[0.0, 0.5], fluctuations=[1.0, 0.5]),
)


def _signal(freq, params, prefix):
    return SignalModel.build(grid=GRID, freq=list(freq), params=params, prefix=prefix, offset=11.0)


def _i0_keys(x):
    return {k for k in x if " i0 " in k}


class TestSingleFrequencyUbik:
    def test_old_single_frequency_model_unchanged(self):
        sm = _signal([1.0], dict(i0=dict(offset_std=[1, 1], fluctuations=[5, 1], loglogavgslope=[-2, 0.5],
                                         flexibility=[1.2, 0.4], asperity=[0.2, 0.2])), "bg.0")
        assert isinstance(sm.model, MultiFrequencyModel)

    def test_new_single_frequency_model(self):
        sm = _signal([1.0], SPATIAL, "bg.0")
        assert not isinstance(sm.model, MultiFrequencyModel)
        x = sm.init(jax.random.PRNGKey(0))
        assert set(x) == _i0_keys(x)  # only the reference-frequency field
        assert np.asarray(sm(x)).shape == (16, 16)
        assert not sm.has_spectral_curvature

    def test_spectral_settings_ignored(self):
        """The multi-frequency params block can be reused for a single-frequency stage."""
        a = _signal([1.0], SPATIAL, "bg.0")
        b = _signal([1.0], dict(SPATIAL, **SPECTRAL), "bg.0")
        xa, xb = a.init(jax.random.PRNGKey(1)), b.init(jax.random.PRNGKey(1))
        assert set(xa) == set(xb)
        np.testing.assert_allclose(np.asarray(a(xa)), np.asarray(b(xa)), rtol=1e-12)

    def test_parameters_are_a_subset_of_the_multi_frequency_model(self):
        sf = _signal([1.0], SPATIAL, "bg")
        mf = _signal(FREQ, dict(SPATIAL, **SPECTRAL), "bg")
        assert set(sf.init(jax.random.PRNGKey(2))) <= set(mf.init(jax.random.PRNGKey(3)))


class TestFrequencyTransition:
    def test_transition_keeps_the_sky(self):
        """transition_freq copies the single-frequency parameters: the multi-frequency
        model reproduces the single-frequency sky exactly at the reference frequency."""
        grid = SignalGrid.build(space=(16, 16), fov=0.01)
        sf = ComponentModel(grid, _signal([1.0], SPATIAL, "bg.0"), "sky.0")
        mf = ComponentModel(grid, _signal(FREQ, dict(SPATIAL, **SPECTRAL), "bg.1"), "sky.1")
        x_sf = sf.init(jax.random.PRNGKey(4))
        new = transition_freq(
            jax.random.PRNGKey(5),
            MySamples(pos=x_sf, samples=None, keys=None),
            lh_old=dict(sky_model=sf, noise_model=None),
            lh_new=dict(sky_model=mf, noise_model=None),
        )
        s_sf = np.asarray(sf(x_sf))
        s_mf = np.asarray(mf(new.pos))
        assert s_mf.shape == (FREQ.size, 16, 16)
        np.testing.assert_allclose(s_mf[SPECTRAL["ref_freq_index"]], s_sf, rtol=1e-10)
        # the spectral parameters start close to their prior means
        alpha = np.asarray(mf.background.spectral_index(new.pos))
        assert abs(alpha.mean() + 1.0) < 0.1

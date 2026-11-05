"""Tests for aim_resolve.fast_resolve.beam and the pb_correction likelihood option."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from resolve.library.meerkat_beam import JimBeam
from scipy.signal import fftconvolve

import aim_resolve.likelihood as lh_mod
from aim_resolve.fast_resolve.beam import (
    BeamedSky,
    beam_offsets,
    build_primary_beam,
    default_beam_model,
)
from aim_resolve.fast_resolve.convolve import PSFConvolve
from aim_resolve.model.grid import SignalGrid
from aim_resolve.model.signal import SignalModel

FREQ = np.array([1.0e9, 1.2e9, 1.4e9])
L_BAND = "MKAT-AA-L-JIM-2020"


def _grid(factor=1, space=(32, 32), fov_deg=2.0, center=(0, 0)):
    return SignalGrid.build(space=space, factor=factor, fov=np.deg2rad(fov_deg), center=center)


def _sky(freq=FREQ, space=(8, 8)):
    return SignalModel.build(
        grid=dict(space=space, fov=float(np.deg2rad(2.0))),
        freq=list(freq),
        params=dict(i0=dict(mean=0.0, std=1.0), alpha=dict(mean=0.0, std=0.5)),
    )


class TestBeamGeometry:
    @pytest.mark.parametrize("factor", [1, 3])
    def test_matches_jimbeam_at_pixel_offsets(self, factor):
        grid = _grid(factor)
        pb = build_primary_beam(grid, FREQ, channel_average=False)
        assert pb.shape == (3,) + grid.shape
        pix_deg = np.rad2deg(grid.dis[0] / factor)
        n = grid.shape[0]
        for i, j in [(n // 2, n // 2), (n // 2 + 5 * factor, n // 2), (3, n - 4)]:
            exp = JimBeam(L_BAND).I(np.array((i - n / 2) * pix_deg), np.array((j - n / 2) * pix_deg), FREQ[2] / 1e6)
            np.testing.assert_allclose(pb[2, i, j], exp, rtol=1e-6)
        assert pb[:, n // 2, n // 2].min() > 0.999

    def test_single_frequency_drops_axis(self):
        pb = build_primary_beam(_grid(), [1.2e9])
        assert pb.shape == (32, 32)

    def test_pointing_offset_moves_peak(self):
        grid = _grid()
        pix_deg = np.rad2deg(grid.dis[0])
        pb = build_primary_beam(grid, FREQ, channel_average=False, pointing_offset=(4 * pix_deg, 0.0))
        assert np.unravel_index(np.argmax(pb[0]), pb[0].shape) == (16 + 4, 16)

    def test_grid_center_shifts_offsets(self):
        xx, yy = beam_offsets(_grid(center=(2, -3)))
        pix_deg = np.rad2deg(_grid().dis[0])
        np.testing.assert_allclose(xx[16, 16], 2 * pix_deg)
        np.testing.assert_allclose(yy[16, 16], -3 * pix_deg)


class TestDefaultModel:
    def test_l_band(self):
        assert default_beam_model([0.99e9, 1.47e9]) == L_BAND

    def test_uhf(self):
        assert default_beam_model([0.6e9, 0.8e9]) == "MKAT-AA-UHF-JIM-2020"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="No MeerKAT beam model"):
            default_beam_model([3.0e9])


class TestChannelAverage:
    def _obs(self, ch_freq, ch_wgt):
        return SimpleNamespace(freq=np.asarray(ch_freq), weight=np.ones((1, 5, len(ch_freq))) * np.asarray(ch_wgt))

    def test_one_channel_per_bin_equals_channel_beam(self):
        grid = _grid()
        ch = FREQ * 1.01  # one data channel inside each bin, off the model frequency
        pb = build_primary_beam(grid, FREQ, self._obs(ch, [1, 1, 1]))
        ref = build_primary_beam(grid, ch, channel_average=False)
        np.testing.assert_allclose(pb, ref, rtol=1e-6)

    def test_weighted_average(self):
        grid = _grid()
        ch = [1.0e9, 1.05e9, 1.2e9, 1.4e9]  # first two channels fall into the first bin
        pb = build_primary_beam(grid, FREQ, self._obs(ch, [3, 1, 1, 1]))
        single = build_primary_beam(grid, ch, channel_average=False)
        np.testing.assert_allclose(pb[0], 0.75 * single[0] + 0.25 * single[1], rtol=1e-6)
        np.testing.assert_allclose(pb[1:], single[2:], rtol=1e-6)

    def test_explicit_freq_bounds(self):
        grid = _grid()
        ch = [1.0e9, 1.09e9, 1.2e9, 1.4e9]
        # default midpoint bounds put 1.09 GHz into the first bin, explicit bounds into the second
        bounds = [0.9e9, 1.05e9, 1.3e9, 1.5e9]
        pb = build_primary_beam(grid, FREQ, self._obs(ch, [1, 1, 1, 1]), freq_bounds=bounds)
        single = build_primary_beam(grid, ch, channel_average=False)
        np.testing.assert_allclose(pb[0], single[0], rtol=1e-6)
        np.testing.assert_allclose(pb[1], 0.5 * single[1] + 0.5 * single[2], rtol=1e-6)
        np.testing.assert_allclose(pb[2], single[3], rtol=1e-6)


class TestSingleFrequencyBand:
    """A single model frequency (placeholder of a single-frequency stage) gets the beam
    averaged over all data channels, with the beam model chosen from the channels."""

    def _obs(self, ch_freq, ch_wgt):
        return SimpleNamespace(freq=np.asarray(ch_freq), weight=np.ones((1, 5, len(ch_freq))) * np.asarray(ch_wgt))

    def test_placeholder_frequency_averages_all_channels(self):
        grid = _grid()
        ch = [0.99e9, 1.05e9, 1.12e9]
        pb = build_primary_beam(grid, [1.0], self._obs(ch, [2, 1, 1]))
        single = build_primary_beam(grid, ch, channel_average=False)
        assert pb.shape == grid.shape
        np.testing.assert_allclose(pb, 0.5 * single[0] + 0.25 * single[1] + 0.25 * single[2], rtol=1e-6)

    def test_zero_weight_channels_are_ignored(self):
        grid = _grid()
        ch = [0.99e9, 1.05e9, 1.12e9]
        pb = build_primary_beam(grid, [1.0], self._obs(ch, [1, 0, 1]))
        single = build_primary_beam(grid, ch, channel_average=False)
        np.testing.assert_allclose(pb, 0.5 * single[0] + 0.5 * single[2], rtol=1e-6)

    def test_placeholder_without_data_raises(self):
        with pytest.raises(ValueError, match="No MeerKAT beam model"):
            build_primary_beam(_grid(), [1.0])


class TestBeamedSky:
    def test_output_is_beam_times_sky(self):
        sky = _sky()
        beam = np.random.default_rng(0).uniform(0.1, 1.0, sky.target.shape)
        app = BeamedSky(sky, beam)
        x = sky.init(jax.random.PRNGKey(0))
        np.testing.assert_allclose(app(x), beam * np.asarray(sky(x)), rtol=1e-6)
        assert app.domain == sky.domain and app.grid is sky.grid

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="beam shape"):
            BeamedSky(_sky(), np.ones((8, 8)))

    def test_psf_convolution_of_apparent_sky(self):
        """PSFConvolve(BeamedSky) must be the linear convolution of beam * sky."""
        sky = _sky()
        beam = np.random.default_rng(1).uniform(0.1, 1.0, sky.target.shape)
        kernel = np.random.default_rng(2).random((3, 16, 16))
        conv = PSFConvolve(BeamedSky(sky, beam), kernel)
        x = sky.init(jax.random.PRNGKey(3))
        s_app = beam * np.asarray(sky(x))
        ref = np.stack([fftconvolve(s_app[f], kernel[f], mode="full")[8:16, 8:16] for f in range(3)])
        np.testing.assert_allclose(conv(x), ref * sky.grid.dvol, rtol=1e-5, atol=1e-8)


class TestLikelihoodOption:
    @pytest.fixture
    def patched(self, monkeypatch):
        """fast_likelihood with all heavy resolve / kernel parts replaced."""
        calls = {}
        obs = SimpleNamespace(freq=FREQ, weight=np.ones((1, 4, 3)), vis=np.zeros((1, 4, 3)))
        obs.to_double_precision = lambda: obs
        R = SimpleNamespace(adjoint=lambda v: SimpleNamespace(val=np.zeros((3, 8, 8))))
        monkeypatch.setattr(lh_mod, "build_exact_responses", lambda *a: (R, None, "RNR", "RNR_l"))
        monkeypatch.setattr(lh_mod, "makeOp", lambda w: (lambda v: v))

        def psf_build(*, sky, **kw):
            calls["psf_sky"] = sky
            return "psf"

        monkeypatch.setattr(lh_mod.PSFConvolve, "build", staticmethod(psf_build))
        monkeypatch.setattr(
            lh_mod.NInvConvolve, "build", staticmethod(lambda **kw: SimpleNamespace(noise_model="nm"))
        )
        return obs, calls

    def test_off_keeps_intrinsic_sky(self, patched):
        obs, calls = patched
        sky = _sky()
        lh = lh_mod.fast_likelihood(sky=sky, data=obs)
        assert calls["psf_sky"] is sky and "sky_app" not in lh

    @pytest.mark.parametrize("opt", [True, {"channel_average": False}])
    def test_on_feeds_apparent_sky(self, patched, opt):
        obs, calls = patched
        sky = _sky()
        lh = lh_mod.fast_likelihood(sky=sky, data=obs, pb_correction=opt)
        assert isinstance(calls["psf_sky"], BeamedSky) and lh["sky_app"] is calls["psf_sky"]
        assert lh["sky_model"] is sky

    def test_radio_raises(self):
        with pytest.raises(NotImplementedError, match="only implemented for mode: fast"):
            lh_mod.radio_likelihood(sky=None, data=None, pb_correction=True)

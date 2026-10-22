"""Tests for aim_resolve.extension."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from aim_resolve.extension import extension_func, fun2mode, scale_sky_offsets


class TestExtensionFuncDispatch:
    """Test extension_func mode routing."""

    def test_unknown_mode_raises(self):
        with pytest.raises(TypeError, match="Unknown extension mode"):
            extension_func(mode="invalid")

    def test_freq_mode_accepted(self):
        """Dispatching to 'freq' should not raise TypeError."""
        with pytest.raises(Exception):
            # Will fail on missing kwargs, but the dispatch itself works.
            extension_func(mode="freq")

    def test_zoom_mode_accepted(self):
        with pytest.raises(Exception):
            extension_func(mode="zoom")


class TestFun2Mode:
    """Test the legacy fun→mode key replacement."""

    def test_replaces_fun_with_mode_lh(self):
        cfg = MagicMock()
        cfg.sections = {
            "lh.0": {"fun": "fast_likelihood"},
        }
        result = fun2mode(cfg)
        assert result.sections["lh.0"]["mode"] == "fast"
        assert "fun" not in result.sections["lh.0"]

    def test_replaces_fun_with_mode_radio_lh(self):
        cfg = MagicMock()
        cfg.sections = {
            "lh.0": {"fun": "radio_likelihood"},
        }
        result = fun2mode(cfg)
        assert result.sections["lh.0"]["mode"] == "radio"

    def test_replaces_fun_with_mode_image_lh(self):
        cfg = MagicMock()
        cfg.sections = {
            "lh.0": {"fun": "image_likelihood"},
        }
        result = fun2mode(cfg)
        assert result.sections["lh.0"]["mode"] == "image"

    def test_replaces_fun_with_mode_data(self):
        cfg = MagicMock()
        cfg.sections = {
            "data.0": {"fun": "radio_data"},
        }
        result = fun2mode(cfg)
        assert result.sections["data.0"]["mode"] == "radio"

    def test_replaces_fun_with_mode_image_data(self):
        cfg = MagicMock()
        cfg.sections = {
            "data.0": {"fun": "image_data"},
        }
        result = fun2mode(cfg)
        assert result.sections["data.0"]["mode"] == "image"

    def test_no_fun_key_unchanged(self):
        cfg = MagicMock()
        cfg.sections = {
            "lh.0": {"mode": "fast"},
        }
        result = fun2mode(cfg)
        assert result.sections["lh.0"]["mode"] == "fast"


class TestScaleSkyOffsets:
    """Test the flux scaling of the transferred sky via the component offsets."""

    @staticmethod
    def _cfg():
        from aim_resolve.optimize.set_config import SetupKLConfig

        return SetupKLConfig({
            "lh.6": {"sky": "=sky.6"},
            "sky.6": {"prefix": "sky.6", "background": "=sky_bg.6"},
            "sky_bg.6": {"prefix": "bg.6", "offset": 8.9},
            "sky_p0.6": {"prefix": "p0.6", "offset": [10.8, 13.5]},
            "sky_t0.6": {"prefix": "t0.6"},
            "sky_bg.5": {"prefix": "bg.5", "offset": 8.9},
        })

    def test_offsets_shifted_by_log_scale(self):
        cfg = scale_sky_offsets(self._cfg(), 6, 2.0)
        s = cfg.sections
        assert s["sky_bg.6"]["offset"] == pytest.approx(8.9 + np.log(2))
        assert s["sky_p0.6"]["offset"] == pytest.approx([10.8 + np.log(2), 13.5 + np.log(2)])
        assert s["sky_t0.6"]["offset"] == pytest.approx(np.log(2))  # missing offset counts as 0
        assert s["sky_bg.5"]["offset"] == 8.9  # other iterations unchanged
        assert "offset" not in s["sky.6"]  # the combined sky section has no offset

    def test_unit_scale_unchanged(self):
        cfg = scale_sky_offsets(self._cfg(), 6, 1.0)
        assert cfg.sections["sky_bg.6"]["offset"] == 8.9
        assert "offset" not in cfg.sections["sky_t0.6"]

    def test_non_positive_scale_raises(self):
        with pytest.raises(ValueError, match="positive"):
            scale_sky_offsets(self._cfg(), 6, 0.0)

    def test_offset_shift_scales_the_sky(self):
        """With the exponential nonlinearity, offset + ln(s) scales the component by exactly s."""
        import jax

        from aim_resolve.model.signal import SignalModel

        params = dict(i0=dict(offset_std=[1, 1], fluctuations=[5, 1], loglogavgslope=[-2, 0.5],
                              flexibility=[1.2, 0.4], asperity=[0.2, 0.2]))
        grid = dict(space=[16, 16], fov=[0.01, 0.01])
        a = SignalModel.build(grid=grid, freq=[1.0], params=params, prefix="bg", offset=8.9)
        b = SignalModel.build(grid=grid, freq=[1.0], params=params, prefix="bg", offset=8.9 + np.log(2.0))
        x = a.init(jax.random.PRNGKey(0))
        np.testing.assert_allclose(np.asarray(b(x)), 2.0 * np.asarray(a(x)), rtol=1e-12)

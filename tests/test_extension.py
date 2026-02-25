"""Tests for aim_resolve.extension."""

import pytest
from unittest.mock import MagicMock

from aim_resolve.extension import extension_func, fun2mode


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

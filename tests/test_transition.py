"""Tests for aim_resolve.transition."""

from functools import partial

import pytest

from aim_resolve.transition import transition_func


class TestTransitionFuncDispatch:
    """Test transition_func mode routing."""

    def test_unknown_mode_raises(self):
        with pytest.raises(TypeError, match="Unknown transition mode"):
            transition_func(mode="invalid")

    def test_anew_returns_partial(self):
        result = transition_func(mode="anew")
        assert isinstance(result, partial)

    def test_freq_returns_partial(self):
        result = transition_func(mode="freq")
        assert isinstance(result, partial)

    def test_addt_returns_partial(self):
        result = transition_func(mode="addt")
        assert isinstance(result, partial)

    def test_zoom_returns_partial(self):
        result = transition_func(mode="zoom")
        assert isinstance(result, partial)

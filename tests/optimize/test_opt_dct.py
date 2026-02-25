"""Tests for aim_resolve.optimize.opt_dct — callable_optimize_dict."""

import pytest

from aim_resolve.optimize.opt_dct import callable_optimize_dict, make_callable


class TestMakeCallable:
    def test_scalar(self):
        result = make_callable(42)
        assert result == 42

    def test_list(self):
        result = make_callable([10, 20, 30])
        assert callable(result)
        assert result(0) == 10
        assert result(2) == 30


class TestCallableOptimizeDict:
    def test_basic(self):
        dct = {
            "n_total_iterations": 3,
            "n_samples": 2,
            "draw_linear_kwargs": {},
            "nonlinearly_update_kwargs": {},
            "kl_kwargs": {},
            "sample_mode": "nonlinear_resample",
        }
        result = callable_optimize_dict(dct)
        assert "n_total_iterations" in result
        assert result["n_total_iterations"] == 3

    def test_missing_needed_raises(self):
        dct = {"n_samples": 2}  # missing n_total_iterations
        with pytest.raises(ValueError):
            callable_optimize_dict(dct)

"""Tests for aim_resolve.model.prior — prior_model dispatcher and model factories."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nifty.re import Model, VModel

from aim_resolve.model.prior import (
    prior_model,
    correlated_field_model,
    inverse_gamma_model,
    uniform_model,
    CFM_KEYS,
    NM_KEYS,
    IGM_KEYS,
    UM_KEYS,
)
from aim_resolve.model.grid import SignalGrid


@pytest.fixture
def grid():
    return SignalGrid.build(space=(16, 16))


# ---------- prior_model dispatcher ----------

class TestPriorModelDispatch:
    def test_dispatch_normal(self, grid):
        model, pspec = prior_model("test ", grid, mean=0.0, std=1.0)
        assert isinstance(model, Model)
        assert pspec is None

    def test_dispatch_inverse_gamma(self, grid):
        model, pspec = prior_model("test ", grid, alpha=3.0, scale=2.0)
        assert isinstance(model, Model)
        assert pspec is None

    def test_dispatch_uniform(self, grid):
        model, pspec = prior_model("test ", grid, u_min=0.0, u_max=1.0)
        assert isinstance(model, Model)
        assert pspec is None

    def test_dispatch_cfm(self, grid):
        model, pspec = prior_model(
            "test ", grid,
            offset_std=(1.0, 0.1),
            fluctuations=(1.0, 0.1),
            loglogavgslope=(-2.0, 0.5),
        )
        assert isinstance(model, Model)
        assert pspec is not None

    def test_invalid_keys_raises(self, grid):
        with pytest.raises(ValueError):
            prior_model("test ", grid, bad_key=42)


# ---------- correlated_field_model ----------

class TestCorrelatedFieldModel:
    def test_basic(self):
        model, power = correlated_field_model(
            prefix="cfm ",
            shape=(16, 16),
            distances=(1/16, 1/16),
            offset_std=(1.0, 0.1),
            fluctuations=(1.0, 0.1),
            loglogavgslope=(-2.0, 0.5),
        )
        assert isinstance(model, Model)
        assert isinstance(power, Model)
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (16, 16)

    def test_multi_copy(self):
        model, power = correlated_field_model(
            prefix="cfm ",
            shape=(8, 8),
            distances=(1/8, 1/8),
            offset_std=(1.0, 0.1),
            fluctuations=(1.0, 0.1),
            loglogavgslope=(-2.0, 0.5),
            n_copies=2,
        )
        assert isinstance(model, VModel)


# ---------- inverse_gamma_model ----------

class TestInverseGammaModel:
    def test_alpha_scale(self):
        model = inverse_gamma_model(prefix="ig", shape=(4, 4), alpha=3.0, scale=2.0)
        assert isinstance(model, Model)
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (4, 4)

    def test_mean_mode(self):
        model = inverse_gamma_model(prefix="ig", shape=(4,), mean=2.0, mode=1.5)
        assert isinstance(model, Model)

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError):
            inverse_gamma_model(prefix="ig", shape=(4,), mean=2.0, alpha=3.0)


# ---------- uniform_model ----------

class TestUniformModel:
    def test_basic(self):
        model = uniform_model(prefix="um", shape=(4, 4), u_min=0.0, u_max=1.0)
        assert isinstance(model, Model)
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (4, 4)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    def test_multi_copy(self):
        model = uniform_model(prefix="um", shape=(4,), u_min=-1.0, u_max=1.0, n_copies=3)
        assert isinstance(model, VModel)

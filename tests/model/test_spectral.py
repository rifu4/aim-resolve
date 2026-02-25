"""Tests for aim_resolve.model.spectral — MultiFrequencyModel and spectral_prior_model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nifty.re import Model

from aim_resolve.model.grid import SignalGrid
from aim_resolve.model.normal import normal_model
from aim_resolve.model.spectral import MultiFrequencyModel, spectral_prior_model


@pytest.fixture
def grid():
    return SignalGrid.build(space=(8, 8))


class TestMultiFrequencyModelSingleFreq:
    """Test MultiFrequencyModel with a single frequency (i0 only)."""

    def test_output_shape(self, grid):
        i0 = normal_model(prefix="i0 ", shape=grid.shape, mean=0.0, std=1.0)
        mfm = MultiFrequencyModel(i0=i0, nonlinearity=jnp.exp)
        key = jax.random.PRNGKey(0)
        x = mfm.init(key)
        result = mfm(x)
        assert result.shape == (8, 8)

    def test_no_nonlinearity(self, grid):
        i0 = normal_model(prefix="i0 ", shape=grid.shape, mean=0.0, std=1.0)
        mfm = MultiFrequencyModel(i0=i0, nonlinearity=None)
        key = jax.random.PRNGKey(0)
        x = mfm.init(key)
        result = mfm(x)
        assert result.shape == (8, 8)


class TestSpectralPriorModelSingleFreq:
    """Test spectral_prior_model with a single frequency."""

    def test_single_freq_build(self, grid):
        model = spectral_prior_model(
            prefix="sp ",
            grid=grid,
            freq=np.ones(1),
            i0=dict(mean=0.0, std=1.0),
        )
        assert isinstance(model, Model)
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (8, 8)

    def test_multi_freq_requires_alpha(self, grid):
        with pytest.raises(ValueError, match="alpha"):
            spectral_prior_model(
                prefix="sp ",
                grid=grid,
                freq=np.array([1.0, 2.0]),
                i0=dict(mean=0.0, std=1.0),
            )

    def test_multi_freq_with_alpha(self, grid):
        model = spectral_prior_model(
            prefix="sp ",
            grid=grid,
            freq=np.array([1.0, 2.0, 4.0]),
            i0=dict(mean=0.0, std=1.0),
            alpha=dict(mean=0.0, std=0.5),
        )
        assert isinstance(model, Model)
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (3, 8, 8)

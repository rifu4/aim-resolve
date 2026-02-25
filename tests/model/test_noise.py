"""Tests for aim_resolve.model.noise — NoiseModel and LazyNoise."""

import jax
import pytest

from aim_resolve.model.noise import NoiseModel, LazyNoise


class TestNoiseModelBuild:
    def test_lazy_noise_no_params(self):
        nm = NoiseModel.build(shape=(8, 8))
        assert isinstance(nm, LazyNoise)
        assert nm(None) == 1

    def test_lazy_noise_properties(self):
        nm = LazyNoise()
        assert nm.model is None
        assert nm.prefix is None
        assert nm.scaling is False
        assert nm.varcov is False

    def test_scaling_and_varcov_conflict(self):
        from aim_resolve.model.prior import inverse_gamma_model
        m = inverse_gamma_model(prefix=None, shape=(2, 2), alpha=2.0, scale=1.0)
        with pytest.raises(ValueError):
            NoiseModel(model=m, prefix='nm', scaling=True, varcov=True)

    def test_build_with_scaling(self):
        nm = NoiseModel.build(
            shape=(4, 4),
            parameters=dict(alpha=2.0, scale=1.0),
            scaling=True,
        )
        assert isinstance(nm, NoiseModel)
        assert nm.scaling is True
        assert nm.varcov is False
        # Calling with init
        key = jax.random.PRNGKey(0)
        x = nm.init(key)
        result = nm(x)
        # Result is 1/model(x), should be positive
        assert result.shape == (4, 4)

    def test_build_with_varcov(self):
        nm = NoiseModel.build(
            shape=(4, 4),
            parameters=dict(mean=2.0, mode=1.5),
            varcov=True,
        )
        assert isinstance(nm, NoiseModel)
        assert nm.varcov is True

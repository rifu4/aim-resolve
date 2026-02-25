"""Tests for aim_resolve.model.integer — IntegerPrior and integer_model."""

import jax
import jax.numpy as jnp

from aim_resolve.model.integer import IntegerPrior, integer_model, random_int


class TestRandomInt:
    def test_output_in_range(self):
        fn = random_int(0, 10, 1)
        xi = jnp.array([0.0, 0.5, -0.5, 2.0, -2.0])
        result = fn(xi)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 10)
        assert result.dtype == jnp.int32 or result.dtype == jnp.int64

    def test_step_size(self):
        fn = random_int(0, 10, 2)
        xi = jnp.linspace(-3, 3, 100)
        result = fn(xi)
        # All values should be even
        assert jnp.all(result % 2 == 0)


class TestIntegerPrior:
    def test_attributes(self):
        ip = IntegerPrior(5, 15, step=2, shape=(3,), name="test")
        assert ip.a_min == 5
        assert ip.a_max == 15
        assert ip.step == 2


class TestIntegerModel:
    def test_single_copy(self):
        model = integer_model(prefix="im", shape=(4,), i_min=0, i_max=5)
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (4,)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 5)

    def test_multi_copy(self):
        from nifty.re import VModel

        model = integer_model(prefix="im", shape=(4,), i_min=0, i_max=5, n_copies=3)
        assert isinstance(model, VModel)

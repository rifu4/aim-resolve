"""Tests for aim_resolve.optimize.samples — domain_tree, domain_keys, model_init, MySamples."""

import jax
import jax.numpy as jnp
import pytest
from nifty.re import Model, Vector

from aim_resolve.optimize.samples import domain_tree, domain_keys, model_init, MySamples
from aim_resolve.model.normal import normal_model


# ---------- domain_tree ----------

class TestDomainTree:
    def test_model(self):
        m = normal_model(prefix="test ", shape=(4,), mean=0.0, std=1.0)
        tree = domain_tree(m)
        assert isinstance(tree, dict)
        assert len(tree) > 0

    def test_dict(self):
        d = {"a": jnp.ones(3)}
        tree = domain_tree(d)
        assert tree is d

    def test_none(self):
        tree = domain_tree(None)
        assert tree == {}

    def test_iterable(self):
        m1 = normal_model(prefix="a ", shape=(2,), mean=0.0, std=1.0)
        m2 = normal_model(prefix="b ", shape=(2,), mean=0.0, std=1.0)
        tree = domain_tree([m1, m2])
        assert isinstance(tree, dict)
        # Should contain keys from both models
        assert len(tree) >= 2

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            domain_tree(42, error=True)

    def test_invalid_no_error(self):
        tree = domain_tree(42, error=False)
        assert tree == {}


# ---------- domain_keys ----------

class TestDomainKeys:
    def test_returns_set(self):
        m = normal_model(prefix="test ", shape=(4,), mean=0.0, std=1.0)
        keys = domain_keys(m)
        assert isinstance(keys, set)
        assert len(keys) > 0


# ---------- model_init ----------

class TestModelInit:
    def test_single(self):
        m = normal_model(prefix="test ", shape=(4,), mean=0.0, std=1.0)
        init = model_init(m)
        key = jax.random.PRNGKey(0)
        params = init(key)
        assert isinstance(params, dict)

    def test_iterable(self):
        m1 = normal_model(prefix="a ", shape=(2,), mean=0.0, std=1.0)
        m2 = normal_model(prefix="b ", shape=(2,), mean=0.0, std=1.0)
        init = model_init([m1, m2])
        key = jax.random.PRNGKey(0)
        params = init(key)
        assert isinstance(params, dict)

    def test_none(self):
        from nifty.re import Initializer
        init = model_init(None)
        assert isinstance(init, Initializer)


# ---------- MySamples ----------

class TestMySamples:
    def test_mean_map(self):
        pos = Vector({"x": jnp.array([1.0, 2.0])})
        s = MySamples(pos=pos, samples=None, keys=None)
        result = s.mean()
        assert isinstance(result, Vector)

    def test_mean_and_std_single(self):
        pos = Vector({"x": jnp.array([1.0, 2.0])})
        s = MySamples(pos=pos, samples=None, keys=None)
        # With < 2 samples and identity model returning a Vector,
        # jnp.zeros_like can't handle Vector — use a model that extracts an array
        model = lambda x: x["x"]
        m, std = s.mean_and_std(model)
        assert jnp.all(std == 0.0)

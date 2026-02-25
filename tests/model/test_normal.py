"""Tests for aim_resolve.model.normal — normal_model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aim_resolve.model.normal import normal_model


class TestNormalModel:
    def test_single_copy(self):
        model = normal_model(prefix="nm", shape=(8, 8), mean=0.0, std=1.0)
        key = jax.random.PRNGKey(42)
        x = model.init(key)
        result = model(x)
        assert result.shape == (8, 8)

    def test_multi_copy(self):
        model = normal_model(prefix="nm", shape=(4, 4), mean=[1.0, 2.0], std=[0.5, 0.5], n_copies=2)
        key = jax.random.PRNGKey(42)
        x = model.init(key)
        result = model(x)
        assert result.shape == (2, 4, 4)

    def test_zero_copies_per_element(self):
        model = normal_model(prefix="nm", shape=(3,), mean=[1.0, 2.0, 3.0], std=[0.1, 0.2, 0.3], n_copies=0)
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (3,)

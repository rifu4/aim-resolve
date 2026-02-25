"""Tests for aim_resolve.model.gaussian — gaussian_model, prior_or_const, centered_coos."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aim_resolve.model.gaussian import centered_coos, gaussian_model, prior_or_const


class TestCenteredCoos:
    def test_shape(self):
        coos = centered_coos(np.array([8, 8]), np.array([1.0, 1.0]))
        assert coos.shape == (2, 8, 8)

    def test_center_is_zero(self):
        coos = centered_coos(np.array([9, 9]), np.array([1.0, 1.0]))
        # For odd shape the center pixel should be at 0
        np.testing.assert_allclose(coos[:, 4, 4], [0.0, 0.0])

    def test_scaling(self):
        coos_1 = centered_coos(np.array([8, 8]), np.array([1.0, 1.0]))
        coos_2 = centered_coos(np.array([8, 8]), np.array([2.0, 2.0]))
        np.testing.assert_allclose(coos_2, 2 * coos_1)


class TestPriorOrConst:
    def test_constant_int(self):
        ptree = {}
        result = prior_or_const(5, ptree, "test")
        assert result == 5
        assert ptree == {}

    def test_constant_float(self):
        ptree = {}
        result = prior_or_const(3.14, ptree, "test")
        assert result == 3.14

    def test_tuple_creates_prior(self):
        ptree = {}
        result = prior_or_const((1.0, 0.1), ptree, "test")
        assert callable(result)
        assert len(ptree) > 0

    def test_invalid_type_raises(self):
        ptree = {}
        with pytest.raises(TypeError):
            prior_or_const("invalid", ptree, "test")


class TestGaussianModel:
    def test_output_shape(self):
        model = gaussian_model(
            prefix="gm ",
            shape=np.array([16, 16]),
            distances=np.array([1.0, 1.0]),
            cov_x=2.0,
            cov_y=2.0,
            scale=1.0,
            theta=0.0,
        )
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (16, 16)

    def test_positive_values(self):
        model = gaussian_model(
            prefix="gm ",
            shape=np.array([8, 8]),
            distances=np.array([1.0, 1.0]),
            cov_x=1.0,
            cov_y=1.0,
        )
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert jnp.all(result >= 0)

    def test_peak_at_center(self):
        model = gaussian_model(
            prefix="gm ",
            shape=np.array([9, 9]),
            distances=np.array([1.0, 1.0]),
            cov_x=2.0,
            cov_y=2.0,
            scale=1.0,
            theta=0.0,
        )
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        # Maximum should be at center (4,4)
        assert result[4, 4] == result.max()

    def test_with_prior_params(self):
        model = gaussian_model(
            prefix="gm ",
            shape=np.array([8, 8]),
            distances=np.array([1.0, 1.0]),
            cov_x=(2.0, 0.1),
            cov_y=(2.0, 0.1),
        )
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (8, 8)

    def test_n_copies(self):
        from nifty.re import VModel

        model = gaussian_model(
            prefix="gm ",
            shape=np.array([8, 8]),
            distances=np.array([1.0, 1.0]),
            cov_x=1.0,
            cov_y=1.0,
            n_copies=3,
        )
        assert isinstance(model, VModel)

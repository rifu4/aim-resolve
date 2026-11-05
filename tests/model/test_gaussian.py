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
            cov_x=(1.0, 0.1),
            cov_y=(1.0, 0.1),
            n_copies=3,
        )
        assert isinstance(model, VModel)

    def test_n_copies_constant(self):
        # constant taper: no parameters to map over, the same profile for every copy
        model = gaussian_model(
            prefix="gm ",
            shape=np.array([8, 8]),
            distances=np.array([1.0, 1.0]),
            cov_x=1.0,
            cov_y=1.0,
            n_copies=3,
        )
        out = np.asarray(model({}))
        assert out.shape == (3, 8, 8)
        assert np.allclose(out[0], out[2])


class TestCovPrior:
    def _model(self, cov_prior):
        return gaussian_model(
            prefix="gm ",
            shape=np.array([16, 16]),
            distances=np.array([1.0, 1.0]),
            cov_x=(0.3, 0.1),
            cov_y=(0.3, 0.1),
            cov_prior=cov_prior,
        )

    def test_normal_collapses_lognormal_stays_positive(self):
        """At -4 sigma the normal width (0.3 - 0.4) has crossed 0, the lognormal one is still positive."""
        from aim_resolve.model.gaussian import lognormal_prior, normal_prior

        assert normal_prior(0.3, 0.1)(-4.0) < 0
        assert 0 < lognormal_prior(0.3, 0.1)(-4.0) < 0.3
        assert lognormal_prior(0.3, 0.1)(-10.0) > 0

    def test_lognormal_mean_and_std(self):
        from aim_resolve.model.gaussian import lognormal_prior

        v = np.asarray(lognormal_prior(0.3, 0.1)(jax.random.normal(jax.random.PRNGKey(1), (200000,))))
        np.testing.assert_allclose([v.mean(), v.std()], [0.3, 0.1], rtol=0.02)

    def test_lognormal_model_is_finite_at_extreme_excitations(self):
        model = self._model("lognormal")
        x = {k: jnp.asarray(-8.0) for k in model.domain}
        result = np.asarray(model(x))
        assert np.all(np.isfinite(result)) and result.max() > 0

    def test_default_is_normal(self):
        a = self._model("normal")
        b = gaussian_model(prefix="gm ", shape=np.array([16, 16]), distances=np.array([1.0, 1.0]),
                           cov_x=(0.3, 0.1), cov_y=(0.3, 0.1))
        x = a.init(jax.random.PRNGKey(2))
        np.testing.assert_array_equal(a(x), b(x))

    def test_unknown_prior_raises(self):
        with pytest.raises(ValueError, match="cov_prior"):
            self._model("uniform")

    def test_tile_config_accepts_cov_prior(self):
        from aim_resolve.model.tiles import TileModel

        tm = TileModel.build(
            grid=dict(space=[32, 32], fov=[0.02, 0.02]),
            tile_grid=dict(space=[8, 8], center=[[-8, -8], [8, 8]], n_copies=2, fov=[0.005, 0.005]),
            params=dict(zero_mode=[1, 1], spatial_amplitude=dict(fluctuations=[1, 0.5], loglogavgslope=[-3, 0.5],
                                                                 flexibility=[1.2, 0.4], asperity=[0.2, 0.2])),
            prefix="t0", offset=[9.0, 7.0], gaussian=dict(cov_x=[0.3, 0.1], cov_y=[0.3, 0.1], cov_prior="lognormal"),
        )
        x = {k: (jnp.full(v.shape, -8.0) if "gm" in k else jnp.zeros(v.shape)) for k, v in tm.domain.tree.items()}
        assert np.all(np.isfinite(np.asarray(tm(x, map=False))))

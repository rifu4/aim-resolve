"""Tests for aim_resolve.fast_resolve.fast_kl — SkyResidualModel and build_likelihood."""

import jax.numpy as jnp
import numpy as np
from nifty.re import Model, Vector

from aim_resolve.fast_resolve.fast_kl import SkyResidualModel, build_likelihood


# A trivial nifty Model used as a stand-in sky_response.
class _IdentityModel(Model):
    """Minimal Model that broadcasts the sum of its input to a fixed shape."""

    def __init__(self, shape):
        self._shape = shape
        domain = Vector({"x": jnp.zeros(int(np.prod(shape)))})
        super().__init__(domain=domain, init=domain)

    def __call__(self, x, *args, **kwargs):
        return jnp.broadcast_to(jnp.sum(x["x"]), self._shape)


class TestSkyResidualModel:
    """Tests for the SkyResidualModel wrapper."""

    def test_stores_attributes(self):
        sky_resp = _IdentityModel((4, 4))
        old_rec = jnp.zeros((4, 4))
        res_data = jnp.zeros((4, 4))
        model = SkyResidualModel(sky_resp, old_rec, res_data)

        assert model.sky_response is sky_resp
        np.testing.assert_array_equal(model.old_reconstruction, old_rec)
        np.testing.assert_array_equal(model.residual_data, res_data)

    def test_call_returns_array(self):
        sky_resp = _IdentityModel((4, 4))
        old_rec = jnp.zeros((4, 4))
        res_data = jnp.zeros((4, 4))
        model = SkyResidualModel(sky_resp, old_rec, res_data)

        x = Vector({"x": jnp.ones(16)})
        result = model(x)
        assert result.shape == (4, 4)


class TestBuildLikelihood:
    """Tests for the fast-resolve Gaussian likelihood builder."""

    def test_returns_callable(self):
        sky_resp = _IdentityModel((4,))
        old_rec = jnp.zeros((4,))
        res_data = jnp.zeros((4,))

        lh = build_likelihood(
            sky_response=sky_resp,
            old_reconstruction=old_rec,
            residual_data=res_data,
        )
        assert callable(lh)

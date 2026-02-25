"""Tests for aim_resolve.resolve.response — rotate and response functions."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from aim_resolve.model.grid import SignalGrid
from aim_resolve.resolve.observation import Observation
from aim_resolve.resolve.response import one_point_response, rotate, signal_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(nrow=4, nfreq=1):
    """Minimal Observation for response tests."""
    rng = np.random.default_rng(0)
    pol = np.array(["I"])
    freq = np.linspace(1e9, 1.5e9, nfreq)
    uvw = np.column_stack(
        [rng.standard_normal(nrow), rng.standard_normal(nrow), np.zeros(nrow)]
    )
    ant1 = np.zeros((nrow, 1), dtype=int)
    ant2 = np.arange(nrow, dtype=int).reshape(-1, 1)
    time = np.linspace(0, 1, nrow).reshape(-1, 1)
    dfs = []
    for a, k in zip(
        [uvw, ant1, ant2, time], [["u", "v", "w"], ["ant1"], ["ant2"], ["time"]]
    ):
        dfs.append(pd.DataFrame(a, columns=[k]))
    antpos = pd.concat(dfs, axis=1)
    idx = pd.MultiIndex.from_product([pol, freq], names=["pol", "freq"])
    vis_arr = rng.standard_normal((nrow, len(idx))) + 1j * rng.standard_normal(
        (nrow, len(idx))
    )
    vis = pd.DataFrame(vis_arr.astype(np.complex128), columns=idx)
    wgt = pd.DataFrame(np.ones((nrow, len(idx))), columns=idx)
    return Observation(pol, freq, antpos, vis, wgt)


# ---------------------------------------------------------------------------
# Tests — rotate
# ---------------------------------------------------------------------------


class TestRotate:
    """Test the rotate function."""

    def test_identity_rotation(self):
        xy = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = rotate(xy, 0.0)
        np.testing.assert_allclose(result, xy, atol=1e-15)

    def test_90deg(self):
        xy = np.array([[1.0, 0.0]])
        result = rotate(xy, np.pi / 2)
        np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-14)

    def test_180deg(self):
        xy = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = rotate(xy, np.pi)
        np.testing.assert_allclose(result, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-14)

    def test_360deg_roundtrip(self):
        rng = np.random.default_rng(1)
        xy = rng.standard_normal((5, 2))
        result = rotate(xy, 2 * np.pi)
        np.testing.assert_allclose(result, xy, atol=1e-14)

    def test_output_shape(self):
        xy = np.zeros((3, 2))
        result = rotate(xy, 0.5)
        assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# Tests — one_point_response
# ---------------------------------------------------------------------------


class TestOnePointResponse:
    """Test the single-point response function."""

    def test_output_shape(self):
        obs = _make_obs(nrow=6, nfreq=2)
        x = jnp.ones((1, 1))  # scalar amplitude, broadcasts
        in_coos = jnp.array([0.0, 0.0])
        in_dis = np.array([1e-4, 1e-4])
        result = one_point_response(x, in_coos, in_dis, obs)
        # shape: (1, nrow, nfreq)
        assert result.shape == (1, 6, 2)

    def test_zero_amplitude(self):
        obs = _make_obs(nrow=4, nfreq=1)
        x = jnp.zeros((1, 1))
        in_coos = jnp.array([0.0, 0.0])
        in_dis = np.array([1e-4, 1e-4])
        result = one_point_response(x, in_coos, in_dis, obs)
        np.testing.assert_allclose(result, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# Tests — signal_response (type_check only, actual computation needs finufft/ducc)
# ---------------------------------------------------------------------------


class TestSignalResponse:
    """Test signal_response dispatching and type checks."""

    def test_type_check_grid(self):
        obs = _make_obs()
        with pytest.raises(TypeError):
            signal_response("not_a_grid", obs)

    def test_type_check_obs(self):
        grid = SignalGrid.build(space=(16, 16), fov="1deg")
        with pytest.raises(TypeError):
            signal_response(grid, "not_an_obs")

    def test_finu_response_returns_callable(self):
        grid = SignalGrid.build(space=(16, 16), fov="1deg")
        obs = _make_obs(nrow=4, nfreq=1)
        fn = signal_response(grid, obs, wgridding=False)
        assert callable(fn)

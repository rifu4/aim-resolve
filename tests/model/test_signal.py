"""Tests for aim_resolve.model.signal — SignalModel."""

import jax
import numpy as np

from aim_resolve.model.signal import SignalModel


class TestSignalModelBuild:
    def test_basic_build(self):
        sm = SignalModel.build(
            grid=dict(space=(16, 16)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
            prefix="sm",
        )
        assert isinstance(sm, SignalModel)
        assert sm.grid.shape == (16, 16)

    def test_callable(self):
        sm = SignalModel.build(
            grid=dict(space=(8, 8)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
        )
        key = jax.random.PRNGKey(42)
        x = sm.init(key)
        result = sm(x)
        assert result.shape == (8, 8)

    def test_no_nonlinearity(self):
        sm = SignalModel.build(
            grid=dict(space=(8, 8)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
            nonlinearity=None,
        )
        key = jax.random.PRNGKey(42)
        x = sm.init(key)
        result = sm(x)
        assert result.shape == (8, 8)

    def test_multi_freq(self):
        sm = SignalModel.build(
            grid=dict(space=(8, 8)),
            freq=[1.0, 2.0, 4.0],
            params=dict(
                i0=dict(mean=0.0, std=1.0),
                alpha=dict(mean=0.0, std=0.5),
            ),
        )
        key = jax.random.PRNGKey(42)
        x = sm.init(key)
        result = sm(x)
        assert result.shape == (3, 8, 8)


class TestSignalModelProperties:
    def test_shape_single_freq(self):
        sm = SignalModel.build(
            grid=dict(space=(16, 16)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
        )
        assert sm.shape == (16, 16)

    def test_copy(self):
        sm = SignalModel.build(
            grid=dict(space=(8, 8)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
        )
        sm2 = sm.copy()
        assert isinstance(sm2, SignalModel)
        assert sm2.prefix == sm.prefix

    def test_set_offset(self):
        sm = SignalModel.build(
            grid=dict(space=(8, 8)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
        )
        sm.set_offset(5.0)
        np.testing.assert_allclose(sm.offset, 5.0)

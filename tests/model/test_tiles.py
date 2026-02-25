"""Tests for aim_resolve.model.tiles — TileModel."""

import jax

from aim_resolve.model.tiles import TileModel


class TestTileModelBuild:
    def test_basic_build(self):
        tm = TileModel.build(
            grid=dict(space=(32, 32)),
            tile_grid=dict(space=(8, 8), n_copies=2, center=[(0, 0), (0, 0)]),
            params=dict(i0=dict(mean=0.0, std=1.0)),
            prefix="tm",
        )
        assert isinstance(tm, TileModel)
        assert tm.grid.shape == (32, 32)
        assert tm.n_copies == 2

    def test_callable(self):
        tm = TileModel.build(
            grid=dict(space=(32, 32)),
            tile_grid=dict(space=(8, 8), n_copies=2, center=[(0, 0), (0, 0)]),
            params=dict(i0=dict(mean=0.0, std=1.0)),
        )
        key = jax.random.PRNGKey(42)
        x = tm.init(key)
        result = tm(x)
        assert result.ndim >= 2


class TestTileModelProperties:
    def test_copy(self):
        tm = TileModel.build(
            grid=dict(space=(32, 32)),
            tile_grid=dict(space=(8, 8), n_copies=2, center=[(0, 0), (0, 0)]),
            params=dict(i0=dict(mean=0.0, std=1.0)),
        )
        tm2 = tm.copy()
        assert isinstance(tm2, TileModel)
        assert tm2.prefix == tm.prefix

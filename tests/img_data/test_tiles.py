"""Tests for aim_resolve.img_data.tiles — TileGenerator.build."""

import pytest
import jax
import jax.numpy as jnp

from aim_resolve.img_data.tiles import TileGenerator


class TestTileGeneratorBuild:
    """Test the build classmethod with minimal config."""

    def _grid_cfg(self, size=64):
        return dict(space=(size, size))

    def test_build_returns_model(self):
        tg = TileGenerator.build(
            n_min=1,
            n_max=3,
            grid=self._grid_cfg(),
            tile_size=(16, 16),
            i0=dict(mean=0.0, std=1.0),
        )
        assert isinstance(tg, TileGenerator)
        assert tg.grid.shape == (64, 64)

    def test_callable(self):
        tg = TileGenerator.build(
            n_min=1,
            n_max=2,
            grid=self._grid_cfg(32),
            tile_size=(8, 8),
            i0=dict(mean=0.0, std=1.0),
        )
        key = jax.random.PRNGKey(42)
        x = tg.init(key)
        result = tg(x)
        # TileGenerator stacks (x_val, zeros, y_val) → shape (3, H, W)
        assert result.shape == (3, 32, 32)

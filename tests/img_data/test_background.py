"""Tests for aim_resolve.img_data.background — BackgroundGenerator.build."""

import jax

from aim_resolve.img_data.background import BackgroundGenerator


class TestBackgroundGeneratorBuild:
    """Test the build classmethod with minimal config."""

    def _grid_cfg(self, size=32):
        return dict(space=(size, size))

    def test_build_returns_model(self):
        bg = BackgroundGenerator.build(
            grid=self._grid_cfg(),
            i0=dict(mean=0.0, std=1.0),
            func="exp",
        )
        assert isinstance(bg, BackgroundGenerator)
        assert bg.grid.shape == (32, 32)

    def test_build_func_none(self):
        bg = BackgroundGenerator.build(
            grid=self._grid_cfg(),
            i0=dict(mean=0.0, std=1.0),
            func=None,
        )
        assert bg.func is None

    def test_callable(self):
        bg = BackgroundGenerator.build(
            grid=self._grid_cfg(16),
            i0=dict(mean=0.0, std=1.0),
        )
        key = jax.random.PRNGKey(42)
        x = bg.init(key)
        result = bg(x)
        # BackgroundGenerator stacks (x_val, y_val, y_val) → shape (3, H, W)
        assert result.shape == (3, 16, 16)

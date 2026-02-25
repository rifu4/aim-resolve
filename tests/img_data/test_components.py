"""Tests for aim_resolve.img_data.components — ComponentGenerator.build."""

import pytest
import jax
import jax.numpy as jnp

from aim_resolve.img_data.components import ComponentGenerator


class TestComponentGeneratorBuild:
    """Test the build classmethod with a background-only config."""

    def _grid_cfg(self, size=32):
        return dict(space=(size, size))

    def test_build_background_only(self):
        cg = ComponentGenerator.build(
            grid=self._grid_cfg(),
            background=dict(
                i0=dict(mean=0.0, std=1.0),
            ),
        )
        assert isinstance(cg, ComponentGenerator)
        assert cg.points is None
        assert cg.tiles is None
        assert cg.objects is None

    def test_callable(self):
        cg = ComponentGenerator.build(
            grid=self._grid_cfg(16),
            background=dict(
                i0=dict(mean=0.0, std=1.0),
            ),
        )
        key = jax.random.PRNGKey(42)
        x = cg.init(key)
        result = cg(x)
        # ComponentGenerator stacks 3 channels: (intensity, points_map, objects_map)
        assert result.shape == (3, 16, 16)

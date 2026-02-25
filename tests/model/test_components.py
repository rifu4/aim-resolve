"""Tests for aim_resolve.model.components — ComponentModel."""

import jax
import pytest

from aim_resolve.model.components import ComponentModel
from aim_resolve.model.signal import SignalModel


@pytest.fixture
def background():
    return SignalModel.build(
        grid=dict(space=(16, 16)),
        params=dict(i0=dict(mean=0.0, std=1.0)),
        prefix="bg",
    )


class TestComponentModelBuild:
    def test_background_only(self, background):
        cm = ComponentModel.build(background=background)
        assert isinstance(cm, ComponentModel)
        assert cm.background is background
        assert len(cm.components) == 0

    def test_with_signal_component(self, background):
        obj = SignalModel.build(
            grid=dict(space=(16, 16)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
            prefix="obj",
        )
        cm = ComponentModel.build(background=background, sky_0=obj)
        assert len(cm.models) == 2
        assert len(cm.objects) == 1

    def test_duplicate_prefix_raises(self, background):
        dup = SignalModel.build(
            grid=dict(space=(16, 16)),
            params=dict(i0=dict(mean=0.0, std=1.0)),
            prefix="bg",  # same as background
        )
        with pytest.raises(ValueError, match="prefix"):
            ComponentModel.build(background=background, dup=dup)


class TestComponentModelProperties:
    def test_signals(self, background):
        cm = ComponentModel.build(background=background)
        assert len(cm.signals) == 1
        assert cm.signals[0] is background

    def test_points_empty(self, background):
        cm = ComponentModel.build(background=background)
        assert cm.points == ()

    def test_tiles_empty(self, background):
        cm = ComponentModel.build(background=background)
        assert cm.tiles == ()

    def test_shape(self, background):
        cm = ComponentModel.build(background=background)
        assert cm.shape == (16, 16)


class TestComponentModelCallable:
    def test_call(self, background):
        cm = ComponentModel.build(background=background)
        key = jax.random.PRNGKey(42)
        x = cm.init(key)
        result = cm(x)
        assert result.shape == (16, 16)

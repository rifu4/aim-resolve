"""Tests for aim_resolve.builders."""

import pytest

from aim_resolve.builders import get_builders
from aim_resolve.data import data_func
from aim_resolve.likelihood import likelihood_func
from aim_resolve.transition import transition_func

# ---------------------------------------------------------------------------
# Dispatch for data / likelihood / transition sections
# ---------------------------------------------------------------------------


class TestGetBuildersDispatch:
    """Test that get_builders selects the correct builder for each prefix."""

    def test_data_prefix(self):
        sections = {"data.0": {}}
        builders = get_builders(sections)
        assert builders["data.0"] is data_func

    def test_obs_prefix(self):
        sections = {"obs.0": {}}
        builders = get_builders(sections)
        assert builders["obs.0"] is data_func

    def test_lh_prefix(self):
        sections = {"lh.0": {}}
        builders = get_builders(sections)
        assert builders["lh.0"] is likelihood_func

    def test_likelihood_prefix(self):
        sections = {"likelihood.1": {}}
        builders = get_builders(sections)
        assert builders["likelihood.1"] is likelihood_func

    def test_trans_prefix(self):
        sections = {"trans.1": {}}
        builders = get_builders(sections)
        assert builders["trans.1"] is transition_func

    def test_transition_prefix(self):
        sections = {"transition.2": {}}
        builders = get_builders(sections)
        assert builders["transition.2"] is transition_func


# ---------------------------------------------------------------------------
# Dispatch for sky-model sections
# ---------------------------------------------------------------------------


class TestGetBuildersSkyDispatch:
    """Test sky-model type resolution based on section values."""

    def test_sky_background(self):
        sections = {"sky.0": {"background": True, "grid": {}}}
        builders = get_builders(sections)
        from aim_resolve.model.components import ComponentModel

        assert builders["sky.0"].__func__ is ComponentModel.build.__func__

    def test_sky_point_grid(self):
        sections = {"sky.0": {"point_grid": {}, "grid": {}}}
        builders = get_builders(sections)
        from aim_resolve.model.points import PointModel

        assert builders["sky.0"].__func__ is PointModel.build.__func__

    def test_sky_tile_grid(self):
        sections = {"sky.0": {"tile_grid": {}, "grid": {}}}
        builders = get_builders(sections)
        from aim_resolve.model.tiles import TileModel

        assert builders["sky.0"].__func__ is TileModel.build.__func__

    def test_sky_params(self):
        sections = {"sky.0": {"params": {}}}
        builders = get_builders(sections)
        from aim_resolve.model.signal import SignalModel

        assert builders["sky.0"].__func__ is SignalModel.build.__func__

    def test_sky_unknown_raises(self):
        sections = {"sky.0": {"unknown_key": True}}
        with pytest.raises(ValueError, match="Cannot determine"):
            get_builders(sections)


# ---------------------------------------------------------------------------
# Mixed sections
# ---------------------------------------------------------------------------


class TestGetBuildersMixed:
    """Test with a combination of different section types."""

    def test_multiple_sections(self):
        sections = {
            "data.0": {},
            "lh.0": {},
            "trans.1": {},
        }
        builders = get_builders(sections)
        assert len(builders) == 3
        assert builders["data.0"] is data_func
        assert builders["lh.0"] is likelihood_func
        assert builders["trans.1"] is transition_func

    def test_empty_sections(self):
        builders = get_builders({})
        assert builders == {}

    def test_unrecognised_prefix_is_ignored(self):
        """Sections with unknown prefixes should be silently skipped."""
        sections = {"something_else": {}}
        builders = get_builders(sections)
        assert "something_else" not in builders

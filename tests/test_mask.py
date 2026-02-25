"""Tests for aim_resolve.mask."""

import numpy as np
import pytest

from aim_resolve.mask import (
    add_freq_axis,
    add_margin,
    masks_from_maps,
    remove_freq_axis,
)


# ---------------------------------------------------------------------------
# add_freq_axis / remove_freq_axis
# ---------------------------------------------------------------------------

class TestAddFreqAxis:
    """Test inserting a frequency dimension."""

    def test_single_freq_no_change_2d(self):
        arr = np.ones((32, 32))
        result = add_freq_axis(arr, [1.0])
        assert result.shape == (32, 32)

    def test_multi_freq_2d(self):
        arr = np.ones((32, 32))
        result = add_freq_axis(arr, [1.0, 2.0])
        assert result.shape == (1, 32, 32)

    def test_multi_freq_3d(self):
        arr = np.ones((4, 32, 32))
        result = add_freq_axis(arr, [1.0, 2.0])
        assert result.shape == (4, 1, 32, 32)


class TestRemoveFreqAxis:
    """Test removing a frequency dimension."""

    def test_single_freq_no_change(self):
        arr = np.ones((32, 32))
        result = remove_freq_axis(arr, [1.0])
        assert result.shape == (32, 32)

    def test_multi_freq_3d(self):
        arr = np.ones((1, 32, 32))
        result = remove_freq_axis(arr, [1.0, 2.0])
        assert result.shape == (32, 32)

    def test_multi_freq_4d(self):
        arr = np.ones((4, 1, 32, 32))
        result = remove_freq_axis(arr, [1.0, 2.0])
        assert result.shape == (4, 32, 32)


class TestFreqAxisRoundtrip:
    """Adding then removing the freq axis should be the identity."""

    def test_roundtrip_2d(self):
        arr = np.random.default_rng(0).random((32, 32))
        freq = [1.0, 2.0]
        result = remove_freq_axis(add_freq_axis(arr, freq), freq)
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_3d(self):
        arr = np.random.default_rng(0).random((4, 32, 32))
        freq = [1.0, 2.0]
        result = remove_freq_axis(add_freq_axis(arr, freq), freq)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# add_margin
# ---------------------------------------------------------------------------

class TestAddMargin:
    """Test the distance-based margin addition."""

    def test_empty_array_unchanged(self):
        arr = np.zeros((32, 32))
        result = add_margin(arr, 5)
        np.testing.assert_array_equal(result, arr)

    def test_output_shape_matches_input(self):
        arr = np.zeros((64, 64))
        arr[30, 30] = 1
        result = add_margin(arr, 5)
        assert result.shape == arr.shape

    def test_values_in_unit_interval(self):
        arr = np.zeros((64, 64))
        arr[30:35, 30:35] = 1
        result = add_margin(arr, 10)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_round_yields_binary(self):
        arr = np.zeros((64, 64))
        arr[30:35, 30:35] = 1
        result = add_margin(arr, 5, round=True)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_margin_expands_region(self):
        arr = np.zeros((64, 64))
        arr[30:35, 30:35] = 1
        result = add_margin(arr, 5)
        # Pixels outside the original blob but within the margin should be > 0
        assert result[28, 32] > 0

    def test_tuple_margin(self):
        arr = np.zeros((64, 64))
        arr[30:35, 30:35] = 1
        result = add_margin(arr, (5, 10))
        assert result.shape == arr.shape
        assert result.sum() > arr.sum()


# ---------------------------------------------------------------------------
# masks_from_maps
# ---------------------------------------------------------------------------

class TestMasksFromMaps:
    """Test creating component masks from detection maps."""

    def _make_inputs(self):
        points = np.zeros((64, 64))
        points[10, 10] = 1
        objects = np.zeros((1, 64, 64))
        objects[0, 40:50, 40:50] = 1
        return points, objects

    def test_basic_keys_present(self):
        points, objects = self._make_inputs()
        masks = masks_from_maps(points, objects, it=0)
        assert "p0.0" in masks
        assert "bg.0" in masks
        assert "sum" in masks

    def test_object_key_created(self):
        points, objects = self._make_inputs()
        masks = masks_from_maps(points, objects, it=1)
        assert "o0.1" in masks

    def test_bg_mask_is_complement(self):
        """Background mask should cover non-object regions."""
        points = np.zeros((64, 64))
        objects = np.zeros((0, 64, 64))
        masks = masks_from_maps(points, objects, it=0)
        bg = masks["bg.0"]
        # Almost all pixels should belong to the background
        assert bg.sum() > 0.5 * bg.size

    def test_freq_axis_added_for_multiple_freqs(self):
        points, objects = self._make_inputs()
        masks = masks_from_maps(points, objects, it=0, freq=[1.0, 2.0])
        # Background gains freq axis: (1, H, W)
        assert masks["bg.0"].ndim == 3

    def test_no_points_no_objects(self):
        """Empty maps should still produce bg and sum masks."""
        points = np.zeros((32, 32))
        objects = np.zeros((0, 32, 32))
        masks = masks_from_maps(points, objects, it=0)
        assert "bg.0" in masks
        assert "sum" in masks

    def test_max_objects_limits_entries(self):
        """Only up to max_objects individual object masks should exist."""
        points = np.zeros((128, 128))
        # 8 separate object blobs
        objects = np.zeros((8, 128, 128))
        for i in range(8):
            r = 10 + i * 14
            objects[i, r : r + 5, r : r + 5] = 1

        masks = masks_from_maps(
            points, objects, it=0, max_objects=3, tile_size=0
        )
        obj_keys = [k for k in masks if k.startswith("o")]
        assert len(obj_keys) <= 3

    def test_tile_grouping(self):
        """Small objects should be grouped into tile masks."""
        points = np.zeros((128, 128))
        # 3 small objects that fit within tile_size=32
        objects = np.zeros((3, 128, 128))
        objects[0, 50:55, 50:55] = 1
        objects[1, 70:75, 70:75] = 1
        objects[2, 90:95, 90:95] = 1

        masks = masks_from_maps(
            points, objects, it=0, tile_size=32
        )
        tile_keys = [k for k in masks if k.startswith("t")]
        assert len(tile_keys) >= 1

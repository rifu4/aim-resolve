"""Tests for aim_resolve.clustering."""

import numpy as np
import pytest

from aim_resolve.clustering import dbscan_clustering, objects2points


# ---------------------------------------------------------------------------
# dbscan_clustering
# ---------------------------------------------------------------------------

class TestDbscanClustering:
    """Tests for the DBSCAN-based object clustering."""

    def test_empty_map_returns_empty(self):
        """An all-zero map should produce zero clusters and zero noise."""
        objects_map = np.zeros((64, 64))
        result = dbscan_clustering(objects_map, print_cl=False)
        # When no objects, only the empty cluster_maps array is returned
        assert result.shape[0] == 0
        assert result.shape[1:] == objects_map.shape

    def test_single_blob(self):
        """A single connected blob should yield one cluster map."""
        objects_map = np.zeros((64, 64))
        objects_map[20:30, 20:30] = 1

        cluster_maps, noise_map = dbscan_clustering(
            objects_map, print_cl=False, eps=0.5, min_samples=3
        )

        assert cluster_maps.shape[0] >= 1
        assert noise_map.shape == objects_map.shape
        # All blob pixels should appear across clusters + noise
        total = cluster_maps.sum(axis=0) + noise_map
        assert np.all(total[objects_map == 1] >= 1)

    def test_two_well_separated_blobs(self):
        """Two distant blobs should be detected as two clusters."""
        objects_map = np.zeros((128, 128))
        objects_map[10:20, 10:20] = 1
        objects_map[100:110, 100:110] = 1

        cluster_maps, noise_map = dbscan_clustering(
            objects_map, print_cl=False, eps=0.5, min_samples=3
        )

        assert cluster_maps.shape[0] == 2

    def test_clusters_sorted_by_descending_size(self):
        """Cluster maps should be ordered from largest to smallest."""
        objects_map = np.zeros((128, 128))
        # Smaller blob
        objects_map[10:15, 10:15] = 1
        # Larger blob
        objects_map[80:100, 80:100] = 1

        cluster_maps, _ = dbscan_clustering(
            objects_map, print_cl=False, eps=0.5, min_samples=3
        )

        sizes = [cm.sum() for cm in cluster_maps]
        assert sizes == sorted(sizes, reverse=True)

    def test_output_shapes(self):
        """Cluster and noise maps should match the spatial shape."""
        objects_map = np.zeros((48, 64))
        objects_map[10:20, 10:20] = 1

        cluster_maps, noise_map = dbscan_clustering(
            objects_map, print_cl=False, eps=0.5, min_samples=3
        )

        assert cluster_maps.shape[1:] == (48, 64)
        assert noise_map.shape == (48, 64)

    def test_noise_map_binary(self):
        """The noise map should only contain 0s and 1s."""
        objects_map = np.zeros((64, 64))
        objects_map[10:20, 10:20] = 1
        # Scatter a few isolated pixels to create noise
        rng = np.random.default_rng(42)
        for _ in range(15):
            r, c = rng.integers(0, 64, size=2)
            objects_map[r, c] = 1

        _, noise_map = dbscan_clustering(
            objects_map, print_cl=False, eps=0.3, min_samples=5
        )
        assert set(np.unique(noise_map)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# objects2points
# ---------------------------------------------------------------------------

class TestObjects2Points:
    """Tests for converting single-pixel noise clusters to points."""

    def test_empty_noise_returns_points(self):
        """When noise_map is empty the original points_map is returned."""
        points_map = np.zeros((64, 64))
        points_map[5, 5] = 1
        noise_map = np.zeros((64, 64))

        result = objects2points(points_map, noise_map, print_ps=False, eps=0.5)
        np.testing.assert_array_equal(result, points_map)

    def test_single_pixel_noise_added(self):
        """Isolated single-pixel noise should be added to points_map."""
        points_map = np.zeros((64, 64))
        noise_map = np.zeros((64, 64))
        noise_map[30, 30] = 1  # isolated single pixel

        result = objects2points(
            points_map.copy(), noise_map, print_ps=False, eps=0.3
        )
        assert result[30, 30] == 1

    def test_output_is_binary(self):
        """Result should be clipped to [0, 1]."""
        points_map = np.zeros((64, 64))
        points_map[10, 10] = 1
        noise_map = np.zeros((64, 64))
        noise_map[10, 10] = 1  # overlapping pixel

        result = objects2points(
            points_map.copy(), noise_map, print_ps=False, eps=0.3
        )
        assert result.max() <= 1
        assert result.min() >= 0

    def test_preserves_existing_points(self):
        """Original points should survive after adding noise points."""
        points_map = np.zeros((64, 64))
        points_map[5, 5] = 1
        points_map[50, 50] = 1
        noise_map = np.zeros((64, 64))
        noise_map[30, 30] = 1

        result = objects2points(
            points_map.copy(), noise_map, print_ps=False, eps=0.3
        )
        assert result[5, 5] == 1
        assert result[50, 50] == 1

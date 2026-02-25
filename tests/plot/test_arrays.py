"""Tests for aim_resolve.plot.arrays — plot_arrays smoke tests."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aim_resolve.plot.arrays import plot_arrays


class TestPlotArrays:
    def test_single_2d(self):
        arr = np.random.default_rng(0).random((16, 16))
        plot_arrays(arr)
        plt.close("all")

    def test_multiple_2d(self):
        arrs = [np.random.default_rng(i).random((8, 8)) for i in range(4)]
        plot_arrays(arrs, rows=2, cols=2)
        plt.close("all")

    def test_1d_power(self):
        arr = np.abs(np.random.default_rng(0).random(32)) + 0.01
        plot_arrays([arr])
        plt.close("all")

    def test_save(self, tmp_path):
        arr = np.random.default_rng(0).random((8, 8))
        plot_arrays(arr, odir=str(tmp_path), name="multi")
        assert (tmp_path / "multi.png").exists()
        plt.close("all")

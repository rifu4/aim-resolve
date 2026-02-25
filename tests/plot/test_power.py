"""Tests for aim_resolve.plot.power — plot_power smoke tests."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from aim_resolve.plot.power import plot_power


class TestPlotPower:
    def test_basic(self):
        arr = np.abs(np.random.default_rng(0).random(32)) + 0.01
        plot_power(arr)
        plt.close("all")

    def test_with_label(self):
        arr = np.abs(np.random.default_rng(0).random(16)) + 0.01
        plot_power(arr, label="pspec")
        plt.close("all")

    def test_save(self, tmp_path):
        arr = np.abs(np.random.default_rng(0).random(16)) + 0.01
        plot_power(arr, odir=str(tmp_path), name="pspec")
        assert (tmp_path / "pspec.png").exists()
        plt.close("all")

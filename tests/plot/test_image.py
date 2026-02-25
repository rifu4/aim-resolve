"""Tests for aim_resolve.plot.image — plot_image smoke tests."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aim_resolve.plot.image import plot_image


class TestPlotImage:
    def test_basic(self):
        """plot_image should not raise on a simple 2D array."""
        arr = np.random.default_rng(0).random((16, 16))
        plot_image(arr)
        plt.close("all")

    def test_with_axes(self):
        fig, ax = plt.subplots()
        axes = [ax]
        arr = np.random.default_rng(0).random((8, 8))
        plot_image(arr, axes=axes, label="test")
        plt.close("all")

    def test_log_norm(self):
        arr = np.random.default_rng(0).random((16, 16))
        plot_image(arr, norm="log")
        plt.close("all")

    def test_no_cbar(self):
        arr = np.random.default_rng(0).random((8, 8))
        plot_image(arr, cbar=False)
        plt.close("all")

    def test_save(self, tmp_path):
        arr = np.random.default_rng(0).random((8, 8))
        plot_image(arr, odir=str(tmp_path), name="test_img")
        assert (tmp_path / "test_img.png").exists()
        plt.close("all")

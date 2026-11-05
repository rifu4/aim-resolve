"""Tests for aim_resolve.plot.panels — pipeline panel plots, style and helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from aim_resolve.model.grid import SignalGrid
from aim_resolve.plot.panels import (
    _HALO,
    component_boxes,
    dynamic_limits,
    plot_components,
    plot_panels,
    plot_style,
    split_style,
)


def image(shape=(32, 32), seed=0):
    return np.random.default_rng(seed).random(shape) * 1e3 + 1.0


class TestDynamicLimits:
    def test_range_from_reference_max(self):
        ref = np.array([[1.0, 10.0], [100.0, 1e4]])
        assert dynamic_limits(ref, 1e2) == (100.0, 1e4)

    def test_explicit_limits_take_precedence(self):
        ref = np.array([[1.0, 1e4]])
        assert dynamic_limits(ref, 1e2, vmin=5.0, vmax=50.0) == (5.0, 50.0)
        assert dynamic_limits(ref, 1e2, vmax=1e3) == (10.0, 1e3)

    def test_without_dynamic_range_uses_positive_minimum(self):
        ref = np.array([[-1.0, 0.0, 2.0, 8.0]])
        assert dynamic_limits(ref, None) == (2.0, 8.0)


class TestComponentBoxes:
    def test_tile_and_object_boxes_in_pixels(self):
        grid = SignalGrid.build(space=(64, 64))
        sections = {
            "sky_o0.1": {"grid": {"space": [10, 6], "center": [0, 0]}},
            "sky_t0.1": {"tile_grid": {"space": [4, 4], "center": [[-20, -20], [10, 5]], "n_copies": 2}},
            "sky_o0.2": {"grid": {"space": [8, 8], "center": [0, 0]}},  # other iteration
        }
        tiles, objects = component_boxes(sections, grid, 1)
        assert len(objects) == 1 and len(tiles) == 2
        x0, y0, w, h = objects[0]
        # centred 10 x 6 box on a 64 x 64 grid: pixel edges 27 - 37 and 29 - 35 (minus half a pixel)
        assert (x0, y0, w, h) == pytest.approx((26.5, 28.5, 10.0, 6.0))
        assert tiles[0] == pytest.approx((9.5, 9.5, 4.0, 4.0))


class TestPlotStyle:
    def test_theme_colors_and_restore(self):
        before = plt.rcParams["text.color"], dict(_HALO)
        with plot_style("dark"):
            assert plt.rcParams["text.color"] == "white"
            assert plt.rcParams["savefig.transparent"]
            assert _HALO["color"] == "black"
        with plot_style("light", halo=False):
            assert plt.rcParams["axes.edgecolor"] == "black"
            assert _HALO["color"] is None
        assert (plt.rcParams["text.color"], dict(_HALO)) == before

    def test_unknown_theme(self):
        with pytest.raises(ValueError, match="unknown theme"):
            with plot_style("blue"):
                pass

    def test_split_style_defaults(self):
        theme, drange, pnl = split_style(None)
        assert theme == dict(theme="dark", transparent=True, halo=True, halo_width=1.0)
        assert drange == 1e4 and pnl == {}
        theme, drange, pnl = split_style(dict(theme="light", dynamic_range="1.0e3", cols=2, dpi=100))
        assert theme["theme"] == "light" and drange == 1e3 and pnl == dict(cols=2, dpi=100)


class TestPlotPanels:
    def test_saves_transparent_png(self, tmp_path):
        with plot_style("dark"):
            plot_panels([dict(arrays=[image()], labels=["a"])], name="one.png", odir=str(tmp_path), step="test", dpi=50)
        im = Image.open(tmp_path / "one.png").convert("RGBA")
        assert np.asarray(im)[..., 3].min() == 0  # transparent background

    def test_same_width_for_any_number_of_panels(self, tmp_path):
        widths = []
        for n in (1, 2, 5, 9):
            plot_panels([dict(arrays=[image(seed=k) for k in range(n)])], name=f"{n}.png", odir=str(tmp_path), dpi=50)
            widths.append(Image.open(tmp_path / f"{n}.png").size[0])
        assert max(widths) - min(widths) <= 4

    def test_groups_overlays_and_flexible_columns(self, tmp_path):
        mask = np.zeros((32, 32))
        mask[10:20, 12:18] = 1
        overlay = dict(circles=[(5, 5, 2)], boxes=[(8, 8, 4, 4)], contours=[mask], contour_numbers=True)
        groups = [
            dict(arrays=[image(), image(seed=1), image(seed=2)], overlays=[overlay, None, None]),
            dict(arrays=[mask, mask], norm="linear", vmin=0, vmax=1, cbar_label="mask"),
        ]
        plot_panels(groups, name="groups.png", odir=str(tmp_path), cols=1, dpi=50)
        assert (tmp_path / "groups.png").exists()

    def test_empty_groups(self, tmp_path):
        plot_panels([dict(arrays=[])], name="empty.png", odir=str(tmp_path))
        assert not (tmp_path / "empty.png").exists()


class _Model:
    def __init__(self, prefix, scale):
        self.prefix = prefix
        self.scale = scale


class _Samples:
    def mean(self, model):
        return image() * model.scale


class TestPlotComponents:
    def test_sky_and_components(self, tmp_path):
        models = [_Model("sky.1", 2.0), _Model("bg.1", 1.0), _Model("o0.1", 0.5)]
        plot_components(models, _Samples(), name="rec.png", odir=str(tmp_path), step="reconstruct · it 1", dpi=50)
        assert (tmp_path / "rec.png").exists()


class TestTransitionPlot:
    def test_target_and_fit(self, tmp_path):
        from aim_resolve.transition import _plot_target_and_fit

        fit = image()
        _plot_target_and_fit(fit * 1.1, fit, dict(name="stage.png", odir=str(tmp_path)), {"dpi": 50}, "pre-fit · it 1 · o0.1")
        _plot_target_and_fit(None, np.stack([fit, fit]), dict(name="cb.png", odir=str(tmp_path)), {"dpi": 50}, None)
        assert (tmp_path / "stage.png").exists() and (tmp_path / "cb.png").exists()


class TestSubGrids:
    def test_sub_grid_panel_with_overlays(self, tmp_path):
        tiles = [image((8, 8), seed=k) for k in range(5)]
        overlays = [dict(contours=[t > 500]) for t in tiles]
        group = dict(arrays=[image(), tiles], labels=["sky", "t0"], overlays=[None, overlays])
        plot_panels([group], name="sub.png", odir=str(tmp_path), dpi=50)
        assert (tmp_path / "sub.png").exists()

    def test_cutouts_centered_and_padded(self):
        from aim_resolve.plot.panels import _cutouts

        img = np.arange(100.0).reshape(10, 10)
        a, b = _cutouts(img, [(5, 5), (0, 9)], 4)
        assert a.shape == b.shape == (4, 4)
        assert a[2, 2] == img[5, 5]
        assert b[2, 2] == img[0, 9] and np.isnan(b[0, 0]) and np.isnan(b[2, 3])


# real component models with a point and a tile component (MAP samples)
FREQ = np.array([0.99, 1.05, 1.12, 1.33, 1.40, 1.47]) * 1e9
CF = dict(fluctuations=[1, 0.5], loglogavgslope=[-3, 0.5], flexibility=[1, 0.4], asperity=[0.2, 0.2])
SIGNAL_SF = dict(i0=dict(offset_std=[1, 1], **CF))
SIGNAL_MF = dict(
    ref_freq_index=1,
    zero_mode=[1, 1],
    spatial_amplitude=CF,
    spectral_index=dict(mean=[-1, 0.3], fluctuations=[0.5, 0.2]),
    deviations=dict(process="wiener", sigma=[0.2, 0.08]),
    spectral_curvature=dict(mean=[0, 0.2], fluctuations=[0.2, 0.1]),
)
POINTS_SF = dict(i0=dict(mean=0.0, std=3.0))
POINTS_MF = dict(
    ref_freq_index=1, i0=dict(mean=0.0, std=3.0), alpha=dict(mean=-1.0, std=0.5), beta=dict(mean=0.0, std=0.3),
    deviations=dict(process="wiener", sigma=[0.2, 0.08]),
)


def _sky(freq):
    import jax

    from aim_resolve.model import ComponentModel, PointModel, SignalModel, TileModel
    from aim_resolve.optimize import MySamples

    mf = len(freq) > 1
    sig, pts = (SIGNAL_MF, POINTS_MF) if mf else (SIGNAL_SF, POINTS_SF)
    grid = dict(space=[32, 32])
    bg = SignalModel.build(grid=grid, freq=list(freq), params=sig, prefix="bg")
    pm = PointModel.build(
        grid=grid, point_grid=dict(coordinates=[[-10.5, 6.5], [3.5, -2.5], [12.5, 12.5]], n_copies=3),
        freq=list(freq), params=pts, prefix="p0",
    )
    tm = TileModel.build(
        grid=grid, tile_grid=dict(space=(8, 8), n_copies=5, center=[(-8, -8), (0, 0), (8, 8), (-8, 8), (8, -8)]),
        freq=list(freq), params=sig, prefix="t0",
    )
    sky = ComponentModel.build(background=bg, prefix="sky", p0=pm, t0=tm)
    return sky, pm, MySamples(pos=sky.init(jax.random.PRNGKey(3)), samples=None)


class TestComponentSubGrids:
    def test_point_pixels_match_mapping(self):
        from aim_resolve.plot.panels import _point_pixels

        sky, _, samples = _sky([1.0])
        pm = sky.models[1]
        mapped = np.squeeze(np.asarray(pm(samples.pos, map=True)))
        expected = {tuple(p) for p in np.argwhere(mapped != 0)}
        assert {tuple(p) for p in _point_pixels(pm, sky.grid)} == expected

    def test_tile_and_point_images(self):
        from aim_resolve.plot.panels import _panel_image

        sky, _, samples = _sky([1.0])
        sky_img = np.squeeze(np.asarray(samples.mean(sky)))
        tiles = _panel_image(sky.models[2], samples)
        points = _panel_image(sky.models[1], samples, sky_img, point_cutout=6)
        assert len(tiles) == 5 and all(t.shape == (8, 8) for t in tiles)
        assert len(points) == 3 and all(p.shape == (6, 6) for p in points)

    def test_plot_components_with_sub_grids(self, tmp_path):
        sky, _, samples = _sky([1.0])
        plot_components([sky] + list(sky.models), samples, name="rec.png", odir=str(tmp_path), dpi=50)
        assert (tmp_path / "rec.png").exists()

    def test_kl_callback_multi_frequency(self, tmp_path):
        from aim_resolve.plot.panels import plot_kl_callback

        sky, _, samples = _sky(FREQ)
        plot_kl_callback([sky], samples, 3, str(tmp_path), dpi=50)
        names = {p.name for p in tmp_path.iterdir()}
        assert {"3_sky.png", "3_sky_alpha.png", "3_sky_beta.png"} <= names


class TestCap:
    def test_brightest_kept_in_order_and_labelled(self):
        from aim_resolve.plot.panels import _capped

        subs = [np.full((4, 4), v) for v in (1.0, 5.0, 3.0, 4.0, 2.0)]
        images, labels, sel = _capped([image(), subs], ["sky", "t0"], 3)
        assert sel == [None, [1, 2, 3]]
        assert [a[0, 0] for a in images[1]] == [5.0, 3.0, 4.0]
        assert labels == ["sky", "t0 · 3 of 5"]

    def test_within_limit_unchanged(self):
        from aim_resolve.plot.panels import _capped

        subs = [np.ones((4, 4))] * 3
        images, labels, sel = _capped([subs], ["p0"], 100)
        assert images[0] is subs and labels == ["p0"] and sel == [None]

    def test_kl_callback_with_cap(self, tmp_path):
        from aim_resolve.plot.panels import plot_kl_callback

        sky, _, samples = _sky(FREQ)
        plot_kl_callback([sky], samples, 4, str(tmp_path), dpi=50, max_sub=2, point_cutout=8)
        assert (tmp_path / "4_sky_beta.png").exists()

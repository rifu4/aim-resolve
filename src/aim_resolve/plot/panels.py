"""Multi-panel plots sharing one colorbar, with light/dark theme and transparent background."""

import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.patches import Circle, Rectangle

from ..model.grid import SignalGrid
from .util import rows_and_cols

THEME_COLORS = {"light": "black", "dark": "white"}

# halo (outline in the opposite color) of the active plot_style, used by plot_panels
_HALO = {"color": None, "width": 0.0}


@contextmanager
def plot_style(theme="dark", transparent=True, halo=True, halo_width=1.0):
    """Context setting the plot theme for all figures created and saved inside it.

    Parameters
    ----------
    theme : str, optional
        'light' (black axes, ticks and text) or 'dark' (white axes, ticks and text).
        Default is 'dark'.
    transparent : bool, optional
        Whether figures are saved with a transparent background. Default is True.
    halo : bool, optional
        Whether frames, colorbar outlines, ticks and tick labels of ``plot_panels`` get
        a thin outline in the opposite color, so they stay readable on both light and
        dark backgrounds. Default is True.
    halo_width : float, optional
        Width of the halo in points. Default is 1.0.
    """
    if theme not in THEME_COLORS:
        raise ValueError(f"unknown theme '{theme}', use 'light' or 'dark'")
    color = THEME_COLORS[theme]
    rc = {
        "text.color": color,
        "axes.labelcolor": color,
        "axes.edgecolor": color,
        "axes.titlecolor": color,
        "xtick.color": color,
        "ytick.color": color,
        "xtick.labelcolor": color,
        "ytick.labelcolor": color,
        "legend.labelcolor": color,
        "legend.edgecolor": color,
    }
    if transparent:
        rc |= {
            "savefig.transparent": True,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "legend.facecolor": "none",
            "legend.framealpha": 0.0,
        }
    previous = dict(_HALO)
    _HALO.update(color=THEME_COLORS["dark" if theme == "light" else "light"] if halo else None, width=halo_width)
    try:
        with plt.rc_context(rc):
            yield
    finally:
        _HALO.update(previous)


def _halo(linewidth):
    """Path effects drawing the active halo around an artist of the given line width."""
    if _HALO["color"] is None:
        return []
    return [
        patheffects.Stroke(linewidth=linewidth + 2 * _HALO["width"], foreground=_HALO["color"]),
        patheffects.Normal(),
    ]


def dynamic_limits(reference, dynamic_range=1e4, vmin=None, vmax=None):
    """Color limits from a reference image and a dynamic range.

    Parameters
    ----------
    reference : np.ndarray
        Reference image (e.g. the full sky); its maximum sets ``vmax``.
    dynamic_range : float or None, optional
        Ratio vmax / vmin. None leaves ``vmin`` to the positive minimum of the
        reference. Default is 1e4.
    vmin, vmax : float, optional
        Explicit limits; they take precedence over the reference and the
        dynamic range.

    Returns
    -------
    vmin, vmax : float
    """
    reference = np.asarray(reference, dtype="float64")
    finite = reference[np.isfinite(reference)]
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if vmin is None:
        if dynamic_range:
            vmin = vmax / float(dynamic_range)
        else:
            pos = finite[finite > 0]
            vmin = float(pos.min()) if pos.size else vmax / 1e4
    return float(vmin), float(vmax)


def component_boxes(sections, grid, it):
    """Pixel boxes of the tile and object components of an iteration.

    Parameters
    ----------
    sections : dict
        Model configuration sections (with ``sky_t*.{it}`` / ``sky_o*.{it}``).
    grid : SignalGrid
        Grid of the plotted image.
    it : int or str
        Iteration number.

    Returns
    -------
    tiles, objects : list of tuple
        Boxes (x0, y0, width, height) in pixel coordinates of ``grid`` (x: axis 0).
    """
    lims0 = np.asarray(grid.lims)[:, 0]
    fac = np.asarray(grid.fac, dtype="float64")

    def to_pixels(lims):
        lims = np.asarray(lims, dtype="float64").reshape((-1, 2, 2))
        boxes = []
        for lm in lims:
            p0 = (lm[:, 0] - lims0) * fac - 0.5
            p1 = (lm[:, 1] - lims0) * fac - 0.5
            boxes.append((p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]))
        return boxes

    tiles, objects = [], []
    for k, v in sections.items():
        if not k.endswith(f".{it}"):
            continue
        if k.startswith("sky_t") and "tile_grid" in v:
            tiles += to_pixels(SignalGrid.build(**v["tile_grid"]).lims)
        elif k.startswith("sky_o") and "grid" in v:
            objects += to_pixels(SignalGrid.build(**v["grid"]).lims)
    return tiles, objects


def _finite(arrays):
    """Finite values of a list of arrays (or of lists of arrays) as one flat array."""
    flat = [b for a in arrays for b in (a if isinstance(a, list) else [a])]
    return np.concatenate([np.asarray(b)[np.isfinite(b)].ravel() for b in flat] + [np.zeros(0)])


def _each(func, *arrs):
    """Apply ``func`` to arrays, or element-wise to lists of arrays (sub-grid panels)."""
    if isinstance(arrs[0], list):
        return [func(*z) for z in zip(*arrs, strict=True)]
    return func(*arrs)


def _norm(norm, vmin, vmax):
    return LogNorm(vmin=vmin, vmax=vmax) if norm == "log" else Normalize(vmin=vmin, vmax=vmax)


def _draw_overlay(ax, overlay):
    """Circles, boxes and contours on top of an image (pixel coordinates, x: axis 0)."""
    lw = overlay.get("lw", 0.5)
    for x, y, r in overlay.get("circles", []):
        ax.add_patch(Circle((x, y), r, fill=False, ec=overlay.get("circle_color", "cyan"), lw=lw))
    for x0, y0, w, h in overlay.get("boxes", []):
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=overlay.get("box_color", "white"), lw=lw))
    contours = overlay.get("contours", [])
    numbers = overlay.get("contour_numbers", False)
    color = overlay.get("contour_color", "white")
    for k, mask in enumerate(contours):
        mask = np.asarray(mask, dtype="float64")
        if not mask.any():
            continue
        ax.contour(mask.T, levels=[0.5], colors=[color], linewidths=lw, origin="lower")
        if numbers:
            # number centred above the highest point of the object (x: axis 0, y: axis 1)
            idx = np.argwhere(mask > 0.5)
            x = 0.5 * (idx[:, 0].min() + idx[:, 0].max())
            txt = ax.annotate(
                str(k), (x, idx[:, 1].max()), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", color=color, fontsize=overlay.get("number_size", 6),
            )
            txt.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="black")])


def plot_panels(
    groups,
    name=None,
    odir=None,
    step=None,
    step_label=True,
    rows=None,
    cols=None,
    fig_width=8.0,
    cbar_frac=0.025,
    space=0.01,
    label_color="white",
    label_size=9,
    dpi=300,
):
    """Plot groups of 2D images, every group sharing one horizontal colorbar below it.

    Every figure has the same width (``fig_width``) and the colorbar thickness and
    all gaps are fixed fractions of it, so plots with any number of panels look the
    same when shown at the same width (fonts have a fixed size, too).

    Parameters
    ----------
    groups : list of dict
        One dict per group with the keys

        - ``arrays``: list of 2D arrays (plotted transposed, origin lower, as ``plot_arrays``).
          An entry may also be a list of 2D arrays, drawn as a sub-grid inside its panel
          (as square as possible, e.g. 4x4 for 16 tiles); its overlay is then a list, too.
        - ``labels``: optional list of panel labels (drawn inside the top-left corner)
        - ``cmap``, ``norm`` ('log' or 'linear'), ``vmin``, ``vmax``: shared color mapping
          (limits default to the data range of the group)
        - ``cbar_label``: optional label of the group's colorbar
        - ``bad_color``: color of pixels without a value (NaN). Default is black.
        - ``cbar_ticks``: optional tick positions of the group's colorbar
        - ``cmap`` may also be a matplotlib Colormap
        - ``overlays``: optional list of dicts per panel with ``circles`` [(x, y, r)],
          ``boxes`` [(x0, y0, w, h)] and ``contours`` [2D masks] in pixels, and
          ``circle_color``, ``box_color``, ``contour_color`` (default white), ``lw``,
          ``contour_numbers`` (number each contour by its index) and ``number_size``

    name, odir : str, optional
        If both are given the figure is saved to ``odir/name``, otherwise shown.
    step : str, optional
        Pipeline step and iteration (e.g. 'reconstruct · it 1'), drawn left-aligned in a
        thin header strip above the panels (theme color with halo). Default is None.
    step_label : bool, optional
        Whether to draw the ``step`` header. Default is True.
    rows, cols : int, optional
        Number of panel rows or columns of a group (as in ``plot_arrays``). Default:
        as square as possible, with at least as many columns as rows.
    fig_width : float, optional
        Width of the figure in inches; the panel size follows from the number of
        columns. Default is 8.0.
    cbar_frac : float, optional
        Thickness of the colorbars as a fraction of the figure width. Default is 0.025.
    space : float, optional
        Gap between panels (and between panels and colorbar) as a fraction of the
        figure width. Default is 0.01.
    label_color : str, optional
        Color of the panel labels. Default is 'white'.
    label_size : float, optional
        Font size of the panel labels. Default is 9.
    dpi : int, optional
        Resolution of the saved figure. Default is 300.
    """
    layout = []
    for group in groups:
        arrays = [
            [np.asarray(b, dtype="float64") for b in a] if isinstance(a, list) else np.asarray(a, dtype="float64")
            for a in group["arrays"]
        ]
        if not arrays:
            continue
        if rows is None and cols is None:
            g_cols, g_rows = rows_and_cols(len(arrays))
        else:
            g_rows, g_cols = rows_and_cols(len(arrays), rows, cols)
        layout.append((group, arrays, g_cols, g_rows))
    if not layout:
        return
    n_cols = max(g[2] for g in layout)

    # layout in inches: panels of equal size filling the figure width, a colorbar of
    # fixed relative thickness below each group and room for its tick labels
    gap = space * fig_width
    cbar_h = cbar_frac * fig_width
    group_gap = 0.06 * fig_width
    panel = (fig_width - (n_cols - 1) * gap) / n_cols
    heights = [n_rows * panel + (n_rows - 1) * gap + gap + cbar_h for *_, n_rows in layout]
    header_h = 0.04 * fig_width if (step and step_label) else 0.0
    fig_h = header_h + sum(heights) + (len(layout) - 1) * group_gap
    figure = plt.figure(figsize=(fig_width, fig_h), dpi=dpi)
    if header_h:
        header = figure.text(
            0.0, 1 - 0.5 * header_h / fig_h, step, ha="left", va="center", fontsize=label_size + 1,
            color=plt.rcParams["text.color"],
        )
        header.set_path_effects(_halo(0.0))

    def add_axes(x, y, w, h):
        """Axes from a box in inches (y measured from the top)."""
        return figure.add_axes([x / fig_width, 1 - (y + h) / fig_h, w / fig_width, h / fig_h])

    top = header_h
    colorbars = []
    for (row, arrays, g_cols, _), height in zip(layout, heights, strict=True):
        norm = row.get("norm", "log")
        finite = _finite(arrays)
        vmin, vmax = row.get("vmin"), row.get("vmax")
        if norm == "log":
            pos = finite[finite > 0]
            vmin = vmin if vmin is not None else (float(pos.min()) if pos.size else 1e-10)
        else:
            vmin = vmin if vmin is not None else float(finite.min())
        vmax = vmax if vmax is not None else float(finite.max())
        color_norm = _norm(norm, vmin, vmax)
        labels = row.get("labels") or [None] * len(arrays)
        overlays = row.get("overlays") or [None] * len(arrays)

        cmap = plt.get_cmap(row.get("cmap", "inferno")).copy()
        cmap.set_bad(row.get("bad_color", "black"))

        def draw(arr, overlay, x, y, w, h, lw=0.5):
            ax = add_axes(x, y, w, h)
            data = arr.clip(vmin, None) if norm == "log" else arr  # noqa: B023
            img = ax.imshow(data.T, cmap=cmap, norm=color_norm, origin="lower", interpolation="nearest")  # noqa: B023
            if overlay:
                _draw_overlay(ax, overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(lw)
                spine.set_path_effects(_halo(lw))
            return ax, img

        def label(text, ax=None, x=0.0, y=0.0):
            """Panel label inside the top-left corner of ``ax`` or at (x, y) inches."""
            if not text:
                return
            kw = dict(ha="left", va="top", color=label_color, fontsize=label_size, zorder=10)
            if ax is not None:
                txt = ax.text(0.03, 0.97, text, transform=ax.transAxes, **kw)
            else:
                txt = figure.text(x / fig_width, 1 - y / fig_h, text, **kw)
            txt.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground="black")])

        img = None
        for k, arr in enumerate(arrays):
            x0, y0 = (k % g_cols) * (panel + gap), top + (k // g_cols) * (panel + gap)
            if isinstance(arr, list):
                # sub-grid of small panels (tiles, point sources), centered vertically in the panel
                if not arr:
                    continue
                s_cols, s_rows = rows_and_cols(len(arr))
                s_gap = 0.25 * gap
                sub = (panel - (s_cols - 1) * s_gap) / s_cols
                y0 += 0.5 * (panel - s_rows * sub - (s_rows - 1) * s_gap)
                subs = overlays[k] or [None] * len(arr)
                for j, (a, ov) in enumerate(zip(arr, subs, strict=True)):
                    x, y = x0 + (j % s_cols) * (sub + s_gap), y0 + (j // s_cols) * (sub + s_gap)
                    _, img = draw(a, ov, x, y, sub, sub, lw=0.3)
                label(labels[k], x=x0 + 0.03 * panel, y=y0 + 0.03 * panel)
            else:
                ax, img = draw(arr, overlays[k], x0, y0, panel, panel)
                label(labels[k], ax)

        # one colorbar spanning the columns used by this group
        width = g_cols * panel + (g_cols - 1) * gap
        cax = add_axes(0.0, top + height - cbar_h, width, cbar_h)
        cb = figure.colorbar(img, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.5)
        cb.outline.set_path_effects(_halo(0.5))
        if row.get("cbar_ticks") is not None:
            cb.set_ticks(row["cbar_ticks"])
        if row.get("cbar_label"):
            cb.set_label(row["cbar_label"])
        colorbars.append(cb)
        top += height + group_gap

    # halo around the colorbar ticks and tick labels (created when the figure is drawn)
    if _HALO["color"] is not None:
        figure.canvas.draw()
        for cb in colorbars:
            for line in cb.ax.xaxis.get_ticklines() + cb.ax.xaxis.get_ticklines(minor=True):
                line.set_path_effects(_halo(line.get_markeredgewidth()))
            for text in cb.ax.get_xticklabels() + [cb.ax.xaxis.label]:
                text.set_path_effects(_halo(0.0))

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if not name.endswith(".png"):
            name += ".png"
        figure.savefig(os.path.join(odir, name), bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(figure)


def split_style(style=None):
    """Split the ``base_style`` config section into its parts.

    Returns
    -------
    theme : dict
        Arguments of ``plot_style`` (theme, transparent, halo, halo_width).
    dynamic_range : float or None
        Dynamic range of the sky plots (vmax / vmin).
    panel_kwargs : dict
        Remaining settings, passed to ``plot_panels`` (step_label, rows, cols, fig_width, dpi, ...).
    """
    style = dict(style or {})
    theme = dict(
        theme=style.pop("theme", "dark"),
        transparent=style.pop("transparent", True),
        halo=style.pop("halo", True),
        halo_width=style.pop("halo_width", 1.0),
    )
    dynamic_range = style.pop("dynamic_range", 1e4)
    dynamic_range = float(dynamic_range) if dynamic_range is not None else None
    return theme, dynamic_range, style


def plot_components(
    models,
    samples,
    name=None,
    odir=None,
    dynamic_range=1e4,
    vmin=None,
    vmax=None,
    cmap="inferno",
    cbar_label=None,
    point_cutout=32,
    max_sub=100,
    **panel_kwargs,
):
    """Posterior means of the sky and its components with one shared colorbar.

    The first model is the full sky: its maximum sets ``vmax`` and
    ``vmin = vmax / dynamic_range`` (unless ``vmin``/``vmax`` are given).
    Tile components are shown as a sub-grid of their tiles, point components as
    a sub-grid of sky cutouts (``point_cutout`` pixels) around each point source.

    Parameters
    ----------
    models : list
        Full sky model followed by its components (each with ``prefix`` and ``grid``),
        e.g. ``[sky] + list(sky.models)``.
    samples : MySamples
        Posterior samples.
    name, odir : str, optional
        Output file name and directory.
    dynamic_range, vmin, vmax, cmap, cbar_label : optional
        Color mapping, see ``dynamic_limits``.
    point_cutout : int, optional
        Size of the point source cutouts in pixels. Default is 32.
    max_sub : int, optional
        Maximum number of tiles / point sources per component; only the brightest
        are shown (label 'N of M'). Default is 100.
    **panel_kwargs
        Passed to ``plot_panels``.
    """
    sky = _model_mean(models[0], samples)
    grid = getattr(models[0], "grid", None)
    means = [sky] + [_panel_image(md, samples, sky, point_cutout, grid=grid) for md in models[1:]]
    means, labels, _ = _capped(means, [md.prefix for md in models], max_sub)
    vmin, vmax = dynamic_limits(means[0], dynamic_range, vmin, vmax)
    row = dict(
        arrays=means,
        labels=labels,
        cmap=cmap,
        norm="log",
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
    )
    plot_panels([row], name=name, odir=odir, **panel_kwargs)



def _model_mean(model, samples):
    """Posterior mean of a model as a 2D image (central frequency for cubes)."""
    mean = np.squeeze(np.asarray(samples.mean(model), dtype="float64"))
    return mean[mean.shape[0] // 2] if mean.ndim == 3 else mean


def _point_pixels(model, grid=None):
    """Pixel positions (x: axis 0) of the point sources of a PointModel on ``grid`` (default: its grid)."""
    grid = grid if grid is not None else model.grid
    lims0 = np.asarray(grid.lims)[:, 0]
    coos = np.asarray(model.points.grid.coordinates, dtype="float64").reshape((-1, 2))
    return np.round((coos - lims0) * np.asarray(grid.fac, dtype="float64") - 0.5).astype(int)


def _cutouts(image, pixels, size):
    """Square cutouts of ``size`` pixels of an image, centered on the given pixels (NaN outside)."""
    h = size // 2
    pad = np.pad(np.asarray(image, dtype="float64"), h, constant_values=np.nan)
    return [pad[x : x + size, y : y + size] for x, y in pixels]


def _panel_image(model, samples, sky_image=None, point_cutout=32, positions=None, grid=None):
    """Image of one panel: the posterior mean of a model on its grid.

    Tile models give a list of their tiles (unmapped), point models a list of
    cutouts of ``sky_image`` (default: the point model itself on its output grid)
    around each point source, both drawn as sub-grids by ``plot_panels``. Point
    positions are taken from ``positions`` (a PointModel, default ``model``) on
    ``grid``, the grid of ``sky_image`` (default: the grid of the point model).
    """
    from ..model import PointModel, TileModel

    if isinstance(model, TileModel):
        tiles = np.asarray(samples.mean(lambda x: model(x, map=False)), dtype="float64")
        if model.tiles.grid.n_copies == 1:
            tiles = tiles[None]
        if tiles.ndim == 4:  # (tiles, freq, x, y): central frequency
            tiles = tiles[:, tiles.shape[1] // 2]
        return list(tiles)
    if isinstance(model, PointModel):
        image = sky_image if sky_image is not None else _model_mean(model, samples)
        return _cutouts(image, _point_pixels(positions or model, grid), point_cutout)
    return _model_mean(model, samples)


def _brightest(images, max_sub):
    """Indices of the (at most ``max_sub``) brightest sub-images of a sub-grid panel, in their order.

    Returns None for single images and for sub-grids within the limit.
    """
    if not isinstance(images, list) or max_sub is None or len(images) <= max_sub:
        return None
    peak = [np.nanmax(a) if np.isfinite(a).any() else -np.inf for a in images]
    return sorted(np.argsort(peak)[::-1][:max_sub].tolist())


def _select(images, idx):
    """Sub-images ``idx`` of a sub-grid panel (all if ``idx`` is None)."""
    return images if idx is None else [images[i] for i in idx]


def _capped(images, labels, max_sub):
    """Limit the sub-grid panels to their brightest ``max_sub`` images, noted in the labels."""
    sel = [_brightest(a, max_sub) for a in images]
    labels = [lb if i is None else f"{lb} · {len(i)} of {len(a)}" for lb, a, i in zip(labels, images, sel, strict=True)]
    return [_select(a, i) for a, i in zip(images, sel, strict=True)], labels, sel


def _spectral_maps(models, attr, samples, refs, cut, point_cutout=32):
    """Spectral maps (``attr``) of a sky, its background and its components.

    Components are shown where their reference brightness exceeds ``cut``, the
    background as it is. The sky map is composed of the components only: each
    pixel takes the value of the brightest component above ``cut`` (NaN elsewhere).
    """
    if len(models) < 3:  # no components: the models as they are
        return [_model_mean(getattr(md, attr), samples) for md in models]
    sky, bg, comps = models[0], models[1], models[2:]

    # components mapped onto the sky grid
    spec, ref = getattr(sky, attr), sky.ref_freq_model
    vals, bright = [], []
    for k in range(1, len(sky.models)):
        val = _model_mean(lambda x, m=spec.models[k]: m(x, map=True), samples)
        br = _model_mean(lambda x, m=ref.models[k]: m(x, map=True), samples)
        vals.append(val)
        bright.append(np.where(br > cut, br, -np.inf))
    if vals:
        bright, vals = np.stack(bright), np.stack(vals)
        best = np.argmax(bright, axis=0)
        sky_map = np.where(np.isfinite(bright.max(axis=0)), np.take_along_axis(vals, best[None], 0)[0], np.nan)
    else:
        sky_map = np.full_like(refs[0], np.nan)

    # background as it is, components above the cut (point sources: cutouts of the sky map)
    maps = [sky_map, _model_mean(getattr(bg, attr), samples)]
    for md, r in zip(comps, refs[2:], strict=True):
        val = _panel_image(getattr(md, attr), samples, sky_map, point_cutout, positions=md, grid=sky.grid)
        maps.append(_each(lambda r, v: np.where(r > cut, v, np.nan), r, val))
    return maps


def _alpha_cmap(vmin, vmax, center=-1.0):
    """RdYlBu_r with its center at the spectral index ``center`` (two-slope, as in the paper plots)."""
    f0 = float(np.clip((center - vmin) / (vmax - vmin), 0.01, 0.99)) if vmax > vmin else 0.5
    x = np.linspace(0.0, 1.0, 512)
    return ListedColormap(plt.get_cmap("RdYlBu_r")(np.where(x < f0, 0.5 * x / f0, 0.5 + 0.5 * (x - f0) / (1.0 - f0))))


def plot_kl_callback(
    skies, samples, nit, odir, dynamic_range=1e5, spectral_range=1e3, cmap="inferno", point_cutout=32, max_sub=100,
    **panel_kwargs,
):
    """Callback plots of an optimize_kl run (start_rec / extend_rec).

    For every sky model whose parameters are in the samples, one plot with the
    sky and all its components, each sharing one colorbar:

    - ``{nit}_{prefix}.png``: mean brightness (at the reference frequency for
      multi-frequency skies), log scale with ``dynamic_range`` relative to the sky
    - ``{nit}_{prefix}_alpha.png`` (multi-frequency): spectral index
    - ``{nit}_{prefix}_beta.png`` (multi-frequency with curvature): spectral curvature

    The spectral maps use the panels of the brightness plot. Components are left out
    (NaN, transparent) where their reference brightness is below ``max / spectral_range``
    (max: sky maximum), the sky panel is composed of the components (the brightest one
    above the cut per pixel), the background is shown as it is. They carry
    contours of it at 1e-3, 1e-2 and 1e-1 of the sky maximum. Colors follow the
    paper plots: spectral index on RdYlBu_r centered at -1 with limits at the 0.1 /
    99.9 percentiles, curvature on coolwarm symmetric at the 99th percentile of |beta|.

    Parameters
    ----------
    skies : list
        Sky models (e.g. the instantiated ``sky.N`` sections).
    samples : MySamples
        Current samples.
    nit : int
        Iteration number of the optimization (file names and step label).
    odir : str
        Output directory.
    dynamic_range : float, optional
        Ratio between the sky maximum and the lower color limit. Default is 1e5.
    spectral_range : float, optional
        Spectral maps are shown where the brightness exceeds the sky maximum divided
        by this. Default is 1e3.
    cmap : str, optional
        Colormap of the brightness plots. Default is 'inferno'.
    point_cutout : int, optional
        Size of the point source cutouts in pixels (tiles and point sources are
        drawn as sub-grids, see ``plot_components``). Default is 32.
    max_sub : int, optional
        Maximum number of tiles / point sources per component (the brightest at the
        reference frequency, the same in all three plots). Default is 100.
    **panel_kwargs
        Passed to ``plot_panels`` (dpi, fig_width, rows, cols, ...).
    """
    from ..optimize import domain_keys

    keys = domain_keys(samples)
    for sky in skies:
        if not domain_keys(sky).issubset(keys):
            continue
        models = [sky] + list(getattr(sky, "models", []))
        labels = [md.prefix for md in models]
        step = f"KL {nit} · {sky.prefix}"

        if np.asarray(sky.freq).size == 1:
            plot_components(
                models, samples, name=f"{nit}_{sky.prefix}.png", odir=odir, step=step,
                dynamic_range=dynamic_range, cmap=cmap, point_cutout=point_cutout, max_sub=max_sub,
                **panel_kwargs,
            )
            continue

        # brightness at the reference frequency
        sky_ref = _model_mean(sky.ref_freq_model, samples)
        refs = [sky_ref] + [
            _panel_image(md.ref_freq_model, samples, sky_ref, point_cutout, positions=md, grid=sky.grid)
            for md in models[1:]
        ]
        full_refs = refs
        refs, labels, sel = _capped(refs, labels, max_sub)
        vmin, vmax = dynamic_limits(refs[0], dynamic_range)
        group = dict(arrays=refs, labels=labels, cmap=cmap, vmin=vmin, vmax=vmax)
        plot_panels([group], name=f"{nit}_{sky.prefix}.png", odir=odir, step=f"{step} · reference frequency", **panel_kwargs)

        # spectral index and curvature in the same panels as the brightness: components where
        # the brightness exceeds max / spectral_range, the sky composed of them, the background
        # as it is; with contours of the brightness
        cut = vmax / float(spectral_range)
        overlays = [_each(lambda r: dict(contours=[r > vmax / d for d in (1e3, 1e2, 10)], lw=0.4), r) for r in refs]  # noqa: B023
        skip = {1} if len(models) > 2 else set()  # color limits without the background
        alphas = _spectral_maps(models, "spectral_index", samples, full_refs, cut, point_cutout)
        alphas = [_select(a, i) for a, i in zip(alphas, sel, strict=True)]
        finite = _finite([a for i, a in enumerate(alphas) if i not in skip])
        a_min, a_max = (float(v) for v in np.percentile(finite, [0.1, 99.9])) if finite.size else (-2.0, 0.0)
        group = dict(
            arrays=alphas, labels=labels, cmap=_alpha_cmap(a_min, a_max), norm="linear", vmin=a_min, vmax=a_max,
            overlays=overlays, cbar_label="spectral index", bad_color="none",
        )
        plot_panels([group], name=f"{nit}_{sky.prefix}_alpha.png", odir=odir, step=f"{step} · spectral index", **panel_kwargs)

        if getattr(sky, "has_spectral_curvature", False):
            betas = _spectral_maps(models, "spectral_curvature", samples, full_refs, cut, point_cutout)
            betas = [_select(b, i) for b, i in zip(betas, sel, strict=True)]
            finite = np.abs(_finite([b for i, b in enumerate(betas) if i not in skip]))
            lim = (float(np.percentile(finite, 99)) if finite.size else 0.0) or 1.0
            group = dict(
                arrays=betas, labels=labels, cmap="coolwarm", norm="linear", vmin=-lim, vmax=lim, overlays=overlays,
                cbar_label="spectral curvature", bad_color="none",
            )
            plot_panels([group], name=f"{nit}_{sky.prefix}_beta.png", odir=odir, step=f"{step} · spectral curvature", **panel_kwargs)

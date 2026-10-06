# %%
# ---------------------------------------------------------------------------
# Paper plots reproduced from the FITS products of `save_fits.py` alone (no
# model, config or pickled samples needed):
#   <PREFIX>_<nu>MHz_posterior_{mean,sample_<k>}.fits   sky brightness per frequency
#   <PREFIX>_spectral_index_posterior_{mean,sample_<k>}.fits  spectral index
# Each file holds the component layers BG, O0..On (cutouts), TILES, TILECUBE,
# TILEPOS and POINTS. Means come from the *_posterior_mean files, uncertainties
# from the spread over the *_posterior_sample_<k> files (ddof=1, as in the
# model-based scripts). The plotting code is copied from the model-based
# scripts (eso_components.py, eso_compare.py, eso_alpha_vs_flux.py,
# eso_c{1,2}_profiles.py, plot_rgb_freq.py), so the figures match theirs.
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# %%
# ---------------------------------------------------------------------------
# Imports.
# ---------------------------------------------------------------------------
import glob
from functools import lru_cache

import astropy.io.fits as pyfits
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import MaxNLocator, MultipleLocator, NullFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates as ndi_map_coordinates
from scipy.spatial import ConvexHull

import aim_resolve as aim
from aim_resolve.model.util import is_val, to_shape
from aim_resolve.plot.util import plot_figure

# %%
# ---------------------------------------------------------------------------
# Input / output.
# ---------------------------------------------------------------------------
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"
mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"

PREFIX = "eso137"
fits_dir = f"{dir}/opt/{mf_rec}/fits"
odir = f"{dir}/opt/{mf_rec}/paper_new"
paper_dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/paper"  # uSARA / AIRI FITS, regions.yml
REF_IDX = 1  # reference-frequency index (nu_ref = 1054 MHz)

os.makedirs(odir, exist_ok=True)

# %%
# ===========================================================================
# Infrastructure copied from the model-based scripts.
# ===========================================================================
# --- copied from eso_components.py ---
class SignalSpace():
    '''Class to represent a signal space at a specific location in the sky. Use `build` function to create the space.'''

    def __init__(self, shape, distances, center=(0., 0.), n_copies=1):
        self.shape = shape
        self.distances = distances
        self.center = center
        self.n_copies = n_copies

    def __repr__(self):
        return f'SignalSpace(shape={self.shape}, distances={self.distances}, center={self.center})'
    
    def __eq__(self, other):
        return isinstance(other, SignalSpace) and self.shape == other.shape and np.all(self.coos == other.coos)

    def __mul__(self, other):
        return self.multiply_shape(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @classmethod
    def build(cls, *, shape, distances=None, fov=None, center=None, n_copies=1):
        '''
        Build a SignalSpace from the given parameters.
        
        Parameters
        ----------
        shape : int or tuple
            The shape of the space
        distances : float or tuple, optional
            The distance between the pixels, by default None
        fov : float or tuple, optional
            The field of view of the space, by default None
        center : float or tuple, optional
            The center of the space, by default None
        n_copies : int, optional
            The number of copies of the space, by default 1
        '''
        shp = to_shape(shape, (2,), 'int64')
        dis = to_shape(distances, (2,), 'float64')
        fov = to_shape(fov, (2,), 'float64')
        cen = to_shape(center, (n_copies, 2), 'float64')

        shape = tuple(shp.tolist())

        if is_val(dis):
            distances = tuple(dis.tolist())
        elif is_val(fov):
            distances = tuple((fov / shp).tolist())
        else:
            distances = tuple((1 / shp).tolist())

        if not is_val(cen):
            cen = np.zeros_like(cen)
        center = tuple(map(tuple, cen.tolist()))

        if n_copies == 1:
            center = center[0]

        return cls(shape, distances, center, n_copies)

    @property
    def shp(self):
        return np.array(self.shape)
    
    @property
    def dis(self):
        return np.array(self.distances)

    @property
    def fov(self):
        return self.shp * self.dis
    
    @property
    def cen(self):
        return np.array(self.center)

    @property
    def coos(self):
        if self.n_copies == 1:
            return space_coos(self.shp, self.dis, self.cen)
        else:
            return vmap(space_coos, in_axes=(None, None, 0, 0))(self.shp, self.dis, self.cen)

    @property
    def lims(self):
        if self.n_copies == 1:
            return space_lims(self.fov, self.cen)
        else:
            return vmap(space_lims, in_axes=(None, 0))(self.fov, self.cen)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def coordinates(self):
        return self.coos

    @property
    def limits(self):
        return self.lims


def space_coos(shp, dis, cen):
    '''Generate the coordinates of the space.'''
    coos = jnp.indices(shp).astype(float)
    coos_T = coos.T.reshape(-1, 2)
    coos_T -= 0.5 * (shp - 1)
    coos_T *= dis
    coos_T += cen
    return coos_T.reshape(coos.T.shape).T


def space_lims(fov, cen):
    '''Generate the limits of the space.'''
    return fov[:,None] / 2 * np.array([-1, 1]) + cen[:,None]


def map_signal(x, in_space, out_space, order=0, vmap_sum=True):
    '''
    Map one or more signals from a SignalSpace to another SignalSpace.
    
    Parameters
    ----------
    x : np.ndarray
        The signal to be mapped
    in_space : SignalSpace
        The input space of the signal
    out_space : SignalSpace
        The output space of the signal
    order : int, optional
        The order of the interpolation, by default 0
    '''
    if x.ndim == 2:
        return map_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
    else:
        if in_space.n_copies > 1:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, 0, None, None))
            res = vmap_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)            
        else:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, None, None, None))
            res = vmap_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
        if vmap_sum:
            return jnp.sum(res, axis=0)
        else:
            return res


def map_one_signal(x, in_dis, in_cen, out_coos, order=0): 
    x = jnp.asarray(x)
    out_coos = jnp.asarray(out_coos)
    out_coos_T = out_coos.T.reshape(-1, 2)
    out_coos_T -= in_cen
    out_coos_T /= in_dis
    out_coos_T += 0.5 * (jnp.array(x.shape) - 1)
    out_coos = out_coos_T.reshape(out_coos.T.shape).T
    return map_coordinates(x, out_coos, order)


def fits2array(file_name):
    import astropy.io.fits as pyfits

    with pyfits.open(file_name) as hdulist:
        # print(hdulist.info())
        data = hdulist[0].data
    arr = np.asarray(data).T
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    return arr


def map2component(nifty_array, usara_array, airi_array, rel_fov, center):
    rel_fov = np.array(rel_fov)

    space_3k = SignalSpace.build(shape=nifty_array.shape, fov=("2deg", "2deg"))
    sp_c1_3k = SignalSpace.build(shape=space_3k.shp*rel_fov, fov=space_3k.fov*rel_fov, center=center)

    nifty_c1 = map_signal(nifty_array, space_3k, sp_c1_3k)
    print("NIFTy mapped shape:", nifty_c1.shape)

    space_4k = SignalSpace.build(shape=usara_array.shape, fov=("1.91deg", "1.91deg"))
    sp_c1_4k = SignalSpace.build(shape=sp_c1_3k.shp*4/3, fov=sp_c1_3k.fov, center=center)

    usara_c1 = map_signal(usara_array, space_4k, sp_c1_4k)
    print("uSARA mapped shape:", usara_c1.shape)

    airi_c1 = map_signal(airi_array, space_4k, sp_c1_4k)
    print("AIRIs mapped shape:", airi_c1.shape)\
    
    return nifty_c1, usara_c1, airi_c1


def compute_spectral_index(I_1, I_0, f_1, f_0):
    return np.log(I_1 / I_0) / np.log(f_1 / f_0)


def plot_tiles_grid(
    arrays,
    rows=6,
    cols=6,
    name=None,
    odir=None,
    cmap="inferno",
    norm="linear",
    vmin=None,
    vmax=None,
    frame=False,
    cbar_label=None,
    cbar_ticks=None,
    labels=None,
    label_color="white",
    label_fontsize=12,
    contour_arrays=None,
    contour_levels=None,
    tile_size=2.0,
    space=0.04,
    scale=1.0,
    dpi=300,
):
    """
    Plot tiles in a `rows` x `cols` grid sharing a single colorbar at the bottom.

    Parameters
    ----------
    arrays : iterable of np.ndarray
        The 2D tiles to plot (plotted row by row).
    rows, cols : int
        The number of rows and columns in the grid.
    name, odir : str, optional
        The filename and output directory for the saved figure.
    cmap, norm, vmin, vmax : optional
        Color mapping options shared by all tiles.
    frame : bool, optional
        Whether to draw a frame (border) around each tile. Default is False.
    cbar_label : str, optional
        The label of the shared colorbar.
    cbar_ticks : iterable of float, optional
        Tick positions of the shared colorbar. Default is matplotlib's choice.
    labels : iterable of str, optional
        Per-tile text drawn in the top-left corner (use None to skip a tile).
        Default is None.
    label_color : str, optional
        Color of the per-tile labels. Default is "white".
    label_fontsize : int, optional
        Font size of the per-tile labels. Default is 12.
    contour_arrays : iterable of np.ndarray, optional
        Per-tile arrays from which to draw white contours (e.g. the flux tiles).
        Must align with `arrays`. Default is None.
    contour_levels : iterable of float, optional
        The contour levels to draw. Default is None.
    tile_size : float, optional
        The size (in inches) of a single tile. Default is 2.0.
    space : float, optional
        The spacing between tiles, equal in x and y (fraction of a tile).
        Default is 0.04.
    scale : float, optional
        Uniformly scales every figure dimension (in inches). Larger values make
        the fixed point-size labels and colorbar ticks relatively smaller.
        Default is 1.0.
    dpi : int, optional
        The dpi of the figure. Default is 300.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    norm_obj = (
        LogNorm(vmin=vmin, vmax=vmax)
        if norm == "log"
        else Normalize(vmin=vmin, vmax=vmax)
    )

    # Layout (in inches): reserve a margin on the sides/top and a strip at the
    # bottom for the shared colorbar. The figure height is chosen so that each
    # cell is square -> a single `space` gives equal gaps in x and y. `scale`
    # grows every inch dimension uniformly, shrinking text relative to the plot.
    tile_size = tile_size * scale
    margin_x = 0.02 * tile_size * cols
    margin_top = 0.1 * scale
    cbar_strip = 0.95 * scale
    cbar_height = 0.28 * scale

    fig_w = tile_size * cols
    grid_w = fig_w - 2 * margin_x
    cell_w = grid_w / (cols + (cols - 1) * space)
    grid_h = cell_w * (rows + (rows - 1) * space)
    fig_h = grid_h + margin_top + cbar_strip

    # Gap between the grid and the colorbar = the absolute gap between subplots
    # (square cells, so `space` is the same fraction of width and height).
    cbar_gap = space * cell_w

    left = margin_x / fig_w
    right = 1 - margin_x / fig_w
    bottom = cbar_strip / fig_h
    top = 1 - margin_top / fig_h

    figure, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    figure.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=space, hspace=space
    )
    axes = np.atleast_1d(axes).ravel()

    img = None
    for i, ax in enumerate(axes):
        if i < len(arrays):
            img = ax.imshow(
                np.asarray(arrays[i], dtype="float64").T,
                cmap=cmap,
                norm=norm_obj,
                origin="lower",
            )
            if contour_arrays is not None and contour_levels is not None:
                ax.contour(
                    np.asarray(contour_arrays[i], dtype="float64").T,
                    levels=contour_levels,
                    colors="black",
                    linewidths=0.5,
                    origin="lower",
                )
            if labels is not None and i < len(labels) and labels[i]:
                ax.text(
                    0.05, 0.93, labels[i],
                    transform=ax.transAxes, ha="left", va="top",
                    color=label_color, fontsize=label_fontsize,
                )
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if frame:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Single colorbar spanning the full width of the grid.
    cax = figure.add_axes(
        [
            left,
            (cbar_strip - cbar_gap - cbar_height) / fig_h,
            right - left,
            cbar_height / fig_h,
        ]
    )
    cbar = figure.colorbar(img, cax=cax, orientation="horizontal")
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    if cbar_label:
        cbar.set_label(cbar_label)

    plot_figure(figure, odir, name)


def plot_column(
    arrays,
    *,
    odir=None,
    name=None,
    cmap="inferno",
    norm="log",
    vmin=None,
    vmax=None,
    cbar=True,
    cbar_label=None,
    cbar_ticks=None,
    cbar_loc="right",
    cbar_width=0.0225,
    labels=None,
    label_color="white",
    text=None,
    text_color=None,
    marker=None,
    frame=False,
    contour=None,
    origin="lower",
    fig_width=5.0,
    hspace=0.025,
    label_offset=-12,
    dpi=300,
):
    """
    Plot a list of 2D arrays stacked in a single column, all with the same
    width in x, sharing one colorbar on the right.

    Mirrors the styling of the `eso_compare` plots: every image fills the same
    figure width (heights follow each image's own aspect ratio) and a single
    shared colorbar spans the combined height of the stack.

    Parameters
    ----------
    arrays : iterable of np.ndarray
        The 2D arrays to plot, one per row.
    odir, name : str, optional
        If both are given the figure is saved to `odir/name`, otherwise shown.
    cmap, norm, vmin, vmax, origin, dpi : optional
        Color mapping / display options shared by all images.
    cbar : bool, optional
        Whether to draw the shared colorbar. Default is True.
    cbar_label : str, optional
        Label drawn alongside the shared colorbar. Default is None.
    cbar_ticks : Iterable of float, optional
        Tick positions of the shared colorbar. Default is matplotlib's choice.
    cbar_loc : str, optional
        "right" (default) or "left".
    cbar_width : float, optional
        Width of the colorbar in figure fractions. Default is 0.0225.
    labels : iterable of str, optional
        Per-row text drawn in the top-right corner (None to skip a row).
    label_color : str, optional
        Color of the per-row corner labels. Default is "white".
    text : str or iterable of str, optional
        Per-row text drawn in the top-LEFT corner (a single str applies to the
        first row). Default is None.
    text_color : str, optional
        Color of the top-left text. Defaults to `label_color`.
    marker : dict or list of dict, optional
        Scatter markers. A single spec ({'x','y',...} or {'m0': {...}, ...}) is
        drawn on every row; a list gives one spec per row (None to skip). Each
        leaf dict is passed straight to `ax.scatter(**dict)`. Default is None.
    frame : bool, optional
        If True, draw a black box around each image (no ticks). Default is False.
    contour : dict or list of dict, optional
        Contour spec(s) passed to `ax.contour`. A single dict applies to every
        row, a list gives one spec per row (None to skip). The optional "array"
        key selects the field the contours are drawn from. Default is None.
    fig_width : float, optional
        Width of the figure in inches. Default is 5.0.
    hspace : float, optional
        Vertical gap between images (fraction of the mean image height).
    dpi : int, optional
        The dpi of the figure. Default is 300.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    arrays = [np.array(a, dtype="float64") for a in arrays]
    n = len(arrays)

    if contour is None:
        contours = [None] * n
    elif isinstance(contour, dict):
        contours = [contour] * n
    else:
        contours = list(contour) + [None] * (n - len(contour))

    row_labels = ([None] * n) if labels is None else list(labels) + [None] * (n - len(labels))

    if text is None:
        texts = [None] * n
    elif isinstance(text, str):
        texts = [text] + [None] * (n - 1)
    else:
        texts = list(text) + [None] * (n - len(text))

    if marker is None:
        markers = [None] * n
    elif isinstance(marker, dict):
        markers = [marker] * n
    else:
        markers = list(marker) + [None] * (n - len(marker))

    text_color = label_color if text_color is None else text_color

    # Shared color limits so the single colorbar is valid for every image.
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if norm == "log":
        pos = finite[finite > 0]
        vmin = (pos.min() / 100 if pos.size else 1.0) if vmin is None else vmin
        color_norm = LogNorm(vmin=vmin, vmax=finite.max() if vmax is None else vmax)
        arrays = [a.clip(vmin, None) for a in arrays]
    else:
        vmin = finite.min() if vmin is None else vmin
        color_norm = Normalize(vmin=vmin, vmax=finite.max() if vmax is None else vmax)

    hspace = hspace if hspace > 0 else 0.025

    # Same width in x for every image regardless of its pixel count: a single
    # column gives every cell the same width, and `height_ratios` set to each
    # image's aspect ratio gives every cell the matching height, so pixels stay
    # square. `aspect="auto"` then makes each image fill its whole cell.
    aspects = [a.shape[1] / a.shape[0] for a in arrays]
    fig_h = fig_width * sum(aspects) * (1 + hspace)
    figure, axes = plt.subplots(
        n,
        1,
        figsize=(fig_width, fig_h),
        dpi=dpi,
        gridspec_kw={"hspace": hspace, "height_ratios": aspects},
    )
    axes = np.atleast_1d(axes).ravel().tolist()

    img = None
    for ax, a, c, lab, txt, mrk in zip(
        axes, arrays, contours, row_labels, texts, markers
    ):
        img = ax.imshow(a.T, cmap=cmap, norm=color_norm, origin=origin, aspect="auto")

        if c:
            c = dict(c)
            c_arr = np.asarray(c.pop("array", a), dtype="float64")
            ax.contour(c_arr.T, origin="lower", **c)

        if mrk:
            # accept a single {x, y, ...} spec or a dict of such subdicts.
            specs = {"m0": mrk} if all(k in mrk for k in ("x", "y")) else mrk
            for spec in specs.values():
                ax.scatter(**spec)

        # Corner labels sit a fixed distance below the image top (offset in
        # points), so the margin is the same regardless of the image height.
        if lab:
            ax.annotate(
                lab, xy=(0.97, 1.0), xycoords="axes fraction",
                xytext=(0, label_offset), textcoords="offset points",
                ha="right", va="top", color=label_color,
            )

        if txt:
            ax.annotate(
                txt, xy=(0.03, 1.0), xycoords="axes fraction",
                xytext=(0, label_offset), textcoords="offset points",
                ha="left", va="top", color=text_color,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(frame)
            if frame:
                spine.set_color("black")
                spine.set_linewidth(0.8)

    # Single colorbar spanning exactly the combined height of the stack.
    if cbar:
        figure.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        top = max(b.y1 for b in boxes)
        bottom = min(b.y0 for b in boxes)
        # Fixed image-to-colorbar gap (figure-width fraction) so every column
        # plot -- single image or multi-row -- has the exact same spacing.
        pad = 0.011

        if cbar_loc == "left":
            x0 = min(b.x0 for b in boxes) - pad - cbar_width
        else:
            x0 = max(b.x1 for b in boxes) + pad
        cax = figure.add_axes([x0, bottom, cbar_width, top - bottom])
        cb = figure.colorbar(img, cax=cax)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        if cbar_label:
            cb.set_label(cbar_label)

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name), bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def crop_component(nifty_array, rel_fov, center):
    """
    Crop a full-grid (2deg) nifty image to a component sub-field of view.

    Works for a single 2D image as well as a stack of images (e.g. one per
    frequency): the mapping acts on the last two dimensions and each leading
    slice is cropped independently.
    """
    nifty_array = np.asarray(nifty_array)
    rel_fov = np.array(rel_fov)
    space = SignalSpace.build(shape=nifty_array.shape[-2:], fov=("2deg", "2deg"))
    sub = SignalSpace.build(
        shape=space.shp * rel_fov, fov=space.fov * rel_fov, center=center
    )
    return map_signal(nifty_array, space, sub, vmap_sum=False)


def curvature_from_cube(cube, u_freq):
    """
    Per-pixel spectral curvature of a multi-frequency cube. A log-parabola is
    fit per pixel,  ln S = a + alpha * u + c * u^2  with  u = ln(nu / nu_ref),
    and the quadratic coefficient `c` is returned:
        c > 0  flattening to high nu -> blend / superposition of two spectra,
        c < 0  steepening / break    -> single, connected, aged population.
    `cube` is (nfreq, nx, ny); needs >= 3 frequencies.
    """
    ln = np.log(np.clip(cube, 1e-12, None))
    c = np.polyfit(u_freq, ln.reshape(cube.shape[0], -1), 2)[0]
    return c.reshape(cube.shape[1:])


# --- copied from eso_compare.py ---
def plot_rows(
    array,
    *,
    odir=None,
    name=None,
    cmap="inferno",
    norm="log",
    vmin=None,
    vmax=None,
    cbar=True,
    cbar_label=None,
    cbar_ticks=None,
    cbar_kwargs=None,
    contour=None,
    labels=None,
    label_color="white",
    ticks=0,
    frame=False,
    origin="lower",
    figsize=(5, 5),
    dpi=300,
    grid_kwargs=None,
):
    '''
    Plot a list of 2D arrays stacked below each other in a single column,
    sharing one colorbar on the right.

    Mirrors the styling of `aim.plot_arrays` (cmap, log/linear norm, shared
    vmin/vmax, contours, ticks, origin, dpi, gridspec spacing), but instead of
    giving every sub-plot its own colorbar it attaches a single shared one.

    Parameters
    ----------
    array : Iterable of np.ndarray
        The 2D arrays to plot, one per row.
    odir, name : str, optional
        If both are given the figure is saved to `odir/name`, otherwise shown.
    cmap, norm, vmin, vmax, ticks, origin, dpi : see `aim.plot_arrays`.
    frame : bool, optional
        If True, draw a black box (spines) around each image with no ticks,
        so the extent of every sub-plot is visible. If False and ``ticks <= 0``
        the axes are turned off entirely. Default is False.
    labels : Iterable of str, optional
        Per-row text drawn in the top-right corner of each image (use None to
        skip a row). Default is None.
    label_color : str, optional
        Color of the per-row corner labels. Default is "white".
    cbar : bool, optional
        Whether to draw the shared colorbar. Default is True.
    cbar_label : str, optional
        Label drawn alongside the shared colorbar. Default is None.
    cbar_ticks : Iterable of float, optional
        Tick positions of the shared colorbar. Default is matplotlib's choice.
    cbar_kwargs : dict, optional
        Keyword arguments for the colorbar. `loc` ("right"/"left"/"top"/
        "bottom"), `fraction` and `pad` are recognised. Default is {}.
    contour : dict or list of dict, optional
        Contour spec(s) passed to `ax.contour`. A single dict is applied to
        every row, a list provides one spec per row (use None to skip a row).
        The optional "array" key selects the field the contours are drawn from
        (defaults to the plotted array). Default is None.
    grid_kwargs : dict, optional
        Keyword arguments passed to the GridSpec (e.g. `hspace`). Default is {}.
    '''
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize

    if cbar_kwargs is None:
        cbar_kwargs = {}
    if grid_kwargs is None:
        grid_kwargs = {}

    arrays = [np.array(a, dtype="float64") for a in array]
    rows = len(arrays)

    if contour is None:
        contours = [None] * rows
    elif isinstance(contour, dict):
        contours = [contour] * rows
    else:
        contours = list(contour) + [None] * (rows - len(contour))

    if labels is None:
        row_labels = [None] * rows
    else:
        row_labels = list(labels) + [None] * (rows - len(labels))

    # Shared color limits across every sub-plot so the single colorbar is valid.
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if norm == "log":
        pos = finite[finite > 0]
        auto_min = pos.min() / 100 if pos.size else 1.0
    else:
        auto_min = finite.min()
    vmin = auto_min if vmin is None else vmin
    vmax = finite.max() if vmax is None else vmax

    if norm == "log":
        color_norm = LogNorm(vmin=vmin, vmax=vmax)
        arrays = [a.clip(vmin, None) for a in arrays]
    else:
        color_norm = Normalize(vmin=vmin, vmax=vmax)

    # Small positive gap between rows. The negative `hspace` values that packed
    # the old per-image colorbars together make the images overlap here, so we
    # clamp to a small positive default.
    grid_kwargs = dict(grid_kwargs)
    grid_kwargs.pop("wspace", None)
    hspace = grid_kwargs.pop("hspace", 0.025)
    if hspace <= 0:
        hspace = 0.025

    # Size the figure from the (shared) image aspect so the stack stays compact.
    aspect = arrays[0].shape[1] / arrays[0].shape[0]  # displayed height / width
    fig_w = float(figsize[0])
    fig_h = fig_w * aspect * rows * (1 + hspace)
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        gridspec_kw={"hspace": hspace, **grid_kwargs},
    )
    axes = np.atleast_1d(axes).ravel().tolist()

    img = None
    for ax, a, c, lab in zip(axes, arrays, contours, row_labels):
        # `aspect="auto"` lets the image fill the whole axes box, while
        # `set_box_aspect` gives that box the image's own height/width ratio.
        # Together the pixels stay square (no distortion) and the box exactly
        # bounds the image (no internal whitespace), so the colorbar below can
        # match the image extent precisely.
        img = ax.imshow(a.T, cmap=cmap, norm=color_norm, origin=origin, aspect="auto")
        ax.set_box_aspect(a.shape[1] / a.shape[0])

        if c:
            c = dict(c)
            c_arr = np.asarray(c.pop("array", a), dtype="float64")
            ax.contour(c_arr.T, origin="lower", **c)

        if lab:
            # Fixed margin (points) below the image top, independent of height.
            ax.annotate(
                lab, xy=(0.97, 1.0), xycoords="axes fraction",
                xytext=(0, -12), textcoords="offset points",
                ha="right", va="top", color=label_color,
            )

        if frame:
            # Keep a black box around the image but drop all ticks/labels.
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.8)
        elif ticks <= 0:
            ax.axis("off")

    # Single colorbar spanning exactly the combined height of the stacked
    # images. Draw first so the axes positions reflect the box aspects set above.
    if cbar:
        figure.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        top = max(b.y1 for b in boxes)
        bottom = min(b.y0 for b in boxes)
        width = cbar_kwargs.get("fraction", 0.0225)

        # Fixed image-to-colorbar gap (figure-width fraction), matching the
        # `plot_column` plots in eso_components.py; `cbar_kwargs["pad"]` overrides.
        pad = cbar_kwargs.get("pad", 0.011)

        if cbar_kwargs.get("loc", "right") == "left":
            x0 = min(b.x0 for b in boxes) - pad - width
        else:
            x0 = max(b.x1 for b in boxes) + pad
        cax = figure.add_axes([x0, bottom, width, top - bottom])
        cb = figure.colorbar(img, cax=cax)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        if cbar_label:
            cb.set_label(cbar_label)

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name), bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# --- copied from plot_rgb_freq.py ---
_XYZ_CMF = np.array(
    [[
        0.000160, 0.000662, 0.002362, 0.007242, 0.019110, 0.043400,
        0.084736, 0.140638, 0.204492, 0.264737, 0.314679, 0.357719,
        0.383734, 0.386726, 0.370702, 0.342957, 0.302273, 0.254085,
        0.195618, 0.132349, 0.080507, 0.041072, 0.016172, 0.005132,
        0.003816, 0.015444, 0.037465, 0.071358, 0.117749, 0.172953,
        0.236491, 0.304213, 0.376772, 0.451584, 0.529826, 0.616053,
        0.705224, 0.793832, 0.878655, 0.951162, 1.014160, 1.074300,
        1.118520, 1.134300, 1.123990, 1.089100, 1.030480, 0.950740,
        0.856297, 0.754930, 0.647467, 0.535110, 0.431567, 0.343690,
        0.268329, 0.204300, 0.152568, 0.112210, 0.081261, 0.057930,
        0.040851, 0.028623, 0.019941, 0.013842, 0.009577, 0.006605,
        0.004553, 0.003145, 0.002175, 0.001506, 0.001045, 0.000727,
        0.000508, 0.000356, 0.000251, 0.000178, 0.000126, 0.000090,
        0.000065, 0.000046, 0.000033,
    ],
     [
         0.000017, 0.000072, 0.000253, 0.000769, 0.002004, 0.004509,
         0.008756, 0.014456, 0.021391, 0.029497, 0.038676, 0.049602,
         0.062077, 0.074704, 0.089456, 0.106256, 0.128201, 0.152761,
         0.185190, 0.219940, 0.253589, 0.297665, 0.339133, 0.395379,
         0.460777, 0.531360, 0.606741, 0.685660, 0.761757, 0.823330,
         0.875211, 0.923810, 0.961988, 0.982200, 0.991761, 0.999110,
         0.997340, 0.982380, 0.955552, 0.915175, 0.868934, 0.825623,
         0.777405, 0.720353, 0.658341, 0.593878, 0.527963, 0.461834,
         0.398057, 0.339554, 0.283493, 0.228254, 0.179828, 0.140211,
         0.107633, 0.081187, 0.060281, 0.044096, 0.031800, 0.022602,
         0.015905, 0.011130, 0.007749, 0.005375, 0.003718, 0.002565,
         0.001768, 0.001222, 0.000846, 0.000586, 0.000407, 0.000284,
         0.000199, 0.000140, 0.000098, 0.000070, 0.000050, 0.000036,
         0.000025, 0.000018, 0.000013,
     ],
     [
         0.000705, 0.002928, 0.010482, 0.032344, 0.086011, 0.197120,
         0.389366, 0.656760, 0.972542, 1.282500, 1.553480, 1.798500,
         1.967280, 2.027300, 1.994800, 1.900700, 1.745370, 1.554900,
         1.317560, 1.030200, 0.772125, 0.570060, 0.415254, 0.302356,
         0.218502, 0.159249, 0.112044, 0.082248, 0.060709, 0.043050,
         0.030451, 0.020584, 0.013676, 0.007918, 0.003988, 0.001091,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000,
     ]],
    dtype=np.float64,
)


_MATRIX_SRGB_D65 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


_CMF_WAVELENGTHS_NM = np.linspace(380.0, 780.0, _XYZ_CMF.shape[1], dtype=np.float64)


def _gamma_corr(inp):
    mask = np.zeros(inp.shape, dtype=np.float64)
    mask[inp <= 0.0031308] = 1.0
    r1 = 12.92 * inp
    a = 0.055
    r2 = (1 + a) * (np.maximum(inp, 0.0031308) ** (1 / 2.4)) - a
    return r1 * mask + r2 * (1.0 - mask)


def _xyz_from_wavelengths(wavelengths_nm):
    lam = np.clip(wavelengths_nm, _CMF_WAVELENGTHS_NM[0], _CMF_WAVELENGTHS_NM[-1])
    x = np.interp(lam, _CMF_WAVELENGTHS_NM, _XYZ_CMF[0])
    y = np.interp(lam, _CMF_WAVELENGTHS_NM, _XYZ_CMF[1])
    z = np.interp(lam, _CMF_WAVELENGTHS_NM, _XYZ_CMF[2])
    return np.stack([x, y, z], axis=0)


def _integration_weights(axis_values):
    if axis_values.ndim != 1:
        raise ValueError("axis_values must be one-dimensional")
    if axis_values.size == 1:
        return np.ones(1, dtype=np.float64)
    delta = np.abs(np.diff(axis_values))
    weights = np.empty(axis_values.size, dtype=np.float64)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    if axis_values.size > 2:
        weights[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    normalizer = weights.sum()
    if normalizer > 0:
        weights /= normalizer
    return weights


def _to_logscale(arr, lo, hi):
    arr = np.asarray(arr, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    eps = np.finfo(np.float64).tiny
    lo = np.maximum(lo, eps)
    hi = np.maximum(hi, lo * (1.0 + 1e-12))
    clipped = arr.clip(lo, hi)
    return np.log(clipped / lo) / np.log(hi / lo)


def _xyz_to_srgb(xyz_data):
    rgb_linear = xyz_data @ _MATRIX_SRGB_D65.T
    return _gamma_corr(rgb_linear).clip(0.0, 1.0)


def build_visible_wavelength_axis(nu_coords, *, axis_scale, lambda_min=400.0, lambda_max=700.0):
    """Map frequency coordinates onto the visible band (low freq -> red, high -> blue)."""
    nu_arr = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
    if nu_arr.size < 1:
        raise ValueError("nu_coords must be non-empty")

    axis_scale_applied = axis_scale
    if axis_scale == "log":
        if np.any(nu_arr <= 0) or np.any(~np.isfinite(nu_arr)):
            scaled = nu_arr
            axis_scale_applied = "linear"
        else:
            scaled = np.log10(nu_arr)
    else:
        scaled = nu_arr

    finite = np.isfinite(scaled)
    if not np.any(finite):
        t = np.linspace(0.0, 1.0, nu_arr.size, dtype=np.float64)
    else:
        lo = float(np.min(scaled[finite]))
        hi = float(np.max(scaled[finite]))
        if hi <= lo:
            t = np.linspace(0.0, 1.0, nu_arr.size, dtype=np.float64)
        else:
            t = (scaled - lo) / (hi - lo)
            t = np.where(np.isfinite(t), t, 0.0)

    # Low frequency maps to red (longer wavelengths), high frequency to blue.
    wavelength_nm = lambda_max - t * (lambda_max - lambda_min)
    return wavelength_nm.astype(np.float64), axis_scale_applied


def convert_mf_to_rgb_new(
    spectral_cube,
    *,
    wavelength_axis_nm,
    intensity_scale="log",
    clip_min=0.0,
    clip_max=1.0,
    dynamic_range=2.5e3,
    after_log_gammacorr=None,
    reuse_brightness_scale=False,
    channel_relative_clip=False,
    channel_clip_reference=None,
):
    """Integrate a spectral cube (freq axis LAST) against the CIE CMFs -> sRGB."""
    shp = spectral_cube.shape[:-1] + (3,)
    n_freqs = spectral_cube.shape[-1]
    spectral_cube = np.asarray(spectral_cube, dtype=np.float64).reshape((-1, n_freqs))
    wavelength_axis_nm = np.asarray(wavelength_axis_nm, dtype=np.float64).reshape(-1)
    if wavelength_axis_nm.size != n_freqs:
        raise ValueError("wavelength_axis_nm must have shape (n_freqs,)")

    if reuse_brightness_scale:
        maxval = float(reuse_brightness_scale)
    else:
        finite = np.isfinite(spectral_cube)
        if np.any(finite):
            maxval = float(np.max(spectral_cube[finite]))
        else:
            maxval = 1.0
    if not np.isfinite(maxval) or maxval <= 0:
        maxval = 1.0

    clip_min = float(np.clip(clip_min, 0.0, 1.0))
    clip_max = float(np.clip(clip_max, 0.0, 1.0))
    if clip_max <= clip_min:
        clip_max = min(1.0, clip_min + 1.0e-3)

    if channel_relative_clip:
        if channel_clip_reference is not None:
            ref = np.asarray(channel_clip_reference, dtype=np.float64).reshape((-1, n_freqs))
            if ref.shape != spectral_cube.shape:
                raise ValueError("channel_clip_reference must have the same shape as spectral_cube")
        else:
            ref = spectral_cube
        finite = np.isfinite(ref)
        mostly_nonnegative = False
        if np.any(finite):
            negative_fraction = float(np.mean(ref[finite] < 0.0))
            mostly_nonnegative = negative_fraction <= 0.25
        valid = np.any(finite, axis=0)
        mins = np.where(finite, ref, np.inf).min(axis=0)
        maxs = np.where(finite, ref, -np.inf).max(axis=0)
        mins = np.where(valid, mins, 0.0)
        maxs = np.where(valid, maxs, mins + 1.0)
        span = np.maximum(maxs - mins, 1.0e-12)
        lo = mins + clip_min * span
        hi = mins + clip_max * span
        hi = np.maximum(hi, lo + 1.0e-12)
        lo = lo[np.newaxis, :]
        hi = hi[np.newaxis, :]
        if mostly_nonnegative:
            lo = np.maximum(lo, 0.0)
            hi = np.maximum(hi, lo + 1.0e-12)

        finite_cur = np.isfinite(spectral_cube)
        cur_mins = np.where(finite_cur, spectral_cube, np.inf).min(axis=0, keepdims=True)
        cur_maxs = np.where(finite_cur, spectral_cube, -np.inf).max(axis=0, keepdims=True)
        valid_cur = np.any(finite_cur, axis=0, keepdims=True)
        if clip_min <= 0.0:
            if mostly_nonnegative:
                cur_floor = np.maximum(cur_mins, 0.0)
                lo = np.where(valid_cur, np.minimum(lo, cur_floor), lo)
                lo = np.maximum(lo, 0.0)
            else:
                lo = np.where(valid_cur, np.minimum(lo, cur_mins), lo)
        if clip_max >= 1.0:
            hi = np.where(valid_cur, np.maximum(hi, cur_maxs), hi)
        hi = np.maximum(hi, lo + 1.0e-12)

        clipped = np.clip(spectral_cube, lo, hi)
        rel = np.maximum(clipped - lo, 0.0)
        span = np.maximum(hi - lo, 1.0e-12)
        span_global = float(np.max(span))
        if not np.isfinite(span_global) or span_global <= 0.0:
            span_global = 1.0
        channel_gain = span / span_global

        if intensity_scale == "log":
            floor = np.maximum(span / max(dynamic_range, 1.0 + 1.0e-12), np.finfo(np.float64).tiny)
            denom = np.log1p(span / floor)
            spectral_norm = np.log1p(rel / floor) / np.maximum(denom, 1.0e-12)
            spectral_norm = spectral_norm * channel_gain
        elif intensity_scale == "sqrt":
            spectral_norm = np.sqrt(rel / span) * channel_gain
        else:
            spectral_norm = rel / span_global
    else:
        hi = maxval * clip_max
        if intensity_scale == "log":
            if clip_min > 0.0:
                lo = maxval * clip_min
            else:
                lo = hi / max(dynamic_range, 1.0 + 1e-12)
            spectral_norm = _to_logscale(spectral_cube, hi=hi, lo=lo)
        else:
            lo = maxval * clip_min
            lo = max(lo, 0.0)
            hi = max(hi, lo + 1.0e-12)
            spectral_norm = np.clip((spectral_cube - lo) / (hi - lo), 0.0, 1.0)
            if intensity_scale == "sqrt":
                spectral_norm = np.sqrt(spectral_norm)
    if after_log_gammacorr is not None:
        spectral_norm = np.float_power(spectral_norm, after_log_gammacorr)

    xyz_response = _xyz_from_wavelengths(wavelength_axis_nm)
    weights = _integration_weights(wavelength_axis_nm)
    weighted_response = xyz_response * weights[np.newaxis, :]
    xyz_data = np.tensordot(spectral_norm, weighted_response, axes=[-1, -1])
    rgb_data = _xyz_to_srgb(xyz_data)
    return rgb_data.reshape(shp), maxval


def normalize_total_flux_brightness(total_flux, *, intensity_mode, clip_min, clip_max, dynamic_range=2.5e3, maxval=None):
    arr = np.asarray(total_flux, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return np.zeros_like(arr, dtype=np.float64)

    # `maxval` (the total flux mapped to full brightness) defaults to this
    # image's own max; pass a shared reference to match brightness across images.
    maxval = float(np.max(arr[finite])) if maxval is None else float(maxval)
    if not np.isfinite(maxval) or maxval <= 0:
        return np.zeros_like(arr, dtype=np.float64)

    hi = maxval * clip_max
    if intensity_mode == "log":
        if clip_min > 0.0:
            lo = maxval * clip_min
        else:
            lo = hi / max(dynamic_range, 1.0 + 1e-12)
        lo = max(lo, np.finfo(np.float64).tiny)
        hi = max(hi, lo * (1.0 + 1e-12))
        clipped = np.clip(arr, lo, hi)
        return np.log(clipped / lo) / np.log(hi / lo)

    lo = max(maxval * clip_min, 0.0)
    hi = max(hi, lo + 1.0e-12)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if intensity_mode == "sqrt":
        norm = np.sqrt(norm)
    return norm


def normalize_spectrum(arr, *, normalize_boost, ref_spectrum=None):
    arr64 = np.asarray(arr, dtype=np.float64)
    if ref_spectrum is None:
        mean_spectrum = np.mean(arr64, axis=(1, 2), dtype=np.float64)
    else:
        # Fixed per-channel reference (e.g. the full-sky mean) so the hue
        # normalization is shared across images instead of being per-image.
        mean_spectrum = np.asarray(ref_spectrum, dtype=np.float64).reshape(-1)
    finite = np.isfinite(mean_spectrum)
    if bool(np.any(finite)):
        median_abs = float(np.median(np.abs(mean_spectrum[finite])))
    else:
        median_abs = 1.0
    if not np.isfinite(median_abs) or median_abs <= 0.0:
        median_abs = 1.0
    floor = max(np.finfo(np.float64).tiny, median_abs * 1.0e-12)
    scale = np.where(finite & (np.abs(mean_spectrum) >= floor), mean_spectrum, 1.0)
    normalized = arr64 / scale[:, np.newaxis, np.newaxis]
    if normalize_boost != 1.0:
        positive = normalized > 0.0
        boosted = np.empty_like(normalized, dtype=np.float64)
        boosted[positive] = np.float_power(np.maximum(normalized[positive], floor), normalize_boost)
        boosted[~positive] = normalized[~positive] * normalize_boost
        return boosted.astype(np.float32)
    return normalized.astype(np.float32)


def apply_deslope(arr, nu_coords, *, deslope):
    if float(deslope) == 0.0:
        return arr, None

    nu_abs = np.abs(np.asarray(nu_coords, dtype=np.float64))
    valid = np.isfinite(nu_abs) & (nu_abs > 0)
    if not bool(np.any(valid)):
        return arr, None

    deslope_ref = float(np.median(nu_abs[valid]))
    weights = np.ones(arr.shape[0], dtype=np.float64)
    weights[valid] = np.power(nu_abs[valid] / deslope_ref, float(deslope))
    return arr * weights[:, np.newaxis, np.newaxis].astype(np.float32), deslope_ref


def prepare_chroma(arr):
    arr_rgb = np.moveaxis(arr, 0, -1).astype(np.float64)
    arr_chroma = np.maximum(arr_rgb, 0.0)
    denom = np.sum(arr_chroma, axis=-1, keepdims=True, dtype=np.float64)
    denom = np.maximum(denom, np.finfo(np.float64).tiny)
    return arr_chroma / denom


def apply_brightness_scale(rgb_cube, brightness_source, *, intensity_mode, clip_min, clip_max, dynamic_range=2.5e3, brightness_max=None):
    total_flux = np.sum(np.maximum(brightness_source, 0.0), axis=0, dtype=np.float64)
    brightness = normalize_total_flux_brightness(
        total_flux, intensity_mode=intensity_mode, clip_min=clip_min, clip_max=clip_max,
        dynamic_range=dynamic_range, maxval=brightness_max,
    )
    luma = 0.2126 * rgb_cube[:, :, 0] + 0.7152 * rgb_cube[:, :, 1] + 0.0722 * rgb_cube[:, :, 2]
    scale = brightness / np.maximum(luma, 1.0e-6)
    return np.clip(rgb_cube * scale[:, :, np.newaxis], 0.0, 1.0)


def spectral_cube_to_rgb(
    cube,
    nu_coords,
    *,
    nu_axis_scale="linear",
    deslope=0.0,
    normalize_spectrum_enabled=False,
    normalize_spectrum_boost=1.0,
    intensity_scale="linear",
    range_min=0.0,
    range_max=100.0,
    dynamic_range=2.5e3,
    lambda_min=400.0,
    lambda_max=700.0,
    brightness_max=None,
    spectrum_ref=None,
):
    """
    Render a spectral cube of shape (f, n, m) into an (n, m, 3) sRGB image.

    Mirrors mobula's `build_multispectral_response` + `CpuMultispectralBackend`:
    the per-pixel spectral shape sets the hue, the total flux sets brightness.

    Parameters
    ----------
    cube : np.ndarray
        Spectral cube, shape (f, n, m) with f the frequency axis.
    nu_coords : np.ndarray
        Frequency coordinates, shape (f,).
    nu_axis_scale : {"linear", "log"}
        Frequency -> visible-wavelength mapping.
    deslope : float
        Spectral tilt exponent applied as (nu / nu_ref) ** deslope.
    normalize_spectrum_enabled : bool
        Divide each channel by its spatial mean before colorizing.
    normalize_spectrum_boost : float
        Power boost for the normalized spectrum (mobula range [0.25, 8.0]).
    intensity_scale : {"linear", "sqrt", "log"}
        Brightness curve applied to the total flux.
    range_min, range_max : float
        Brightness clip percentiles (0..100), -> clip_min/clip_max = range/100.
    dynamic_range : float
        Log-scale floor (hi / dynamic_range) used when clip_min == 0.
    lambda_min, lambda_max : float
        Visible-band endpoints in nm.
    brightness_max : float, optional
        Total-flux value mapped to full brightness. Defaults to this cube's own
        max; pass a shared reference (e.g. the full sky) to match brightness
        across images.
    spectrum_ref : array-like, optional
        Per-channel divisor for `normalize_spectrum`, shape (f,). Defaults to
        this cube's spatial mean; pass a shared reference to match hue.
    """
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D (f, n, m), got shape {arr.shape}")
    nu_coords = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
    if nu_coords.size != arr.shape[0]:
        raise ValueError("nu_coords must have shape (f,) matching cube.shape[0]")
    if arr.shape[0] < 3:
        raise ValueError("need at least 3 spectral channels for multispectral RGB")

    clip_min = float(np.clip(range_min / 100.0, 0.0, 1.0))
    clip_max = float(np.clip(range_max / 100.0, 0.0, 1.0))

    wavelength_axis_nm, _ = build_visible_wavelength_axis(
        nu_coords, axis_scale=nu_axis_scale, lambda_min=lambda_min, lambda_max=lambda_max
    )

    brightness_source = np.asarray(arr, dtype=np.float64).copy()

    if normalize_spectrum_enabled:
        arr = normalize_spectrum(arr, normalize_boost=normalize_spectrum_boost, ref_spectrum=spectrum_ref)

    arr, _ = apply_deslope(arr, nu_coords, deslope=deslope)
    arr_chroma = prepare_chroma(arr)

    rgb_cube, _ = convert_mf_to_rgb_new(
        arr_chroma,
        wavelength_axis_nm=wavelength_axis_nm,
        intensity_scale="linear",
        clip_min=0.0,
        clip_max=1.0,
        channel_relative_clip=False,
    )

    rgb_cube = apply_brightness_scale(
        rgb_cube, brightness_source, intensity_mode=intensity_scale, clip_min=clip_min, clip_max=clip_max,
        dynamic_range=dynamic_range, brightness_max=brightness_max,
    )
    return rgb_cube


def _gap_position(render_nu, label_nu, norm, *, min_ratio=1.5):
    """
    Normalized [0, 1] bar position of the largest gap between consecutive
    frequency labels, or None if there is no gap notably larger than the rest.

    `render_nu` are the channel positions on the bar, `label_nu` the real
    frequencies; the gap is placed midway (in bar coordinates) between the two
    channels straddling the biggest jump in `label_nu`. Returns None unless that
    jump exceeds `min_ratio` times the median jump (so evenly-spaced axes get no
    fade).
    """
    order = np.argsort(np.asarray(render_nu, dtype=np.float64))
    r = np.asarray(render_nu, dtype=np.float64)[order]
    l = np.asarray(label_nu, dtype=np.float64)[order]
    if r.size < 3:
        return None
    diffs = np.abs(np.diff(l))
    i = int(np.argmax(diffs))
    med = float(np.median(diffs))
    if not np.isfinite(med) or med <= 0 or diffs[i] < min_ratio * med:
        return None
    mid = 0.5 * (r[i] + r[i + 1])
    return float(norm(mid))


def frequency_colormap(nu_coords, *, nu_axis_scale="linear", lambda_min=400.0, lambda_max=700.0, n=256, spectral=False, brightness_normalize=True, gap_pos=None, gap_width=0.05, gap_depth=1.0, gap_core=0.03):
    """
    Build a (cmap, norm) pair mapping frequency -> the color a source at that
    frequency takes in the multi-color images, as a legend for the color axis.

    By default this is a true RGB ramp: red (low freq) -> green -> blue (high
    freq), matching the additive red/green/blue convention of the images. Set
    `spectral=True` for the physical spectral-locus (rainbow) colors from the
    CIE color-matching functions. Suitable for
    `fig.colorbar(ScalarMappable(norm=norm, cmap=cmap))`.

    If `gap_pos` (a normalized position in [0, 1], e.g. from `_gap_position`) is
    given, the bar is faded to transparent around it -- fully within a central
    core of half-width `gap_core` (the figure background shows through), with
    Gaussian shoulders of width `gap_width` and peak strength `gap_depth` --
    marking a larger frequency gap between two labels.
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize

    nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
    nu_min, nu_max = float(np.min(nu)), float(np.max(nu))

    use_log = nu_axis_scale == "log" and nu_min > 0
    if use_log:
        norm = LogNorm(vmin=nu_min, vmax=nu_max)
    else:
        norm = Normalize(vmin=nu_min, vmax=nu_max)

    if spectral:
        nu_dense = np.geomspace(nu_min, nu_max, n) if use_log else np.linspace(nu_min, nu_max, n)
        wl, _ = build_visible_wavelength_axis(
            nu_dense, axis_scale=("log" if use_log else "linear"),
            lambda_min=lambda_min, lambda_max=lambda_max,
        )
        rgb = _xyz_to_srgb(_xyz_from_wavelengths(wl).T)  # (n, 3)
        if brightness_normalize:
            # Pure spectral colors vary strongly in luminance; scale each to full
            # brightness so the bar reads as a clean hue ramp.
            rgb = rgb / np.maximum(rgb.max(axis=1, keepdims=True), 1e-6)
    else:
        # Clean red -> green -> blue ramp (low freq red, high freq blue).
        base = LinearSegmentedColormap.from_list(
            "freq_rgb", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], N=n
        )
        rgb = base(np.linspace(0.0, 1.0, n))[:, :3]

    if gap_pos is not None:
        # Fade the bar to transparent around the gap -- fully so in a small
        # central core (figure background shows through), with Gaussian shoulders
        # -- so the bar visibly "breaks".
        t = np.linspace(0.0, 1.0, n)
        d = np.abs(t - gap_pos)
        w = np.where(d <= gap_core, 1.0, np.exp(-0.5 * ((d - gap_core) / max(gap_width, 1e-6)) ** 2))
        w = np.clip(gap_depth * w, 0.0, 1.0)
        rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), (1.0 - w)[:, None]], axis=1)
        return ListedColormap(rgba), norm

    return ListedColormap(np.clip(rgb, 0.0, 1.0)), norm


def _fade_cbar_outline(cb, *, orientation, gap_pos, gap_width=0.05, gap_depth=1.0, gap_core=0.03, n=200, lw=0.8):
    """
    Replace a colorbar's solid outline with one whose two long edges fade to
    transparent around `gap_pos` (normalized 0..1), matching the colormap gap so
    the frame visibly "breaks" there. The short end-caps stay solid. Also makes
    the colorbar's axes patch transparent so the fully-faded core shows the
    figure background.
    """
    from matplotlib.collections import LineCollection

    ax = cb.ax
    cb.outline.set_visible(False)
    ax.patch.set_alpha(0.0)  # let the figure background show through the faded core
    t = np.linspace(0.0, 1.0, n)
    d = np.abs(t - gap_pos)
    w = np.where(d <= gap_core, 1.0, np.exp(-0.5 * ((d - gap_core) / max(gap_width, 1e-6)) ** 2))
    alpha = 1.0 - np.clip(gap_depth * w, 0.0, 1.0)

    if orientation == "horizontal":
        edges = [np.column_stack([t, np.full_like(t, y)]) for y in (0.0, 1.0)]
        short = [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]]
    else:
        edges = [np.column_stack([np.full_like(t, x), t]) for x in (0.0, 1.0)]
        short = [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]

    long_segs, long_alpha = [], []
    for pts in edges:
        for k in range(n - 1):
            long_segs.append([pts[k], pts[k + 1]])
            long_alpha.append(0.5 * (alpha[k] + alpha[k + 1]))
    long_colors = np.zeros((len(long_segs), 4))
    long_colors[:, 3] = long_alpha

    ax.add_collection(LineCollection(long_segs, colors=long_colors, linewidths=lw, transform=ax.transAxes, clip_on=False, zorder=5))
    ax.add_collection(LineCollection(short, colors=[(0.0, 0.0, 0.0, 1.0)] * len(short), linewidths=lw, transform=ax.transAxes, clip_on=False, zorder=5))


def plot_freq_brightness_legend(
    fig,
    rect,
    *,
    render_nu,
    label_nu,
    nu_axis_scale="linear",
    lambda_min=400.0,
    lambda_max=700.0,
    spectral=False,
    gap_pos=None,
    gap_width=0.10,
    gap_depth=1.0,
    gap_core=0.03,
    intensity_mode="log",
    clip_min=0.0,
    clip_max=1.0,
    dynamic_range=2.5e3,
    brightness_max=1.0,
    n_freq=256,
    n_bright=256,
    freq_scale=1e-6,
    freq_decimals=0,
    freq_label="frequency [MHz]",
    bright_label="mJy / arcsec$^2$",
):
    """
    Draw a 2D legend for the multi-color images at figure-fraction `rect`
    (`[x0, y0, w, h]`): x-axis = frequency (hue ramp, with the gap fade), y-axis
    = brightness = total flux (ticks/label on the right).

    Each cell is the pure-frequency hue scaled to that brightness exactly as
    `apply_brightness_scale` does (rgb_hue * b / luma), so the swatch is a
    faithful decoder of the image colors. The flux axis inverts the brightness
    normalization (`normalize_total_flux_brightness`) using the same
    `intensity_mode`, `clip_min/clip_max`, `dynamic_range` and `brightness_max`.
    """
    cmap, norm = frequency_colormap(
        render_nu, nu_axis_scale=nu_axis_scale, lambda_min=lambda_min, lambda_max=lambda_max,
        spectral=spectral, gap_pos=gap_pos, gap_width=gap_width, gap_depth=gap_depth, gap_core=gap_core,
    )
    cols = np.asarray(cmap(np.linspace(0.0, 1.0, n_freq)))  # (n_freq, 4)
    rgb_hue = cols[:, :3]
    col_alpha = cols[:, 3]
    luma = 0.2126 * rgb_hue[:, 0] + 0.7152 * rgb_hue[:, 1] + 0.0722 * rgb_hue[:, 2]

    # Rows = brightness in [0, 1], columns = frequency; scale hue by b / luma.
    b = np.linspace(0.0, 1.0, n_bright)
    scale = b[:, None] / np.maximum(luma[None, :], 1e-6)
    rgb = np.clip(rgb_hue[None, :, :] * scale[:, :, None], 0.0, 1.0)
    a = np.broadcast_to(col_alpha[None, :, None], rgb.shape[:2] + (1,))
    swatch = np.concatenate([rgb, a], axis=-1)  # (n_bright, n_freq, 4)

    ax = fig.add_axes(rect)
    ax.patch.set_alpha(0.0)  # let the figure background show through the gap
    ax.imshow(swatch, origin="lower", extent=[0.0, 1.0, 0.0, 1.0], aspect="auto", interpolation="nearest", zorder=2)

    # x-axis: frequency ticks at the channel positions.
    ax.set_xticks([float(norm(v)) for v in np.asarray(render_nu, dtype=np.float64).reshape(-1)])
    ax.set_xticklabels([f"{v * freq_scale:.{freq_decimals}f}" for v in np.asarray(label_nu, dtype=np.float64).reshape(-1)])
    if freq_label:
        ax.set_xlabel(freq_label)

    # y-axis: brightness -> flux (invert the brightness normalization), on the right.
    hi = brightness_max * clip_max
    if intensity_mode == "log":
        lo = brightness_max * clip_min if clip_min > 0.0 else hi / max(dynamic_range, 1.0 + 1e-12)
        lo = max(lo, np.finfo(np.float64).tiny)
        b_of = lambda fx: np.log(np.asarray(fx) / lo) / np.log(hi / lo)
        k0, k1 = int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi)))
        fx = [10.0 ** k for k in range(k0, k1 + 1)] or [lo, hi]
    else:
        lo = max(brightness_max * clip_min, 0.0)
        span = max(hi - lo, 1e-12)
        if intensity_mode == "sqrt":
            b_of = lambda fx: np.sqrt(np.clip((np.asarray(fx) - lo) / span, 0.0, 1.0))
        else:
            b_of = lambda fx: np.clip((np.asarray(fx) - lo) / span, 0.0, 1.0)
        fx = list(np.linspace(lo, hi, 4))
    fx = [f for f in fx if lo <= f <= hi]
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks([float(np.clip(b_of(f), 0.0, 1.0)) for f in fx])
    ax.set_yticklabels([f"{f:.3g}" for f in fx])
    if bright_label:
        ax.set_ylabel(bright_label, rotation=270, labelpad=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    return ax


def plot_multi_color(
    cube,
    nu_coords,
    *,
    odir=None,
    name=None,
    figsize=(10, 10),
    dpi=500,
    origin="lower",
    marker=None,
    cbar=False,
    cbar_label="frequency [MHz]",
    cbar_loc="right",
    cbar_width=0.0225,
    cbar_nu=None,
    cbar_freq_scale=1e-6,
    cbar_freq_decimals=0,
    cbar_spectral=False,
    cbar_gap=False,
    cbar_gap_width=0.05,
    cbar_gap_depth=1.0,
    cbar_gap_core=0.01,
    cbar_2d=False,
    cbar2d_size=(0.20, 0.24),
    cbar2d_bright_label="mJy / arcsec$^2$",
    **rgb_kwargs,
):
    """
    Render a spectral cube (f, n, m) as a multi-color RGB image and save to PNG.

    All keyword arguments beyond the plotting ones are forwarded to
    `spectral_cube_to_rgb` (nu_axis_scale, deslope, intensity_scale, range_min,
    range_max, ...). If `odir` and `name` are given the figure is saved,
    otherwise it is shown.

    `marker` overplots scatter markers (e.g. `box_markers(...)`): a single
    `{'x', 'y', ...}` spec, or a dict of such specs (each passed to
    `ax.scatter`).

    If `cbar=True` a frequency colorbar is drawn on the right: a true RGB ramp
    (low freq -> red, high -> blue; `cbar_spectral=True` for the rainbow
    spectral-locus). The hue ramp is positioned by `nu_coords` (so it matches
    the image); ticks sit at each channel, labelled with `cbar_nu` (real
    frequencies, e.g. `sky_mf.freq`) if given, else `nu_coords`, scaled by
    `cbar_freq_scale` (default Hz -> MHz) and formatted with `cbar_freq_decimals`
    digits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    rgb = spectral_cube_to_rgb(cube, nu_coords, **rgb_kwargs)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Match the (a.T, origin="lower") orientation used in eso_compare.py.
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin=origin, aspect="auto")
    ax.set_box_aspect(rgb.shape[1] / rgb.shape[0])

    if marker:
        # Accept a single {x, y, ...} spec or a dict of such subdicts.
        specs = {"m0": marker} if all(k in marker for k in ("x", "y")) else marker
        for spec in specs.values():
            ax.scatter(**spec)

    ax.axis("off")

    # Frequency (1D) or frequency x brightness (2D) colorbar on the right.
    if cbar:
        render_nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
        label_nu = render_nu if cbar_nu is None else np.asarray(cbar_nu, dtype=np.float64).reshape(-1)
        cmap_kw = dict(
            nu_axis_scale=rgb_kwargs.get("nu_axis_scale", "linear"),
            lambda_min=rgb_kwargs.get("lambda_min", 400.0),
            lambda_max=rgb_kwargs.get("lambda_max", 700.0),
            spectral=cbar_spectral,
        )
        _, norm = frequency_colormap(render_nu, **cmap_kw)
        gap_pos = _gap_position(render_nu, label_nu, norm) if cbar_gap else None
        fig.canvas.draw()
        box = ax.get_position()
        pad = 0.011  # fixed image-to-colorbar gap, matching plot_column.
        if cbar_2d:
            w, h = cbar2d_size
            x0 = (box.x0 - pad - w) if cbar_loc == "left" else (box.x1 + pad)
            y0 = box.y0 + 0.5 * (box.height - h)
            bmax = rgb_kwargs.get("brightness_max")
            if bmax is None:
                bmax = float(np.sum(np.maximum(np.asarray(cube, dtype=np.float64), 0.0), axis=0).max())
            plot_freq_brightness_legend(
                fig, [x0, y0, w, h], render_nu=render_nu, label_nu=label_nu, **cmap_kw,
                gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core,
                intensity_mode=rgb_kwargs.get("intensity_scale", "linear"),
                clip_min=float(np.clip(rgb_kwargs.get("range_min", 0.0) / 100.0, 0.0, 1.0)),
                clip_max=float(np.clip(rgb_kwargs.get("range_max", 100.0) / 100.0, 0.0, 1.0)),
                dynamic_range=rgb_kwargs.get("dynamic_range", 2.5e3),
                brightness_max=bmax,
                freq_scale=cbar_freq_scale, freq_decimals=cbar_freq_decimals,
                freq_label=cbar_label, bright_label=cbar2d_bright_label,
            )
        else:
            cmap, _ = frequency_colormap(render_nu, **cmap_kw)
            if gap_pos is not None:
                cmap, _ = frequency_colormap(
                    render_nu, **cmap_kw, gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core
                )
            x0 = (box.x0 - pad - cbar_width) if cbar_loc == "left" else (box.x1 + pad)
            cax = fig.add_axes([x0, box.y0, cbar_width, box.height])
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cb.set_ticks(render_nu)
            cb.set_ticklabels([f"{v * cbar_freq_scale:.{cbar_freq_decimals}f}" for v in label_nu])
            if cbar_label:
                cb.set_label(cbar_label)
            if gap_pos is not None:
                _fade_cbar_outline(cb, orientation="vertical", gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core)

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name), bbox_inches="tight")
        print("saved:", os.path.join(odir, name))
    else:
        plt.show()
    plt.close()
    return rgb


def plot_multi_color_column(
    cubes,
    nu_coords,
    *,
    odir=None,
    name=None,
    labels=None,
    label_color="white",
    frame=False,
    origin="lower",
    fig_width=5.0,
    hspace=0.025,
    dpi=300,
    cbar=False,
    cbar_label="frequency [MHz]",
    cbar_loc="right",
    cbar_width=0.0225,
    cbar_nu=None,
    cbar_freq_scale=1e-6,
    cbar_freq_decimals=0,
    cbar_spectral=False,
    cbar_gap=False,
    cbar_gap_width=0.05,
    cbar_gap_depth=1.0,
    cbar_gap_core=0.01,
    cbar_2d=False,
    cbar2d_size=(0.20, 0.24),
    cbar2d_bright_label="mJy / arcsec$^2$",
    **rgb_kwargs,
):
    """
    Render several spectral cubes (each (f, n, m)) as multi-color RGB images
    stacked in a single column, all sharing the same width in x.

    Mirrors the layout of `plot_column` in eso_components.py (same figure width
    per row, heights following each image's aspect ratio). Rendering kwargs are
    forwarded to `spectral_cube_to_rgb`.

    If `cbar=True` a shared frequency colorbar is drawn alongside the stack: a
    true RGB ramp (low freq -> red, high -> blue; `cbar_spectral=True` for the
    rainbow spectral-locus instead). The hue ramp is positioned by the rendering
    `nu_coords` (so it matches the images), while the tick labels use `cbar_nu`
    (the real frequencies, e.g. `sky_mf.freq`) if given, else `nu_coords`,
    scaled by `cbar_freq_scale` (default Hz -> MHz) and formatted with
    `cbar_freq_decimals` digits after the dot (0 -> "1350", 3 -> "1.350").
    Pass `cbar_nu` when `nu_coords` is just channel indices.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    rgbs = [spectral_cube_to_rgb(c, nu_coords, **rgb_kwargs) for c in cubes]
    n = len(rgbs)

    row_labels = ([None] * n) if labels is None else list(labels) + [None] * (n - len(labels))

    hspace = hspace if hspace > 0 else 0.025
    # Same width in x for every image; height_ratios give each its own aspect.
    aspects = [r.shape[1] / r.shape[0] for r in rgbs]
    fig_h = fig_width * sum(aspects) * (1 + hspace)
    figure, axes = plt.subplots(
        n,
        1,
        figsize=(fig_width, fig_h),
        dpi=dpi,
        gridspec_kw={"hspace": hspace, "height_ratios": aspects},
    )
    axes = np.atleast_1d(axes).ravel().tolist()

    for ax, r, lab in zip(axes, rgbs, row_labels):
        ax.imshow(np.transpose(r, (1, 0, 2)), origin=origin, aspect="auto")

        if lab:
            ax.annotate(
                lab, xy=(0.97, 1.0), xycoords="axes fraction",
                xytext=(0, -12), textcoords="offset points",
                ha="right", va="top", color=label_color,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(frame)
            if frame:
                spine.set_color("black")
                spine.set_linewidth(0.8)

    # Shared frequency colorbar spanning the combined height of the stack. The
    # hue ramp is positioned by the rendering `nu_coords` (so it matches the
    # images); ticks sit at each channel and are labelled with `cbar_nu`.
    if cbar:
        render_nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
        label_nu = render_nu if cbar_nu is None else np.asarray(cbar_nu, dtype=np.float64).reshape(-1)
        cmap_kw = dict(
            nu_axis_scale=rgb_kwargs.get("nu_axis_scale", "linear"),
            lambda_min=rgb_kwargs.get("lambda_min", 400.0),
            lambda_max=rgb_kwargs.get("lambda_max", 700.0),
            spectral=cbar_spectral,
        )
        _, norm = frequency_colormap(render_nu, **cmap_kw)
        gap_pos = _gap_position(render_nu, label_nu, norm) if cbar_gap else None
        figure.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        top = max(b.y1 for b in boxes)
        bottom = min(b.y0 for b in boxes)
        pad = 0.011  # fixed image-to-colorbar gap, matching plot_column.
        if cbar_2d:
            w, h = cbar2d_size
            x0 = (min(b.x0 for b in boxes) - pad - w) if cbar_loc == "left" else (max(b.x1 for b in boxes) + pad)
            y0 = bottom + 0.5 * ((top - bottom) - h)
            bmax = rgb_kwargs.get("brightness_max")
            if bmax is None:
                bmax = float(max(np.sum(np.maximum(np.asarray(c, dtype=np.float64), 0.0), axis=0).max() for c in cubes))
            plot_freq_brightness_legend(
                figure, [x0, y0, w, h], render_nu=render_nu, label_nu=label_nu, **cmap_kw,
                gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core,
                intensity_mode=rgb_kwargs.get("intensity_scale", "linear"),
                clip_min=float(np.clip(rgb_kwargs.get("range_min", 0.0) / 100.0, 0.0, 1.0)),
                clip_max=float(np.clip(rgb_kwargs.get("range_max", 100.0) / 100.0, 0.0, 1.0)),
                dynamic_range=rgb_kwargs.get("dynamic_range", 2.5e3),
                brightness_max=bmax,
                freq_scale=cbar_freq_scale, freq_decimals=cbar_freq_decimals,
                freq_label=cbar_label, bright_label=cbar2d_bright_label,
            )
        else:
            cmap, _ = frequency_colormap(render_nu, **cmap_kw)
            if gap_pos is not None:
                cmap, _ = frequency_colormap(
                    render_nu, **cmap_kw, gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core
                )
            x0 = (min(b.x0 for b in boxes) - pad - cbar_width) if cbar_loc == "left" else (max(b.x1 for b in boxes) + pad)
            cax = figure.add_axes([x0, bottom, cbar_width, top - bottom])
            cb = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cb.set_ticks(render_nu)
            # Fixed decimals after the dot (0 -> "1350" for MHz, 3 -> "1.350" for GHz).
            cb.set_ticklabels([f"{v * cbar_freq_scale:.{cbar_freq_decimals}f}" for v in label_nu])
            if cbar_label:
                cb.set_label(cbar_label)
            if gap_pos is not None:
                _fade_cbar_outline(cb, orientation="vertical", gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core)

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name), bbox_inches="tight")
        print("saved:", os.path.join(odir, name))
    else:
        plt.show()
    plt.close()
    return rgbs


def plot_multi_color_grid(
    cubes,
    nu_coords,
    *,
    rows=6,
    cols=6,
    odir=None,
    name=None,
    labels=None,
    label_color="white",
    label_fontsize=12,
    frame=False,
    origin="lower",
    tile_size=2.0,
    space=0.04,
    scale=1.0,
    dpi=300,
    cbar=False,
    cbar_label="frequency [MHz]",
    cbar_nu=None,
    cbar_freq_scale=1e-6,
    cbar_freq_decimals=0,
    cbar_spectral=False,
    cbar_gap=False,
    cbar_gap_width=0.05,
    cbar_gap_depth=1.0,
    cbar_gap_core=0.01,
    cbar_2d=False,
    cbar2d_size=(0.20, 0.24),
    cbar2d_bright_label="mJy / arcsec$^2$",
    **rgb_kwargs,
):
    """
    Render several spectral cubes (each (f, n, m)) as multi-color RGB images in a
    `rows` x `cols` grid, mirroring the layout of `plot_tiles_grid` in
    eso_components.py. Rendering kwargs are forwarded to `spectral_cube_to_rgb`.

    If `cbar=True` a shared horizontal frequency colorbar is drawn at the bottom
    (true RGB ramp, `cbar_spectral=True` for the rainbow spectral-locus). Ticks
    sit at each channel, labelled with `cbar_nu` (real frequencies) if given else
    `nu_coords`, scaled by `cbar_freq_scale` and formatted with
    `cbar_freq_decimals` digits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    rgbs = [spectral_cube_to_rgb(c, nu_coords, **rgb_kwargs) for c in cubes]

    # Layout (inches), matching plot_tiles_grid: side margins, a top margin and
    # a bottom strip for the shared colorbar; the figure height makes cells square.
    tile_size = tile_size * scale
    margin_x = 0.02 * tile_size * cols
    margin_top = 0.1 * scale
    if cbar and cbar_2d:
        cbar_strip = 2.2 * scale  # taller strip to fit the 2D legend swatch
    elif cbar:
        cbar_strip = 0.95 * scale
    else:
        cbar_strip = 0.1 * scale
    cbar_height = 0.28 * scale

    fig_w = tile_size * cols
    grid_w = fig_w - 2 * margin_x
    cell_w = grid_w / (cols + (cols - 1) * space)
    grid_h = cell_w * (rows + (rows - 1) * space)
    fig_h = grid_h + margin_top + cbar_strip

    cbar_strip_gap = space * cell_w

    left = margin_x / fig_w
    right = 1 - margin_x / fig_w
    bottom = cbar_strip / fig_h
    top = 1 - margin_top / fig_h

    figure, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    figure.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=space, hspace=space
    )
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < len(rgbs):
            ax.imshow(np.transpose(rgbs[i], (1, 0, 2)), origin=origin)
            if labels is not None and i < len(labels) and labels[i]:
                ax.text(
                    0.05, 0.93, labels[i],
                    transform=ax.transAxes, ha="left", va="top",
                    color=label_color, fontsize=label_fontsize,
                )
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(frame)
            if frame:
                spine.set_linewidth(0.5)

    # Single horizontal frequency colorbar spanning the grid width.
    if cbar:
        render_nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
        label_nu = render_nu if cbar_nu is None else np.asarray(cbar_nu, dtype=np.float64).reshape(-1)
        cmap_kw = dict(
            nu_axis_scale=rgb_kwargs.get("nu_axis_scale", "linear"),
            lambda_min=rgb_kwargs.get("lambda_min", 400.0),
            lambda_max=rgb_kwargs.get("lambda_max", 700.0),
            spectral=cbar_spectral,
        )
        _, norm = frequency_colormap(render_nu, **cmap_kw)
        gap_pos = _gap_position(render_nu, label_nu, norm) if cbar_gap else None
        if cbar_2d:
            legend_h_in = 1.3 * scale
            legend_w_in = 3.0 * scale
            rect = [
                (fig_w - legend_w_in) / 2.0 / fig_w,
                (0.35 * scale) / fig_h,
                legend_w_in / fig_w,
                legend_h_in / fig_h,
            ]
            bmax = rgb_kwargs.get("brightness_max")
            if bmax is None:
                bmax = float(max(np.sum(np.maximum(np.asarray(c, dtype=np.float64), 0.0), axis=0).max() for c in cubes))
            plot_freq_brightness_legend(
                figure, rect, render_nu=render_nu, label_nu=label_nu, **cmap_kw,
                gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core,
                intensity_mode=rgb_kwargs.get("intensity_scale", "linear"),
                clip_min=float(np.clip(rgb_kwargs.get("range_min", 0.0) / 100.0, 0.0, 1.0)),
                clip_max=float(np.clip(rgb_kwargs.get("range_max", 100.0) / 100.0, 0.0, 1.0)),
                dynamic_range=rgb_kwargs.get("dynamic_range", 2.5e3),
                brightness_max=bmax,
                freq_scale=cbar_freq_scale, freq_decimals=cbar_freq_decimals,
                freq_label=cbar_label, bright_label=cbar2d_bright_label,
            )
        else:
            cmap, _ = frequency_colormap(render_nu, **cmap_kw)
            if gap_pos is not None:
                cmap, _ = frequency_colormap(
                    render_nu, **cmap_kw, gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core
                )
            cax = figure.add_axes(
                [left, (cbar_strip - cbar_strip_gap - cbar_height) / fig_h, right - left, cbar_height / fig_h]
            )
            cb = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
            cb.set_ticks(render_nu)
            cb.set_ticklabels([f"{v * cbar_freq_scale:.{cbar_freq_decimals}f}" for v in label_nu])
            if cbar_label:
                cb.set_label(cbar_label)
            if gap_pos is not None:
                _fade_cbar_outline(cb, orientation="horizontal", gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core)

    # Save without bbox_inches="tight" to preserve the computed layout (plot_figure).
    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name))
        print("saved:", os.path.join(odir, name))
    else:
        plt.show()
    plt.close()
    return rgbs


# --- copied from eso_c1_profiles.py ---
def perp_slices(
    anchors_frac, half_width_arcsec, n_slices, n_perp, shape, pix_arcsec,
):
    """Build sample coordinates for `n_slices` stripes perpendicular to the link.

    The link is the polyline through `anchors_frac` (>= 2 fractional (x, y)
    points). Stripes are placed at equal spacing *along* the polyline (equal arc
    length); within each straight segment that spacing equals the Euclidean
    distance between stripe centers. Each stripe is perpendicular to its local
    segment. Pixels are square, so the geometry is taken directly in pixel space.

    Returns
    -------
    s_arcsec : (n_slices,) arc-length position along the link [arcsec]
    centers  : (n_slices, 2) pixel coords (axis0, axis1) of the stripe centers
    coords   : (2, n_slices * n_perp) pixel coords for `map_coordinates`
    seg_ends : (n_slices, 2, 2) endpoints of each stripe segment (for plotting)
    """
    nx, ny = shape
    anchors = np.asarray(anchors_frac, dtype="float64")
    if anchors.ndim != 2 or anchors.shape[0] < 2 or anchors.shape[1] != 2:
        raise ValueError("`anchors_frac` must be a list of >= 2 (x, y) points")
    verts = anchors * np.array([nx - 1, ny - 1])  # (n_anchor, 2) pixel coords

    # Cumulative arc length along the polyline (pixel space).
    seg_vec = np.diff(verts, axis=0)                       # (n_seg, 2)
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])       # (n_seg,)
    if np.any(seg_len == 0):
        raise ValueError("consecutive anchors must not coincide")
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])  # (n_anchor,)
    total_len = cum_len[-1]

    half_w = half_width_arcsec / pix_arcsec

    # Equal-arc-length stripe positions; locate the segment each falls in.
    s = np.linspace(0.0, total_len, n_slices)              # (n_slices,)
    seg_idx = np.clip(
        np.searchsorted(cum_len, s, side="right") - 1, 0, len(seg_len) - 1
    )

    seg_dir = seg_vec / seg_len[:, None]                   # unit dirs (n_seg, 2)
    local_s = s - cum_len[seg_idx]                         # offset within segment
    centers = verts[seg_idx] + local_s[:, None] * seg_dir[seg_idx]  # (n_slices, 2)

    direction = seg_dir[seg_idx]                                    # (n_slices, 2)
    perp = np.stack([-direction[:, 1], direction[:, 0]], axis=1)    # (n_slices, 2)

    ss = np.linspace(-half_w, half_w, n_perp)
    pts = centers[:, None, :] + ss[None, :, None] * perp[:, None, :]
    coords = pts.reshape(-1, 2).T  # (2, n_slices * n_perp): [axis0, axis1]

    seg_ends = np.stack(
        [centers - half_w * perp, centers + half_w * perp], axis=1
    )  # (n_slices, 2, 2)

    s_arcsec = s * pix_arcsec
    return s_arcsec, centers, coords, seg_ends


def ridge_anchors(flux, anchors, *, method="max", smooth=0.0, y_window_frac=None):
    """Trace the link ridge as (x, y) fractional anchors, one per x-pixel column.

    The start/end x are the first/last `anchors` x; for each x-column between,
    pick the y-pixel by `method`: "max" (brightest pixel) or "median" (flux-
    weighted median position, where the cumulative brightness reaches 50%).

    If `y_window_frac` is given, the search in each column is confined to a
    corridor of half-height `y_window_frac * (ny - 1)` around the guide y(x)
    interpolated from the `anchors` polyline -- so the ridge follows the drawn
    sub-part even when the galaxy is brighter elsewhere. `smooth` gaussian-blurs
    the traced y(x) in pixels (0 disables). Returns (n, 2) fractional (x, y).
    """
    flux = np.clip(np.asarray(flux, dtype="float64"), 0.0, None)  # (nx, ny)
    nx, ny = flux.shape
    anchors = np.asarray(anchors, dtype="float64")
    ax_pix = anchors[:, 0] * (nx - 1)
    ay_pix = anchors[:, 1] * (ny - 1)

    ix0, ix1 = int(round(ax_pix[0])), int(round(ax_pix[-1]))
    step = 1 if ix1 >= ix0 else -1
    xs = np.arange(ix0, ix1 + step, step)
    y_idx = np.arange(ny, dtype="float64")

    # Guide y(x) from the (x-sorted) anchor polyline; corridor half-height.
    order = np.argsort(ax_pix)
    guide_y = np.interp(xs, ax_pix[order], ay_pix[order])
    half = None if y_window_frac is None else y_window_frac * (ny - 1)

    ys = np.empty(xs.size, dtype="float64")
    for k, ix in enumerate(xs):
        if half is not None:
            lo = max(0, int(np.floor(guide_y[k] - half)))
            hi = min(ny, int(np.ceil(guide_y[k] + half)) + 1)
        else:
            lo, hi = 0, ny
        col = flux[ix, lo:hi]
        sub_y = y_idx[lo:hi]
        total = col.sum()
        if method == "max":
            ys[k] = float(sub_y[np.argmax(col)])
        elif method == "median":
            ys[k] = float(np.interp(0.5, np.cumsum(col) / total, sub_y)) if total > 0 else float(sub_y[np.argmax(col)])
        else:
            raise ValueError("RIDGE_METHOD must be 'max' or 'median'")

    if smooth and smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        ys = gaussian_filter1d(ys, float(smooth), mode="nearest")

    return np.column_stack([xs / (nx - 1), ys / (ny - 1)])


# --- copied from eso_alpha_vs_flux.py ---
def assign_regions_to_pixels(mask, regions_config, brightness=None):
    """Assign a region id per pixel from regions.yml shapes + value ranges."""
    region_id = np.full(mask.shape, -1, dtype=int)
    if brightness is not None:
        brightness = np.asarray(brightness)
    y_coords, x_coords = np.where(mask)

    for idx, name in enumerate(regions_config):
        region = regions_config[name]
        region_mask = np.zeros(len(y_coords), dtype=bool)
        for shape_idx, (cen, ext) in enumerate(zip(region["center"], region["extend"])):
            c_y, c_x = cen
            if len(ext) == 1:  # circle
                radius = ext[0] / 2
                dist = np.sqrt((x_coords - c_x) ** 2 + (y_coords - c_y) ** 2)
                geom_mask = dist <= radius
            elif len(ext) == 2:  # rectangle
                f_y, f_x = ext
                geom_mask = (
                    (x_coords >= c_x - f_x / 2)
                    & (x_coords < c_x + f_x / 2)
                    & (y_coords >= c_y - f_y / 2)
                    & (y_coords < c_y + f_y / 2)
                )
            else:
                raise ValueError("Unknown shape type")
            if "value" in region and brightness is not None and shape_idx < len(region["value"]):
                v_min, v_max = (float(v) for v in region["value"][shape_idx])
                bvals = brightness[y_coords, x_coords]
                geom_mask = geom_mask & (bvals >= v_min) & (bvals <= v_max)
            region_mask |= geom_mask
        region_id[y_coords[region_mask], x_coords[region_mask]] = idx
    return region_id


def region_color(name):
    n = name.lower()
    if "link" in n:
        return mcolors.to_rgba("orange")
    if "left" in n:
        return plt.cm.coolwarm(0.0)
    if "right" in n:
        return plt.cm.coolwarm(1.0)
    return mcolors.to_rgba("grey")


def region_order(name):
    n = name.lower()
    if "right" in n:
        return 0
    if "left" in n:
        return 1
    if "link" in n:
        return 2
    return 3


def region_label(name):
    n = name.lower()
    if "right" in n:
        return "right lobe"
    if "left" in n:
        return "left lobe"
    if "link" in n:
        return "threads"
    return name




# %%
# ===========================================================================
# FITS access. Arrays are returned in the model orientation (axis 0 = RA,
# axis 1 = DEC), i.e. transposed back from the FITS (STOKES, FREQ, DEC, RA).
# ===========================================================================
def _freq_files():
    files = glob.glob(os.path.join(fits_dir, f"{PREFIX}_*MHz_posterior_mean.fits"))
    freqs = [pyfits.getheader(f)["CRVAL3"] for f in files]
    return [round(f / 1e6) for f in sorted(freqs)]


FREQ_MHZ = _freq_files()
freq_hz = np.array([pyfits.getheader(os.path.join(fits_dir, f"{PREFIX}_{m}MHz_posterior_mean.fits"))["CRVAL3"] for m in FREQ_MHZ])
SAMPLE_TAGS = sorted(
    {os.path.basename(f).split("posterior_")[1][:-5]
     for f in glob.glob(os.path.join(fits_dir, f"{PREFIX}_spectral_index_posterior_sample_*.fits"))},
    key=lambda t: int(t.split("_")[1]),
)
print("frequencies [MHz]:", FREQ_MHZ, "| samples:", SAMPLE_TAGS)


def fits_file(kind, tag):
    """`kind` is a frequency in MHz (brightness) or "alpha" (spectral index)."""
    name = "spectral_index" if kind == "alpha" else f"{kind}MHz"
    return os.path.join(fits_dir, f"{PREFIX}_{name}_posterior_{tag}.fits")


@lru_cache(maxsize=None)
def layer(kind, tag, ext):
    """One image layer as a float64 array: 2D sky / cutout (x, y) or TILECUBE (tile, x, y)."""
    with pyfits.open(fits_file(kind, tag), memmap=False) as h:
        data = np.asarray(h[ext].data, dtype="float64")
    return data.transpose(0, 2, 1) if ext.startswith("TILECUBE") else data[0, 0].T


def cube(tag, ext):
    """Brightness layer `ext` at all frequencies, (nfreq, x, y) [mJy/arcsec^2]."""
    return np.stack([layer(m, tag, ext) for m in FREQ_MHZ])


def tile_cube(tag):
    """Individual tiles at all frequencies, (n_tiles, nfreq, tx, ty) [mJy/arcsec^2]."""
    return np.stack([layer(m, tag, "TILECUBE") for m in FREQ_MHZ], axis=1)


def table(kind, tag, ext):
    with pyfits.open(fits_file(kind, tag), memmap=False) as h:
        return h[ext].data.copy()


with pyfits.open(fits_file(FREQ_MHZ[REF_IDX], "mean")) as _h:
    SKY_SHAPE = tuple(_h["BG"].data.shape[-1:-3:-1])  # (nx, ny)
    _crpix = np.array([_h["BG"].header["CRPIX1"], _h["BG"].header["CRPIX2"]])
    OBJ_NAMES = [hdu.name for hdu in _h if hdu.name.startswith("O") and hdu.name[1:].isdigit()]
    # 0-based start pixel of every object cutout on the sky grid (from the CRPIX shift).
    OBJ_OFFSETS = {
        n: np.rint(_crpix - [_h[n].header["CRPIX1"], _h[n].header["CRPIX2"]]).astype(int) for n in OBJ_NAMES
    }
print("sky grid:", SKY_SHAPE, "| objects:", {n: tuple(o) for n, o in OBJ_OFFSETS.items()})


def to_grid(cutout, name):
    """Place an object cutout (..., cx, cy) onto the full sky grid (zeros outside)."""
    cutout = np.asarray(cutout)
    out = np.zeros(cutout.shape[:-2] + SKY_SHAPE, dtype=cutout.dtype)
    (x0, y0), (cx, cy) = OBJ_OFFSETS[name], cutout.shape[-2:]
    out[..., x0:x0 + cx, y0:y0 + cy] = cutout
    return out


def samples(fun):
    """`fun(tag)` evaluated for every posterior sample, stacked on axis 0."""
    return np.stack([np.asarray(fun(t)) for t in SAMPLE_TAGS])


def std(values):
    """Spread over the samples (axis 0), bias-corrected as `MySamples.std`."""
    return np.std(values, axis=0, ddof=1)


def box_markers():
    """Point-source and object/tile box scatter markers, as `box_markers` in
    plot_rgb_freq.py: point sources from the POINTS table, 1-pixel outlines of
    the object cutouts and 3-pixel outlines of the tiles (their coarse grid)."""
    pts = table(FREQ_MHZ[REF_IDX], "mean", "POINTS")
    # One circle per pixel of the 3 x 3 sub-pixel footprint of each source (the
    # model-based plot marks every point-grid sub-pixel), centred on the source.
    dx, dy = (d.ravel() for d in np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"))
    ps_mrk = dict(
        x=(np.rint(pts["X_PIX"] - 1)[:, None] + dx).ravel(),
        y=(np.rint(pts["Y_PIX"] - 1)[:, None] + dy).ravel(),
        s=20, facecolors="none", edgecolors="white", linewidths=0.25, marker="o",
    )
    box_map = np.zeros(SKY_SHAPE)
    for n in OBJ_NAMES:
        (x0, y0), (cx, cy) = OBJ_OFFSETS[n], layer(FREQ_MHZ[REF_IDX], "mean", n).shape
        box = np.ones((cx, cy))
        box[1:-1, 1:-1] = 0
        box_map[x0:x0 + cx, y0:y0 + cy] += box
    for r in table(FREQ_MHZ[REF_IDX], "mean", "TILEPOS"):
        x0, y0, tx, ty = r["X0_PIX"] - 1, r["Y0_PIX"] - 1, r["NX"], r["NY"]
        box = np.ones((tx, ty))
        box[3:-3, 3:-3] = 0
        box_map[x0:x0 + tx, y0:y0 + ty] += box
    ox, oy = np.argwhere(box_map > 0).T
    oj_mrk = dict(x=ox, y=oy, s=0.05, c="white", marker=",")
    return dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)


u_freq = np.log(freq_hz / freq_hz[REF_IDX])  # log-parabola frequency axis
galaxy_labels = ["ESO137-006", "ESO137-007"]
COMPONENTS = [
    dict(name="O0", rel_fov=(0.16, 0.08), center=(0, "-0.05deg"), flux_min=1e-2),        # ESO137-006
    dict(name="O1", rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg"), flux_min=5e-3),  # ESO137-007
]


# %%
# ===========================================================================
# 1) Component maps (as eso_components.py): reference-frequency brightness and
#    its relative uncertainty, spectral-index uncertainty, spectral curvature
#    and its uncertainty, plus the per-tile spectral index.
# ===========================================================================
flux_comps, rel_std_comps, alpha_comps, alpha_std_comps = [], [], [], []
curv_comps, curv_std_comps = [], []
for comp in COMPONENTS:
    n = comp["name"]
    crop = lambda x: np.asarray(crop_component(x, comp["rel_fov"], comp["center"]))

    obj_k = samples(lambda t: cube(t, n))  # (n_samples, nfreq, cx, cy)
    flux_mean = to_grid(cube("mean", n)[REF_IDX], n)
    flux_std = to_grid(std(obj_k[:, REF_IDX]), n)
    rel_std = flux_std / flux_mean
    alpha = to_grid(layer("alpha", "mean", n), n)
    alpha_std = to_grid(std(samples(lambda t: layer("alpha", t, n))), n)

    flux_c = crop(flux_mean)
    mask = flux_c > comp["flux_min"]
    curv_k = np.stack([curvature_from_cube(crop(to_grid(o, n)), u_freq) for o in obj_k])

    flux_comps.append(flux_c)
    rel_std_comps.append(crop(rel_std))
    alpha_comps.append(np.where(mask, crop(alpha), np.nan))
    alpha_std_comps.append(np.where(mask, crop(alpha_std), np.nan))
    curv_comps.append(np.where(mask, curv_k.mean(axis=0), np.nan))
    curv_std_comps.append(np.where(mask, std(curv_k), np.nan))

contour_levels = [[1e-2, 1e-1, 1, 10], [5e-3, 5e-2, 5e-1, 5]]
flux_contours = [
    {"array": f, "levels": lvls, "colors": "black", "linewidths": 0.5}
    for f, lvls in zip(flux_comps, contour_levels)
]
# Shared spectral-index colour limits for all alpha maps (both galaxies, the
# uSARA / AIRI comparison and the tiles): 0.1 / 99.9 percentiles of the
# aim-resolve alpha over both galaxies, with a colorbar tick every 1.0.
ALPHA_VMIN, ALPHA_VMAX = (
    float(v) for v in np.percentile(np.concatenate([a[np.isfinite(a)] for a in alpha_comps]), [0.1, 99.9])
)
ALPHA_TICKS = np.arange(np.ceil(ALPHA_VMIN), ALPHA_VMAX + 1e-9, 1.0)
# Spectral-index maps: RdYlBu_r centred on alpha = -1 (two-slope: the blue half spans
# [ALPHA_VMIN, -1], the red half [-1, ALPHA_VMAX]), separating flat / freshly injected
# (red, alpha > -1) from steep / aged emission (blue, alpha < -1).
ALPHA_CENTER = -1.0
_f0 = (ALPHA_CENTER - ALPHA_VMIN) / (ALPHA_VMAX - ALPHA_VMIN)
_x = np.linspace(0.0, 1.0, 512)
ALPHA_CMAP = mcolors.ListedColormap(
    plt.get_cmap("RdYlBu_r")(np.where(_x < _f0, 0.5 * _x / _f0, 0.5 + 0.5 * (_x - _f0) / (1.0 - _f0)))
)

ref_name = f"cs_{FREQ_MHZ[REF_IDX]}mhz"
print(f"plotting {ref_name} ...")
plot_column(
    flux_comps, odir=odir, name=ref_name, cmap="viridis", norm="log", vmin=6e-4,
    vmax=float(max(np.nanmax(f) for f in flux_comps)), cbar_label=r"sky brightness $I$ [mJy / arcsec$^2$]",
    labels=galaxy_labels, fig_width=10.0, dpi=300,
)
plot_column(
    rel_std_comps, odir=odir, name=f"{ref_name}_std", cmap="viridis", norm="log", vmin=1e-3,
    frame=True, label_color="black", contour=flux_contours,
    cbar_label=r"sky brightness uncertainty $\sigma_I / I$", labels=galaxy_labels,
    fig_width=5.0, label_offset=-6, dpi=300,
)
# Uncertainty maps: same colormap as the corresponding value map (not centred, 0 -> vmax),
# vmax = 95th percentile over both galaxies.
std_vmax = lambda comps: float(np.nanpercentile(np.concatenate([c.ravel() for c in comps]), 95))
plot_column(
    alpha_std_comps, odir=odir, name="cs_alpha_std", cmap="RdYlBu_r", norm="linear",
    vmin=0, vmax=std_vmax(alpha_std_comps), cbar_ticks=np.arange(0, std_vmax(alpha_std_comps) + 1e-9, 0.05),
    frame=True, label_color="black", contour=flux_contours,
    cbar_label=r"spectral index uncertainty $\sigma_\alpha$", labels=galaxy_labels,
    fig_width=5.0, label_offset=-6, dpi=300,
)
curv_vlim = float(np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in curv_comps])), 99))
plot_column(
    curv_comps, odir=odir, name="cs_curvature", cmap="coolwarm", norm="linear",
    vmin=-curv_vlim, vmax=curv_vlim, frame=True, label_color="black", contour=flux_contours,
    cbar_label=r"spectral curvature $\beta$", labels=galaxy_labels, fig_width=10.0, dpi=300,
)
plot_column(
    curv_std_comps, odir=odir, name="cs_curvature_std", cmap="coolwarm", norm="linear",
    vmin=0, vmax=std_vmax(curv_std_comps), cbar_ticks=np.arange(0, std_vmax(curv_std_comps) + 1e-9, 0.1),
    frame=True, label_color="black", contour=flux_contours,
    cbar_label=r"spectral curvature uncertainty $\sigma_\beta$", labels=galaxy_labels,
    fig_width=5.0, label_offset=-6, dpi=300,
)

# Per-tile maps: the 36 brightest tiles by the peak of their frequency-averaged
# brightness, each zoomed into its central-half field of view.
tiles_rf_val = layer(FREQ_MHZ[REF_IDX], "mean", "TILECUBE")  # (n_tiles, tx, ty)
tiles_si_val = layer("alpha", "mean", "TILECUBE")
spc_0 = SignalSpace.build(shape=tiles_rf_val.shape[1:], fov=(2, 2))
spc_1 = SignalSpace.build(shape=tiles_rf_val.shape[1:], fov=(1, 1))
tiles_rf_val = np.asarray(map_signal(tiles_rf_val, spc_0, spc_1, order=1, vmap_sum=False))
tiles_si_val = np.asarray(map_signal(tiles_si_val, spc_0, spc_1, order=1, vmap_sum=False))

tiles_avg_val = np.asarray(map_signal(tile_cube("mean").mean(axis=1), spc_0, spc_1, order=1, vmap_sum=False))
tiles_peak = np.array([np.max(t) for t in tiles_avg_val])
tiles_order = np.argsort(tiles_peak)[::-1][:36]
bright_tiles_rf_val = [tiles_rf_val[i] + 1e-10 for i in tiles_order]
bright_tiles_si_val = [np.where(rf > 1e-2, tiles_si_val[i], np.nan) for rf, i in zip(bright_tiles_rf_val, tiles_order)]

print("plotting tiles_alpha ...")
plot_tiles_grid(
    arrays=bright_tiles_si_val, rows=6, cols=6, name="tiles_alpha", odir=odir, dpi=300,
    vmin=ALPHA_VMIN, vmax=ALPHA_VMAX, norm="linear", cmap=ALPHA_CMAP, frame=True,
    cbar_label=r"spectral index $\alpha$", cbar_ticks=ALPHA_TICKS,
    contour_arrays=bright_tiles_rf_val, contour_levels=[1e-2, 1e-1, 1],
)

# Per-tile spectral curvature: log-parabola fit per sample (as cs_curvature),
# averaged over the samples and zoomed into the same central-half field of view.
tiles_curv_k = np.stack([
    [curvature_from_cube(tc, u_freq) for tc in tile_cube(t)[tiles_order]] for t in SAMPLE_TAGS
])  # (n_samples, 36, tx, ty)
tiles_curv_val = np.asarray(map_signal(tiles_curv_k.mean(axis=0), spc_0, spc_1, order=1, vmap_sum=False))
bright_tiles_curv_val = [np.where(rf > 1e-2, c, np.nan) for rf, c in zip(bright_tiles_rf_val, tiles_curv_val)]
# Symmetric colour limit: 99th percentile of |beta| over all tile pixels (as cs_curvature).
tiles_curv_vlim = float(np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in bright_tiles_curv_val])), 99))

print("plotting tiles_curvature ...")
plot_tiles_grid(
    arrays=bright_tiles_curv_val, rows=6, cols=6, name="tiles_curvature", odir=odir, dpi=300,
    vmin=-tiles_curv_vlim, vmax=tiles_curv_vlim, norm="linear", cmap="coolwarm", frame=True,
    cbar_label=r"spectral curvature $\beta$", contour_arrays=bright_tiles_rf_val, contour_levels=[1e-2, 1e-1, 1],
)

# %%
# ===========================================================================
# 2) Spectral index compared to uSARA / AIRI (as eso_compare.py).
# ===========================================================================
usara_1053mhz = fits2array(f"{paper_dir}/fits/ESO137_1053MHz_uSARA.fits") * 1e3 / 3.6
usara_1399mhz = fits2array(f"{paper_dir}/fits/ESO137_1399MHz_uSARA.fits") * 1e3 / 3.6
usara_alpha = compute_spectral_index(usara_1399mhz, usara_1053mhz, 1399e6, 1053e6)
airi_1053mhz = fits2array(f"{paper_dir}/fits/ESO137_1053MHz_AIRI.fits") * 1e3 / 3.6
airi_1399mhz = fits2array(f"{paper_dir}/fits/ESO137_1399MHz_AIRI.fits") * 1e3 / 3.6
airi_alpha = compute_spectral_index(airi_1399mhz, airi_1053mhz, 1399e6, 1053e6)

panel_labels = ["aim-resolve", "AIRI (from Dabbech et al. 22)", "uSARA (from Dabbech et al. 22)"]
compare_dict = dict(norm="linear", cmap=ALPHA_CMAP, cbar=True, cbar_kwargs={"loc": "right"}, ticks=0, dpi=300)

nifty_sky = [to_grid(cube("mean", c["name"])[REF_IDX], c["name"]) for c in COMPONENTS]
nifty_alpha = [to_grid(layer("alpha", "mean", c["name"]), c["name"]) for c in COMPONENTS]

compare_alpha = []
for comp, sky_ref, alpha in zip(COMPONENTS, nifty_sky, nifty_alpha):
    kw = dict(rel_fov=comp["rel_fov"], center=comp["center"])
    n_1053, u_1053, a_1053 = map2component(sky_ref, usara_1053mhz, airi_1053mhz, **kw)
    n_alpha, u_alpha, a_alpha = map2component(alpha, usara_alpha, airi_alpha, **kw)
    f_min = comp["flux_min"]
    compare_alpha.append(dict(
        alpha=[np.where(n_1053 > f_min, n_alpha, np.nan), np.where(a_1053 > f_min, a_alpha, np.nan),
               np.where(u_1053 > f_min, u_alpha, np.nan)],
        flux=[n_1053, a_1053, u_1053],
    ))

for key, c, lvls, hspace in zip(["c1", "c2"], compare_alpha, contour_levels, [-0.95, -1.0]):
    print(f"plotting {key}_alpha ...")
    plot_rows(
        array=c["alpha"], odir=odir, name=f"{key}_alpha", cbar_label=r"spectral index $\alpha$",
        labels=panel_labels, label_color="black", frame=True,
        contour=[{"array": f, "levels": lvls, "colors": "black", "linewidths": 0.5} for f in c["flux"]],
        figsize=(10, 10), vmin=ALPHA_VMIN, vmax=ALPHA_VMAX, cbar_ticks=ALPHA_TICKS, grid_kwargs=dict(hspace=hspace, wspace=0),
        **compare_dict,
    )

# %%
# ===========================================================================
# 3) Region analysis of ESO137-006 (as eso_alpha_vs_flux.py): spectral index vs
#    sky brightness per pixel, coloured by the regions of regions.yml.
# ===========================================================================
regions_config = aim.yaml_load(f"{paper_dir}/regions.yml")
region_names = list(regions_config.keys())
region_colors = [region_color(name) for name in region_names]
order = sorted(range(len(region_names)), key=lambda i: region_order(region_names[i]))
draw_order = sorted(order, key=lambda i: "link" not in region_names[i].lower())

c_ref = layer(FREQ_MHZ[REF_IDX], "mean", "O0")  # native object grid, as the region shapes
c_alpha = layer("alpha", "mean", "O0")
mask = (c_ref > 1e-2) & np.isfinite(c_alpha)
region_id = assign_regions_to_pixels(mask, regions_config, brightness=c_ref)

rgb_image = np.ones(c_ref.shape + (3,))
for rid in draw_order:
    rgb_image[region_id == rid] = mcolors.to_rgb(region_colors[rid])
rgb_image[mask & (region_id == -1)] = mcolors.to_rgb("grey")
x_r, y_r, r_r = c_ref[mask].ravel(), c_alpha[mask].ravel(), region_id[mask].ravel()

ii, jj = np.where(region_id >= 0)
pad_x, pad_y = 0.05 * np.ptp(ii), 0.05 * np.ptp(jj)
rx0, rx1, ry0, ry1 = ii.min() - pad_x, ii.max() + pad_x, jj.min() - pad_y, jj.max() + pad_y
map_aspect = (ry1 - ry0) / (rx1 - rx0)
legend_handles = [Patch(facecolor=region_colors[rid], label=region_label(region_names[rid])) for rid in order]

print("plotting c1_regions ...")
fig_w, sc_aspect = 8.0, 0.7
fig, (ax_map, ax_sc) = plt.subplots(
    2, 1, dpi=200, figsize=(fig_w, fig_w * (map_aspect + sc_aspect)),
    gridspec_kw={"height_ratios": [map_aspect, sc_aspect], "hspace": 0.08 / 3},
)
ax_map.imshow(rgb_image.transpose(1, 0, 2), origin="lower", aspect="auto")
ax_map.contour(c_ref.T, levels=[1e-2, 1e-1, 1, 10], colors="black", linewidths=0.4, origin="lower")
ax_map.set_xlim(rx0, rx1)
ax_map.set_ylim(ry0, ry1)
ax_map.set_xticks([])
ax_map.set_yticks([])
for rid in draw_order:
    sel = r_r == rid
    if np.any(sel):
        ax_sc.scatter(x_r[sel], y_r[sel], s=1, alpha=0.5, color=region_colors[rid], edgecolors="none")
ax_sc.set_xscale("log")
ax_sc.set_xlim(left=1e-2)  # the scatter starts at the 1e-2 flux floor of the mask
ax_sc.set_xlabel(r"sky brightness $I$ [mJy / arcsec$^2$]")
ax_sc.set_ylabel(r"spectral index $\alpha$")
ax_sc.grid(alpha=0.2)
ax_sc.legend(handles=legend_handles, loc="lower right", fontsize=8)
fig.savefig(os.path.join(odir, "c1_regions.png"), bbox_inches="tight")
plt.close(fig)

# %%
# ===========================================================================
# 4) Link profiles along the CST1 (ESO137-006) and ESO137-007 filaments (as
#    eso_c{1,2}_profiles.py): stripes perpendicular to the traced ridge; per
#    sample, alpha and beta from one log-parabola fit of the stripe-mean
#    spectrum, then mean +/- std over the samples.
# ===========================================================================
PROFILES = {
    "c1": dict(
        comp=COMPONENTS[0], name="CST1", anchors=[(0.39, 0.76), (0.525, 0.685), (0.65, 0.59), (0.68, 0.55), (0.70, 0.485)],
        half_width=8.0, ridge_smooth=5.0, ridge_y_window=0.001, dist_unit="arcsec", center="middle",
        img_vmin=5e-3, xlabel_suffix="  (left → right)", z_tick=4,
    ),
    "c2": dict(
        comp=COMPONENTS[1], name="ESO137-007", anchors=[(0.048, 0.12), (0.21, 0.28), (0.41, 0.58), (0.55, 0.65), (0.86, 0.7)],
        half_width=40.0, ridge_smooth=15.0, ridge_y_window=None, dist_unit="arcmin", center="first",
        img_vmin=1e-3, xlabel_suffix="", z_tick=20,
    ),
}
N_SLICES = 100   # number of stripes
N_PERP = 25      # samples across each stripe (profiles)
N_PERP_Z = 101   # samples across each stripe (vs-z plot)
RIDGE_METHOD = "max"
CMAP = "viridis_r"
BAND_FILL = 0.18
LINK_RED = plt.cm.coolwarm(1.0)
LINK_BLUE = plt.cm.coolwarm(0.0)
X_MARGIN = 0.05


def link_profiles(cfg):
    """Stripe geometry and the per-sample stripe statistics of one filament."""
    comp, n = cfg["comp"], cfg["comp"]["name"]
    crop = lambda x: np.asarray(crop_component(x, comp["rel_fov"], comp["center"]))
    flux_c = crop(to_grid(cube("mean", n), n))[REF_IDX]
    nx, ny = flux_c.shape
    pix_arcsec = 2.0 * 3600.0 * comp["rel_fov"][0] / nx

    anchors = ridge_anchors(flux_c, cfg["anchors"], method=RIDGE_METHOD, smooth=cfg["ridge_smooth"],
                            y_window_frac=cfg["ridge_y_window"])
    s_arcsec, centers, coords, seg_ends = perp_slices(anchors, cfg["half_width"], N_SLICES, N_PERP, flux_c.shape, pix_arcsec)

    def sample(arr, crd=coords, n_perp=N_PERP):
        return ndi_map_coordinates(np.asarray(arr, dtype="float64"), crd, order=1, mode="nearest").reshape(N_SLICES, n_perp)

    flux_smean_k, alpha_k, curv_k = [], [], []
    for t in SAMPLE_TAGS:
        flux_cube = crop(to_grid(cube(t, n), n))
        s_freq = np.stack([sample(flux_cube[fi]).mean(axis=1) for fi in range(len(FREQ_MHZ))])
        beta_s, alpha_s, _ = np.polyfit(u_freq, np.log(np.clip(s_freq, 1e-12, None)), 2)
        flux_smean_k.append(s_freq)
        alpha_k.append(alpha_s)
        curv_k.append(beta_s)
    flux_smean_k, alpha_k, curv_k = (np.stack(v) for v in (flux_smean_k, alpha_k, curv_k))

    scale, unit = {"arcsec": (1.0, '["]'), "arcmin": (1.0 / 60.0, "[']")}[cfg["dist_unit"]]
    center = N_SLICES // 2 if cfg["center"] == "middle" else 0
    dist = (s_arcsec - s_arcsec[center]) * scale

    # vs-z geometry (same stripes, finer sampling, NaN outside the crop).
    _, _, coords_z, _ = perp_slices(anchors, cfg["half_width"], N_SLICES, N_PERP_Z, flux_c.shape, pix_arcsec)
    outside_z = ((coords_z[0] < 0) | (coords_z[0] > nx - 1) | (coords_z[1] < 0) | (coords_z[1] > ny - 1)).reshape(N_SLICES, N_PERP_Z)
    alpha_c = crop(to_grid(layer("alpha", "mean", n), n))

    return dict(
        cfg=cfg, flux_c=flux_c, centers=centers, seg_ends=seg_ends, center=center, dist=dist, unit=unit,
        alpha_fit=alpha_k.mean(axis=0), alpha_fit_err=std(alpha_k),
        curv=curv_k.mean(axis=0), curv_err=std(curv_k),
        flux_smean=flux_smean_k.mean(axis=0), flux_smean_err=std(flux_smean_k),
        z_arcsec=np.linspace(-cfg["half_width"], cfg["half_width"], N_PERP_Z),
        flux_z=np.where(outside_z, np.nan, sample(flux_c, coords_z, N_PERP_Z)),
        alpha_z=np.where(outside_z, np.nan, sample(alpha_c, coords_z, N_PERP_Z)),
    )


def plot_link_profiles(p, name, brightness_std=True, curvature=True):
    """Link-profile figure (cutout, spectral index, [curvature], brightness), as in eso_c*_profiles.py."""
    cfg, dist, centers, seg_ends = p["cfg"], p["dist"], p["centers"], p["seg_ends"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(FREQ_MHZ)))
    xc0, xc1 = float(centers[0, 0]), float(centers[-1, 0])
    xm = X_MARGIN * (xc1 - xc0)
    img_x0, img_x1 = xc0 - xm, xc1 + xm
    yv = seg_ends[..., 1]
    ypad = X_MARGIN * float(yv.max() - yv.min())
    img_y0, img_y1 = float(yv.min()) - ypad, float(yv.max()) + ypad
    img_aspect = (img_y1 - img_y0) / (img_x1 - img_x0)

    prof_aspect = 0.5 * 2 / 3
    n_prof = 3 if curvature else 2
    fig = plt.figure(figsize=(8, 8 * (img_aspect + n_prof * prof_aspect)), dpi=200)
    gs = fig.add_gridspec(1 + n_prof, 1, height_ratios=[img_aspect] + [prof_aspect] * n_prof, hspace=0.06)
    ax_img = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharex=ax_a) if curvature else None
    ax_i = fig.add_subplot(gs[-1], sharex=ax_a)

    flux_c = p["flux_c"]
    ax_img.imshow(flux_c.T, origin="lower", cmap="gray_r", norm="log", vmin=cfg["img_vmin"],
                  vmax=float(np.nanmax(flux_c)), aspect="auto")
    ax_img.plot(centers[:, 0], centers[:, 1], color=LINK_RED, lw=0.6, zorder=5)
    for seg in seg_ends:
        ax_img.plot(seg[:, 0], seg[:, 1], color=LINK_RED, lw=0.7, zorder=5)
    ax_img.scatter(*centers[p["center"]], marker="x", color="black", s=45, lw=1.5, zorder=6)
    ax_img.set_xlim(img_x0, img_x1)
    ax_img.set_ylim(img_y0, img_y1)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    ax_a.fill_between(dist, p["alpha_fit"] - p["alpha_fit_err"], p["alpha_fit"] + p["alpha_fit_err"],
                      color=LINK_RED, alpha=0.2, lw=0)
    ax_a.plot(dist, p["alpha_fit"], color=LINK_RED, lw=1.0)
    ax_a.set_ylabel(r"spectral index $\alpha$")
    ax_a.axvline(0.0, color="0.8", lw=0.8, zorder=0)
    ax_a.tick_params(labelbottom=False)

    if curvature:
        curv, curv_err = p["curv"], p["curv_err"]
        ax_c.fill_between(dist, curv - curv_err, curv + curv_err, color=LINK_BLUE, alpha=0.2, lw=0)
        ax_c.plot(dist, curv, color=LINK_BLUE, lw=1.0)
        ax_c.set_ylabel(r"spectral curvature $\beta$")
        ax_c.axvline(0.0, color="0.8", lw=0.8, zorder=0)
        ax_c.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        cmax = 1.05 * float(np.nanmax(np.abs([curv - curv_err, curv + curv_err])))
        ax_c.set_ylim(-cmax, cmax)
        ax_c.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))  # at most 5 labels, symmetric about 0
        ax_c.tick_params(labelbottom=False)

    for fi, mhz in enumerate(FREQ_MHZ):
        ax_i.plot(dist, p["flux_smean"][fi], "-", lw=1.0, color=colors[fi], label=f"{mhz} MHz")
        if brightness_std:
            lo = np.clip(p["flux_smean"][fi] - p["flux_smean_err"][fi], 1e-12, None)
            ax_i.fill_between(dist, lo, p["flux_smean"][fi] + p["flux_smean_err"][fi], color=colors[fi], alpha=0.2, lw=0)
    ax_i.set_yscale("log")
    ax_i.yaxis.set_minor_formatter(NullFormatter())  # label decades only, also for < 1 decade spans
    ax_i.set_ylabel(r"sky brightness $I$ [mJy / arcsec$^2$]")
    ax_i.set_xlabel(f"distance along {cfg['name']} {p['unit']}{cfg['xlabel_suffix']}")
    ax_i.axvline(0.0, color="0.8", lw=0.8, zorder=0)
    ax_i.legend(fontsize=8, ncol=2)

    dm = X_MARGIN * (dist[-1] - dist[0])
    ax_a.set_xlim(dist[0] - dm, dist[-1] + dm)
    fig.savefig(os.path.join(odir, name), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def draw_vs_z(ax, p, values, ylabel, log=False, ymin=None):
    """One line per stripe: `values` against z, colour-coded by the distance along the link."""
    z_arcsec = p["z_arcsec"]
    keep = np.isfinite(values) & (values > 0 if log else True)
    vals = np.where(keep, values, np.nan)
    lines = LineCollection([np.column_stack([z_arcsec, vals[k]]) for k in range(len(vals))],
                           array=p["dist"], cmap=CMAP, linewidths=0.7, alpha=0.75)
    ax.add_collection(lines)
    if log:
        ax.set_yscale("log")
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = (hi / lo) ** 0.04 if log else 0.04 * (hi - lo)
    ax.set_ylim((lo / pad, hi * pad) if log else (lo - pad, hi + pad))
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.set_xlim(z_arcsec[0], z_arcsec[-1])
    ax.axvline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    return lines


def plot_flux_alpha_vs_z(p, name):
    cfg = p["cfg"]
    fig, (ax_f, ax_a) = plt.subplots(2, 1, figsize=(6.6, 8.6), dpi=200, sharex=True, gridspec_kw={"hspace": 0.05})
    lines = draw_vs_z(ax_f, p, p["flux_z"], r"sky brightness $I$ [mJy / arcsec$^2$]", log=True, ymin=1e-3)
    draw_vs_z(ax_a, p, p["alpha_z"], r"spectral index $\alpha$")
    ax_a.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_a.xaxis.set_major_locator(MultipleLocator(cfg["z_tick"]))
    ax_a.set_xlabel(f'distance across {cfg["name"]} ["]')
    pos_f, pos_a = ax_f.get_position(), ax_a.get_position()
    cax = fig.add_axes([pos_f.x1 + 0.015, pos_a.y0, 0.036, pos_f.y1 - pos_a.y0])
    fig.colorbar(lines, cax=cax).set_label(f"distance along {cfg['name']} {p['unit']}")
    fig.align_ylabels([ax_f, ax_a])
    fig.savefig(os.path.join(odir, name), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def plot_curvature_vs_alpha_band(p, name):
    """Curvature vs spectral index per stripe with the connected 1-sigma band."""
    dist, a, b, sa, sb = p["dist"], p["alpha_fit"], p["curv"], p["alpha_fit_err"], p["curv_err"]
    fig, ax = plt.subplots(figsize=(7.4, 5), dpi=200)
    cmap = plt.get_cmap(CMAP)
    norm = plt.Normalize(dist.min(), dist.max())
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)

    def ellipse(i):
        return np.column_stack([a[i] + sa[i] * np.cos(t), b[i] + sb[i] * np.sin(t)])

    for i in range(N_SLICES - 1):
        pts = np.vstack([ellipse(i), ellipse(i + 1)])
        col = np.array(cmap(norm(0.5 * (dist[i] + dist[i + 1]))))[:3]
        ax.add_patch(Polygon(pts[ConvexHull(pts).vertices], closed=True,
                             facecolor=1 - BAND_FILL * (1 - col), edgecolor="none", zorder=1))
    ax.plot(a, b, color="0.6", lw=0.6, zorder=2)
    sc = ax.scatter(a, b, c=dist, cmap=cmap, norm=norm, s=12, alpha=0.9, edgecolors="none", zorder=3)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0.5)
    ax.set_xlabel(r"spectral index $\alpha$")
    ax.set_ylabel(r"spectral curvature $\beta$")
    ax.grid(alpha=0.2)
    ax.autoscale_view()
    fig.colorbar(sc, ax=ax, pad=0.01).set_label(f"distance along {p['cfg']['name']} {p['unit']}")
    fig.savefig(os.path.join(odir, name), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


profiles = {key: link_profiles(cfg) for key, cfg in PROFILES.items()}
plot_link_profiles(profiles["c1"], "c1_profiles.png")
plot_link_profiles(profiles["c2"], "c2_profiles.png")
plot_flux_alpha_vs_z(profiles["c2"], "c2_flux_alpha_vs_z.png")
plot_curvature_vs_alpha_band(profiles["c2"], "c2_curvature_vs_alpha.png")

# %%
# ===========================================================================
# 5) Multi-colour (spectral -> RGB) images (as plot_rgb_freq.py): full sky with
#    the point-source / box markers, the two galaxies and the brightest tiles.
# ===========================================================================
nu_idx = [1, 2, 3, 4, 5, 6]  # even channel positions -> distinct hues; real freqs on the colorbar
color_dict = dict(
    nu_axis_scale="linear", intensity_scale="log", range_min=0.0, range_max=100.0, deslope=1.5,
    normalize_spectrum_enabled=True, normalize_spectrum_boost=7.5, dynamic_range=1e4,
    lambda_min=400.0, lambda_max=700.0,
)

sky = cube("mean", "PRIMARY")  # total sky (sum of all components)
sky_brightness_max = float(np.sum(np.maximum(sky, 0.0), axis=0).max())
sky_spectrum_ref = np.mean(sky, axis=(1, 2))

plot_multi_color(sky, nu_idx, odir=odir, name="sky_rgb_box", marker=box_markers(), **color_dict)
del sky

flux_rgb = [
    np.asarray(crop_component(to_grid(cube("mean", c["name"]), c["name"]), c["rel_fov"], c["center"]))
    for c in COMPONENTS
]
plot_multi_color_column(
    flux_rgb, nu_idx, odir=odir, name="cs_rgb", labels=galaxy_labels, fig_width=10.0,
    brightness_max=sky_brightness_max, spectrum_ref=sky_spectrum_ref, **color_dict,
)

tiles_cube = tile_cube("mean")  # (n_tiles, nfreq, tx, ty)
nt, nf, tx, ty = tiles_cube.shape
tiles_cube = np.asarray(
    map_signal(tiles_cube.reshape(nt * nf, tx, ty), spc_0, spc_1, order=1, vmap_sum=False)
).reshape(nt, nf, tx, ty)
plot_multi_color_grid(
    [tiles_cube[i] for i in tiles_order], nu_idx, rows=6, cols=6, odir=odir, name="tiles_rgb",
    brightness_max=sky_brightness_max, spectrum_ref=sky_spectrum_ref, **color_dict,
)

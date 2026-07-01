# %%
# ---------------------------------------------------------------------------
# GPU / JAX environment setup (must run before importing jax).
# ---------------------------------------------------------------------------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %%
# ---------------------------------------------------------------------------
# Imports.
# ---------------------------------------------------------------------------
import pickle

import jax.numpy as jnp
import numpy as np

import aim_resolve as aim
from aim_resolve.model.util import to_shape, is_val
from aim_resolve.plot.util import plot_figure
from jax import vmap
from jax.scipy.ndimage import map_coordinates

# %%
# ---------------------------------------------------------------------------
# Infrastructure: SignalSpace and mapping of signals between sky sub-fields.
# ---------------------------------------------------------------------------
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


# %%
# ---------------------------------------------------------------------------
# Helpers: PS/box markers, FITS loading, mapping a NIFTy image to the uSARA/AIRI
# component grids, and the two-frequency spectral index.
# ---------------------------------------------------------------------------
def box_markers(cfg, ps_map, grid, it):
    import numpy as np

    from aim_resolve import draw_boxes

    px, py = np.argwhere(ps_map > 0).T
    ps_mrk = dict(x=px, y=py, s=10, c="white", marker="+")
    box_map = draw_boxes(cfg.sections, grid, it)
    ox, oy = np.argwhere(box_map > 0).T
    oj_mrk = dict(x=ox, y=oy, s=0.1, c="white", marker=",")

    return dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)


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


# %%
# ---------------------------------------------------------------------------
# Load the reconstruction: config, sky model, posterior samples, and the
# point-source / object-box markers for the chosen multi-frequency run.
# ---------------------------------------------------------------------------
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"
mf_it = 6

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
CONV_FACTOR = 1000 * AS2RAD**2


opt_yml = f"{dir}/opt/{mf_rec}/opt.yml"
print("load:", opt_yml)

optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)

sky_mf = optim_cfg.instantiate_sec(f"sky.{mf_it}")
print("sky components:", [c.prefix for c in sky_mf.models])

freq = [f"{round(f / 1e6)} MHz" for f in sky_mf.freq]
print("sky freqs:", freq)
ref_freq = freq[1]
print("ref freq:", ref_freq)

with open(f"{dir}/opt/{mf_rec}/last.pkl", "rb") as f:
    samples_mf, *_ = pickle.load(f)
print("samples:", len(samples_mf))

setup_cfg = aim.SetupKLConfig.from_file(opt_yml)
sky_ps = sky_mf.points[0]
ps_map = aim.map_signal(sky_ps.points.grid, sky_ps.grid)(np.ones(sky_ps.shape)).sum(
    axis=0
)
markers_mf = box_markers(setup_cfg, ps_map, sky_mf.grid, mf_it)
print("markers:", [f"{k}: {len(v['x'])}" for k, v in markers_mf.items()])

# %%
# ---------------------------------------------------------------------------
# Shared plot settings and the `plot_tiles_grid` helper (a rows x cols grid of
# image tiles sharing a single bottom colorbar).
# ---------------------------------------------------------------------------
plot_dict = dict(
    odir="/scratch/users/rfuchs/packages/aim-resolve/steering/paper/components",
    norm="log",
    cmap="inferno",
    cbar=False,
    ticks=0,
    dpi=300,
)

panel_labels = [
    "aim-resolve",
    "AIRI (from Dabbech et al. 22)",
    "uSARA (from Dabbech et al. 22)",
]


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
    cbar_height = 0.14 * scale

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
    if cbar_label:
        cbar.set_label(cbar_label)

    plot_figure(figure, odir, name)


# %%
# ---------------------------------------------------------------------------
# Per-tile maps: the 36 brightest sky tiles, shown as reference-frequency
# brightness and spectral index on 6x6 grids.
# ---------------------------------------------------------------------------
import nifty.re as jft

tiles = sky_mf.tiles[0]
tiles_rf = tiles.ref_freq_model
tiles_si = tiles.spectral_index

tiles_rf_val = jft.mean(tuple((tiles_rf(s, map=False)) * CONV_FACTOR for s in samples_mf))
tiles_si_val = jft.mean(tuple(tiles_si(s, map=False) for s in samples_mf))

spc_0 = SignalSpace.build(shape=tiles_rf_val.shape[1:], fov=(2,2))
spc_1 = SignalSpace.build(shape=tiles_rf_val.shape[1:], fov=(1,1))

tiles_rf_val = map_signal(tiles_rf_val, spc_0, spc_1, order=1, vmap_sum=False)
tiles_si_val = map_signal(tiles_si_val, spc_0, spc_1, order=1, vmap_sum=False)


# tiles_rf_val = aim.map_signal(tiles.tiles.grid, tiles.tiles.grid/2)(tiles_rf_val)
print("tiles plotting shape:", tiles_rf_val.shape)


tiles_peak = np.array([np.max(t) for t in tiles_rf_val])
print(np.sort(tiles_peak))
tiles_peak_max = np.max(tiles_peak)
tiles_order = np.argsort(tiles_peak)[::-1][:36]

bright_tiles_rf_val = [tiles_rf_val[i] + 1e-10 for i in tiles_order]
bright_tiles_si_val = [tiles_si_val[i] for i in tiles_order]
bright_tiles_si_val = [np.where(rf > 1e-2, si, np.nan) for rf,si in zip(bright_tiles_rf_val, bright_tiles_si_val)]


print("plotting tiles (6x6) ...")
plot_tiles_grid(
    arrays=bright_tiles_rf_val,
    rows=6,
    cols=6,
    name="tiles_1053mhz",
    odir=plot_dict["odir"],
    dpi=plot_dict["dpi"],
    vmin=1e-3,
    vmax=tiles_peak_max,
    norm="log",
    cmap="inferno",
    frame=False,
    cbar_label="mJy / arcsec$^2$",
)

print("plotting tiles (6x6) ...")
plot_tiles_grid(
    arrays=bright_tiles_si_val,
    rows=6,
    cols=6,
    name="tiles_alpha",
    odir=plot_dict["odir"],
    dpi=plot_dict["dpi"],
    vmin=-4,
    vmax=0,
    norm="linear",
    cmap="coolwarm",
    frame=True,
    cbar_label="spectral index",
    contour_arrays=bright_tiles_rf_val,
    contour_levels=[1e-2, 1e-1, 1],
)

# %%
# ---------------------------------------------------------------------------
# The `plot_column` helper: stack images in a single column, same width, with
# one shared colorbar on the side.
# ---------------------------------------------------------------------------
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
                xytext=(0, -12), textcoords="offset points",
                ha="right", va="top", color=label_color,
            )

        if txt:
            ax.annotate(
                txt, xy=(0.03, 1.0), xycoords="axes fraction",
                xytext=(0, -12), textcoords="offset points",
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


# %%
# ---------------------------------------------------------------------------
# Component maps for both galaxies (c1 + c2): reference-frequency brightness,
# relative uncertainty, spectral index (+ std) and spectral curvature (mean +
# sample-wise uncertainty), stacked with `plot_column`. Pixels below a
# per-component flux floor are masked out.
# ---------------------------------------------------------------------------
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


# Per-component crop geometry and the flux threshold used to mask empty pixels.
galaxy_labels = ["ESO137-006", "ESO137-007"]
components = [
    dict(obj=sky_mf.objects[0], rel_fov=(0.16, 0.08), center=(0, "-0.05deg"), flux_min=1e-2),
    dict(obj=sky_mf.objects[1], rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg"), flux_min=5e-3),
]

# Frequency axis for the log-parabola curvature fit (invariant to the reference).
u_freq = np.log(np.asarray(sky_mf.freq, dtype="float64") / sky_mf.freq[1])

flux_comps, rel_std_comps, alpha_comps, alpha_std_comps = [], [], [], []
curv_comps, curv_std_comps = [], []
for comp in components:
    obj = comp["obj"]

    def to_grid(val):
        return aim.map_signal(obj.grid, sky_mf.grid)(val)

    # Reference-frequency flux mean/std (mJy/arcsec^2) and spectral index mean/std.
    flux_mean = to_grid(samples_mf.mean(obj))[1] * CONV_FACTOR
    flux_std = to_grid(samples_mf.std(obj))[1] * CONV_FACTOR
    rel_std = flux_std / flux_mean
    alpha = to_grid(samples_mf.mean(obj.spectral_index))
    alpha_std = to_grid(samples_mf.std(obj.spectral_index))

    crop = lambda x: crop_component(x, comp["rel_fov"], comp["center"])
    flux_c = crop(flux_mean)
    mask = flux_c > comp["flux_min"]

    # Spectral curvature: mean from the posterior-mean cube, std sample-wise.
    curv_mean = curvature_from_cube(crop(to_grid(samples_mf.mean(obj)) * CONV_FACTOR), u_freq)
    curv_k = np.stack(
        [curvature_from_cube(crop(to_grid(obj(s)) * CONV_FACTOR), u_freq) for s in samples_mf]
    )

    flux_comps.append(flux_c)
    rel_std_comps.append(crop(rel_std))
    alpha_comps.append(np.where(mask, crop(alpha), np.nan))
    alpha_std_comps.append(np.where(mask, crop(alpha_std), np.nan))
    curv_comps.append(np.where(mask, curv_mean, np.nan))
    curv_std_comps.append(np.where(mask, curv_k.std(axis=0), np.nan))

# Black flux contours, used on every map. Per-component levels: c1 (upper
# galaxy, ESO137-006) and c2 (ESO137-007, half the brightness scale).
contour_levels = [[1e-2, 1e-1, 1, 10], [5e-3, 5e-2, 5e-1, 5]]
flux_contours = [
    {"array": f, "levels": lvls, "colors": "black", "linewidths": 0.5}
    for f, lvls in zip(flux_comps, contour_levels)
]

print("plotting mean of the sky brightness ...")
plot_column(
    flux_comps,
    odir=plot_dict["odir"],
    name="cs_1053mhz",
    cmap="inferno",
    norm="log",
    vmin=1e-3,
    vmax=float(max(np.nanmax(f) for f in flux_comps)),
    cbar_label=r"mJy / arcsec$^2$",
    labels=galaxy_labels,
    fig_width=10.0,
    dpi=plot_dict["dpi"],
)

print("plotting std of the sky brightness ...")
plot_column(
    rel_std_comps,
    odir=plot_dict["odir"],
    name="cs_1053mhz_std",
    cmap="inferno",
    norm="log",
    vmin=1e-3,
    frame=True,
    label_color="black",
    contour=flux_contours,
    cbar_label="relative uncertainty",
    labels=galaxy_labels,
    fig_width=10.0,
    dpi=plot_dict["dpi"],
)

print("plotting mean of the spectral index ...")
plot_column(
    alpha_comps,
    odir=plot_dict["odir"],
    name="cs_alpha",
    cmap="coolwarm",
    norm="linear",
    vmin=-4,
    vmax=0,
    frame=True,
    label_color="black",
    contour=flux_contours,
    cbar_label="spectral index",
    labels=galaxy_labels,
    fig_width=10.0,
    dpi=plot_dict["dpi"],
)

print("plotting std of the spectral index ...")
plot_column(
    alpha_std_comps,
    odir=plot_dict["odir"],
    name="cs_alpha_std",
    cmap="coolwarm",
    norm="linear",
    vmax=0.25,
    vmin=0,
    frame=True,
    label_color="black",
    contour=flux_contours,
    cbar_label="relative uncertainty",
    labels=galaxy_labels,
    fig_width=10.0,
    dpi=plot_dict["dpi"],
)

print("plotting mean of the spectral curvature ...")
curv_vlim = float(
    np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in curv_comps])), 99)
)
plot_column(
    curv_comps,
    odir=plot_dict["odir"],
    name="cs_curvature",
    cmap="coolwarm",
    norm="linear",
    vmin=-curv_vlim,
    vmax=curv_vlim,
    frame=True,
    label_color="black",
    contour=flux_contours,
    cbar_label=r"spectral curvature $c$",
    labels=galaxy_labels,
    fig_width=10.0,
    dpi=plot_dict["dpi"],
)

print("plotting std of the spectral curvature ...")
plot_column(
    curv_std_comps,
    odir=plot_dict["odir"],
    name="cs_curvature_std",
    cmap="inferno",
    norm="linear",
    vmin=0,
    frame=True,
    label_color="black",
    contour=flux_contours,
    cbar_label=r"curvature uncertainty $\sigma_c$",
    labels=galaxy_labels,
    fig_width=10.0,
    dpi=plot_dict["dpi"],
)

# %%
# ---------------------------------------------------------------------------
# Full-field reference-frequency sky with the point-source and object-box
# markers overplotted.
# ---------------------------------------------------------------------------
sky_rf_val = samples_mf.mean(sky_mf.ref_freq_model) * CONV_FACTOR

print("plotting ...")
plot_column(
    [sky_rf_val],
    odir=plot_dict["odir"],
    name="sky_1053mhz_box",
    cmap="inferno",
    norm="log",
    vmin=1e-3,
    marker=markers_mf,
    text=ref_freq,
    text_color="white",
    cbar_label="mJy / arcsec$^2$",
    fig_width=10.0,
    dpi=plot_dict["dpi"],
)
# %%
# ---------------------------------------------------------------------------
# Sky zoom-in (central 40% of the FoV) across all 6 frequencies, as a 2x3 tile
# grid with a single bottom colorbar.
# ---------------------------------------------------------------------------
sky_val_mf = samples_mf.mean(sky_mf) * CONV_FACTOR
sky_val = crop_component(sky_val_mf, (0.4, 0.4), (0, 0))

freq_labels = [f"{round(f * 1e-6)} MHz" for f in sky_mf.freq]

print("plotting sky_mf zoom (2x3) ...")
plot_tiles_grid(
    arrays=list(sky_val),
    rows=2,
    cols=3,
    name="sky_mf",
    odir=plot_dict["odir"],
    dpi=plot_dict["dpi"],
    vmin=1e-3,
    vmax=float(np.nanmax(sky_val)),
    norm="log",
    cmap="inferno",
    labels=freq_labels,
    label_color="white",
    label_fontsize=15,
    tile_size=3.0,
    space=0.02,
    scale=2.0,
    cbar_label="mJy / arcsec$^2$",
)
# %%

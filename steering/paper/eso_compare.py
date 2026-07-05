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
# Load the aim-resolve reconstruction: config, sky model, posterior samples and
# the point-source / object-box markers for the chosen multi-frequency run.
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
# Load the reference reconstructions (uSARA and AIRI, Dabbech et al. 2022) at
# 1053 and 1399 MHz from FITS, and their two-frequency spectral index.
# ---------------------------------------------------------------------------
usara_1053mhz = fits2array("/scratch/users/rfuchs/packages/aim-resolve/steering/paper/fits/ESO137_1053MHz_uSARA.fits") * 1e3 / 3.6
print("uSARA @1053 MHz shape:", usara_1053mhz.shape)

usara_1399mhz = fits2array("/scratch/users/rfuchs/packages/aim-resolve/steering/paper/fits/ESO137_1399MHz_uSARA.fits") * 1e3 / 3.6
print("uSARA @1399 MHz shape:", usara_1399mhz.shape)

usara_alpha = compute_spectral_index(usara_1399mhz, usara_1053mhz, 1399e6, 1053e6)
print("uSARA alpha shape:", usara_alpha.shape)


airi_1053mhz = fits2array("/scratch/users/rfuchs/packages/aim-resolve/steering/paper/fits/ESO137_1053MHz_AIRI.fits") * 1e3 / 3.6
print("AIRIs @1053 MHz shape:", airi_1053mhz.shape)

airi_1399mhz = fits2array("/scratch/users/rfuchs/packages/aim-resolve/steering/paper/fits/ESO137_1399MHz_AIRI.fits") * 1e3 / 3.6
print("AIRIs @1399 MHz shape:", airi_1399mhz.shape)

airi_alpha = compute_spectral_index(airi_1399mhz, airi_1053mhz, 1399e6, 1053e6)
print("AIRIs alpha shape:", airi_alpha.shape)

# %%
# ---------------------------------------------------------------------------
# aim-resolve component maps (posterior mean): brightness cube (all freqs) and
# spectral index for both galaxies, mapped onto the full sky grid.
# ---------------------------------------------------------------------------
nifty_o0_sky = samples_mf.mean(sky_mf.objects[0]) * CONV_FACTOR
nifty_o0_sky = aim.map_signal(sky_mf.objects[0].grid, sky_mf.grid)(nifty_o0_sky)
print("NIFTy C1 shape:", nifty_o0_sky.shape)

nifty_o0_alpha = samples_mf.mean(sky_mf.objects[0].spectral_index)
nifty_o0_alpha = aim.map_signal(sky_mf.objects[0].grid, sky_mf.grid)(nifty_o0_alpha)
print("NIFTy C1 alpha shape:", nifty_o0_alpha.shape)

nifty_o1_sky = samples_mf.mean(sky_mf.objects[1]) * CONV_FACTOR
nifty_o1_sky = aim.map_signal(sky_mf.objects[1].grid, sky_mf.grid)(nifty_o1_sky)
print("NIFTy C2 shape:", nifty_o1_sky.shape)

nifty_o1_alpha = samples_mf.mean(sky_mf.objects[1].spectral_index)
nifty_o1_alpha = aim.map_signal(sky_mf.objects[1].grid, sky_mf.grid)(nifty_o1_alpha)
print("NIFTy C2 alpha shape:", nifty_o1_alpha.shape)

# %%
# ---------------------------------------------------------------------------
# The `plot_rows` helper: stack 2D arrays in one column (each the same width),
# sharing a single colorbar; mirrors the styling used across these plots.
# ---------------------------------------------------------------------------
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
# Shared plot settings and the three-row panel labels (aim-resolve / AIRI /
# uSARA).
# ---------------------------------------------------------------------------
plot_dict = dict(
    odir="/scratch/users/rfuchs/packages/aim-resolve/steering/paper/compare",
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


# %%
# ---------------------------------------------------------------------------
# Galaxy 1 (ESO137-006) brightness at 1053 MHz: aim-resolve vs AIRI vs uSARA.
# ---------------------------------------------------------------------------
nifty_c1_1053mhz, usara_c1_1053mhz, airi_c1_1053mhz = map2component(nifty_o0_sky[1], usara_1053mhz, airi_1053mhz, rel_fov=(0.16, 0.08), center=(0, "-0.05deg"))

plot_rows(
    array=[nifty_c1_1053mhz, airi_c1_1053mhz + 1e-16, usara_c1_1053mhz + 1e-16],
    name="c1_1053mhz",
    cbar_label=r"mJy/arcsec$^2$",
    labels=panel_labels,
    figsize=(10,10),
    **plot_dict | dict(
        vmin=1e-3,
        vmax=nifty_c1_1053mhz.max(),
        norm="log",
        cbar=True,
        cbar_kwargs={"loc": "right"},
        grid_kwargs=dict(hspace=-0.95, wspace=0),
    ),
)

# %%
# ---------------------------------------------------------------------------
# Galaxy 1 (ESO137-006) brightness at 1399 MHz: aim-resolve vs AIRI vs uSARA.
# ---------------------------------------------------------------------------
nifty_c1_1399mhz, usara_c1_1399mhz, airi_c1_1399mhz = map2component(nifty_o0_sky[4], usara_1399mhz, airi_1399mhz, rel_fov=(0.16, 0.08), center=(0, "-0.05deg"))

plot_rows(
    array=[nifty_c1_1399mhz, airi_c1_1399mhz + 1e-16, usara_c1_1399mhz + 1e-16],
    name="c1_1399mhz",
    cbar_label=r"mJy/arcsec$^2$",
    labels=panel_labels,
    figsize=(10,10),
    **plot_dict | dict(
        vmin=1e-3,
        vmax=nifty_c1_1399mhz.max(),
        norm="log",
        cbar=True,
        cbar_kwargs={"loc": "right"},
        grid_kwargs=dict(hspace=-0.95, wspace=0),
    ),
)

# %%
# ---------------------------------------------------------------------------
# Galaxy 1 (ESO137-006) spectral index: aim-resolve vs AIRI vs uSARA (masked
# below a flux floor; black brightness contours).
# ---------------------------------------------------------------------------
nifty_c1_alpha, usara_c1_alpha, airi_c1_alpha = map2component(nifty_o0_alpha, usara_alpha, airi_alpha, rel_fov=(0.16, 0.08), center=(0, "-0.05deg"))

alpha_min = 1e-2
nifty_c1_alpha = np.where(nifty_c1_1053mhz > alpha_min, nifty_c1_alpha, np.nan)
usara_c1_alpha = np.where(usara_c1_1053mhz > alpha_min, usara_c1_alpha, np.nan)
airi_c1_alpha = np.where(airi_c1_1053mhz > alpha_min, airi_c1_alpha, np.nan)

# Shared spectral-index colour limits: the min/max of the not-masked NIFTy
# alpha over BOTH components, so the c1 and c2 alpha plots use identical
# vmin/vmax. The c2 NIFTy maps are computed here (same crop/mask as below).
_n_c2_1053 = map2component(nifty_o1_sky[1], usara_1053mhz, airi_1053mhz, rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg"))[0]
_n_c2_alpha = map2component(nifty_o1_alpha, usara_alpha, airi_alpha, rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg"))[0]
_n_c2_alpha = np.where(_n_c2_1053 > 5e-3, _n_c2_alpha, np.nan)
ALPHA_VMIN = float(min(np.nanmin(nifty_c1_alpha), np.nanmin(_n_c2_alpha)))
ALPHA_VMAX = float(max(np.nanmax(nifty_c1_alpha), np.nanmax(_n_c2_alpha)))
print(f"shared alpha vmin/vmax (NIFTy, both components): {ALPHA_VMIN:.3f} / {ALPHA_VMAX:.3f}")

contours = [{"array": c1, "levels": [1e-2, 1e-1, 1, 10], "colors": "black", "linewidths": 0.5} for c1 in [nifty_c1_1053mhz, airi_c1_1053mhz, usara_c1_1053mhz]]

plot_rows(
    array=[nifty_c1_alpha, airi_c1_alpha, usara_c1_alpha],
    name="c1_alpha",
    cbar_label="spectral index",
    labels=panel_labels,
    label_color="black",
    frame=True,
    contour=contours,
    figsize=(10,10),
    **plot_dict | dict(
        vmin=ALPHA_VMIN,
        vmax=ALPHA_VMAX,
        norm="linear",
        cmap="coolwarm",
        cbar=True,
        cbar_kwargs={"loc": "right"},
        grid_kwargs=dict(hspace=-0.95, wspace=0),
    ),
)

# %%
# ---------------------------------------------------------------------------
# Galaxy 2 (ESO137-007) brightness at 1053 MHz: aim-resolve vs AIRI vs uSARA.
# ---------------------------------------------------------------------------
nifty_c2_1053mhz, usara_c2_1053mhz, airi_c2_1053mhz = map2component(nifty_o1_sky[1], usara_1053mhz, airi_1053mhz, rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg"))

plot_rows(
    array=[nifty_c2_1053mhz, airi_c2_1053mhz + 1e-16, usara_c2_1053mhz + 1e-16],
    name="c2_1053mhz",
    cbar_label=r"mJy/arcsec$^2$",
    labels=panel_labels,
    figsize=(10,10),
    **plot_dict | dict(
        vmin=5e-4,
        vmax=nifty_c2_1053mhz.max(),
        norm="log",
        cbar=True,
        cbar_kwargs={"loc": "right"},
        grid_kwargs=dict(hspace=-1.0, wspace=0),
    ),
)

# %%
# ---------------------------------------------------------------------------
# Galaxy 2 (ESO137-007) brightness at 1399 MHz: aim-resolve vs AIRI vs uSARA.
# ---------------------------------------------------------------------------
nifty_c2_1399mhz, usara_c2_1399mhz, airi_c2_1399mhz = map2component(nifty_o1_sky[4], usara_1399mhz, airi_1399mhz, rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg"))

plot_rows(
    array=[nifty_c2_1399mhz, airi_c2_1399mhz + 1e-16, usara_c2_1399mhz + 1e-16],
    name="c2_1399mhz",
    cbar_label=r"mJy/arcsec$^2$",
    labels=panel_labels,
    figsize=(10,10),
    **plot_dict | dict(
        vmin=5e-4,
        vmax=nifty_c2_1399mhz.max(),
        norm="log",
        cbar=True,
        cbar_kwargs={"loc": "right"},
        grid_kwargs=dict(hspace=-1.0, wspace=0),
    ),
)

# %%
# ---------------------------------------------------------------------------
# Galaxy 2 (ESO137-007) spectral index: aim-resolve vs AIRI vs uSARA (masked
# below a flux floor; black brightness contours).
# ---------------------------------------------------------------------------
nifty_c2_alpha, usara_c2_alpha, airi_c2_alpha = map2component(nifty_o1_alpha, usara_alpha, airi_alpha, rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg"))

alpha_min = 5e-3
nifty_c2_alpha = np.where(nifty_c2_1053mhz > alpha_min, nifty_c2_alpha, np.nan)
usara_c2_alpha = np.where(usara_c2_1053mhz > alpha_min, usara_c2_alpha, np.nan)
airi_c2_alpha = np.where(airi_c2_1053mhz > alpha_min, airi_c2_alpha, np.nan) 

contours = [{"array": c1, "levels": [5e-3, 5e-2, 5e-1, 5], "colors": "black", "linewidths": 0.5} for c1 in [nifty_c2_1053mhz, airi_c2_1053mhz, usara_c2_1053mhz]]

plot_rows(
    array=[nifty_c2_alpha, airi_c2_alpha, usara_c2_alpha],
    name="c2_alpha",
    cbar_label="spectral index",
    labels=panel_labels,
    label_color="black",
    frame=True,
    contour=contours,
    figsize=(10,10),
    **plot_dict | dict(
        vmin=ALPHA_VMIN,
        vmax=ALPHA_VMAX,
        norm="linear",
        cmap="coolwarm",
        cbar=True,
        cbar_kwargs={"loc": "right"},
        grid_kwargs=dict(hspace=-1.0, wspace=0),
    ),
)

# %%

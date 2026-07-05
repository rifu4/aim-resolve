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
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from scipy.ndimage import map_coordinates as ndi_map_coordinates

import aim_resolve as aim
from aim_resolve.model.util import is_val, to_shape

# %%
# ---------------------------------------------------------------------------
# Infrastructure copied from `eso_components.py` (SignalSpace + signal mapping
# and the component crop) so this script is self-contained.
# ---------------------------------------------------------------------------


class SignalSpace:
    """Class to represent a signal space at a specific location in the sky."""

    def __init__(self, shape, distances, center=(0.0, 0.0), n_copies=1):
        self.shape = shape
        self.distances = distances
        self.center = center
        self.n_copies = n_copies

    @classmethod
    def build(cls, *, shape, distances=None, fov=None, center=None, n_copies=1):
        shp = to_shape(shape, (2,), "int64")
        dis = to_shape(distances, (2,), "float64")
        fov = to_shape(fov, (2,), "float64")
        cen = to_shape(center, (n_copies, 2), "float64")

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
        return vmap(space_coos, in_axes=(None, None, 0, 0))(self.shp, self.dis, self.cen)


def space_coos(shp, dis, cen):
    coos = jnp.indices(shp).astype(float)
    coos_T = coos.T.reshape(-1, 2)
    coos_T -= 0.5 * (shp - 1)
    coos_T *= dis
    coos_T += cen
    return coos_T.reshape(coos.T.shape).T


def map_signal(x, in_space, out_space, order=0, vmap_sum=True):
    if x.ndim == 2:
        return map_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
    if in_space.n_copies > 1:
        vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, 0, None, None))
        res = vmap_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
    else:
        vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, None, None, None))
        res = vmap_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
    return jnp.sum(res, axis=0) if vmap_sum else res


def map_one_signal(x, in_dis, in_cen, out_coos, order=0):
    x = jnp.asarray(x)
    out_coos = jnp.asarray(out_coos)
    out_coos_T = out_coos.T.reshape(-1, 2)
    out_coos_T -= in_cen
    out_coos_T /= in_dis
    out_coos_T += 0.5 * (jnp.array(x.shape) - 1)
    out_coos = out_coos_T.reshape(out_coos.T.shape).T
    return map_coordinates(x, out_coos, order)


def crop_component(nifty_array, rel_fov, center):
    """Crop a full-grid (2deg) nifty image to a component sub-field of view."""
    nifty_array = np.asarray(nifty_array)
    rel_fov = np.array(rel_fov)
    space = SignalSpace.build(shape=nifty_array.shape[-2:], fov=("2deg", "2deg"))
    sub = SignalSpace.build(
        shape=space.shp * rel_fov, fov=space.fov * rel_fov, center=center
    )
    return np.asarray(map_signal(nifty_array, space, sub, vmap_sum=False))


# %%
# ---------------------------------------------------------------------------
# Load the reconstruction (same run as `eso_components.py`).
# ---------------------------------------------------------------------------
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"
mf_it = 6

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
CONV_FACTOR = 1000 * AS2RAD**2

odir = "/scratch/users/rfuchs/packages/aim-resolve/steering/paper/profiles"

opt_yml = f"{dir}/opt/{mf_rec}/opt.yml"
print("load:", opt_yml)

optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)
sky_mf = optim_cfg.instantiate_sec(f"sky.{mf_it}")

freq = [f"{round(f / 1e6)} MHz" for f in sky_mf.freq]
ref_freq = freq[1]
print("sky freqs:", freq, "| ref freq:", ref_freq)

with open(f"{dir}/opt/{mf_rec}/last.pkl", "rb") as f:
    samples_mf, *_ = pickle.load(f)
print("samples:", len(samples_mf))

# %%
# ---------------------------------------------------------------------------
# ESO137-007 (objects[1]): posterior-mean reference-frequency brightness, used
# for the link cutout image and the stripe geometry. The per-sample brightness
# and spectral-index maps are built inside the sample loop below.
# ---------------------------------------------------------------------------
obj = sky_mf.objects[1]
REL_FOV = (0.25, 0.09)
CENTER = ("0.18deg", "0.31deg")


def to_grid(val):
    return aim.map_signal(obj.grid, sky_mf.grid)(val)


flux_all = to_grid(samples_mf.mean(obj)) * CONV_FACTOR  # (nfreq, H, W)
flux_c_all = crop_component(flux_all, REL_FOV, CENTER)  # (nfreq, nx, ny)
flux_c = np.asarray(flux_c_all[1])  # reference frequency, for display / geometry
freq_mhz = [round(f / 1e6) for f in sky_mf.freq]
print("cropped shape (nfreq, nx, ny):", flux_c_all.shape)

# Physical pixel scale of the crop (square pixels; full FoV is 2 deg).
nx, ny = flux_c.shape
pix_arcsec = 2.0 * 3600.0 * REL_FOV[0] / nx
print(f"pixel scale: {pix_arcsec:.2f} arcsec/pix")

# %%
# ---------------------------------------------------------------------------
# Define the link (left -> right) and the perpendicular slices.
#
# The link is a polyline through a list of anchor points (>= 2: start and end).
# Consecutive anchors are joined by straight segments, so adding intermediate
# anchors lets the link bend. Each anchor is in *fractions of the displayed
# image* (origin lower):
#   x = 0 left edge ... 1 right edge   (horizontal, array axis 0)
#   y = 0 bottom    ... 1 top          (vertical,   array axis 1)
# Measurement stripes are placed at equal spacing *along* the polyline (equal
# arc length) and run perpendicular to the local segment. Pixels are square, so
# the geometry is taken directly in pixel space and the equal-aspect
# `link_slices` plot preserves it. TUNE the anchors against that plot.
# ---------------------------------------------------------------------------
LINK_ANCHORS = [           # polyline anchors (fractional x, y); >= 2 points
    (0.048, 0.12),          #   left (higher) start
    (0.21, 0.28),
    (0.41, 0.58),
    (0.55, 0.65),
    (0.86, 0.7),            #   right end
]
HALF_WIDTH_ARCSEC = 40.0   # stripe half-length, perpendicular to the link
N_SLICES = 100              # number of stripes (odd -> exact center stripe at d=0)
N_PERP = 25                # samples across each perpendicular stripe

# Auto-trace the link ridge instead of hand-tuning the intermediate anchors:
# use the first/last LINK_ANCHORS x as the start/end column, and for every
# x-pixel in between pick the y-pixel by RIDGE_METHOD --
#   "max"    : the brightest y in that column
#   "median" : the flux-weighted median y (cumulative brightness = 50%)
# RIDGE_SMOOTH gaussian-smooths the traced y(x) [pixels; 0 disables] so the
# perpendicular stripes don't jitter with the per-pixel ridge.
RIDGE_METHOD = "max"
RIDGE_SMOOTH = 15.0

# Per-stripe spectral index: how to reduce alpha across the perpendicular
# samples. True -> mean weighted by the (ref-freq) sky brightness at each pixel;
# False -> plain unweighted mean over the stripe.
ALPHA_WEIGHTED = False


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


def ridge_anchors(flux, x_start_frac, x_end_frac, *, method="max", smooth=0.0):
    """Trace the link ridge as (x, y) fractional anchors, one per x-pixel column.

    For each x-column between the start and end fraction, pick the y-pixel by
    `method`: "max" (brightest pixel) or "median" (flux-weighted median position,
    where the cumulative column brightness reaches 50%). `smooth` gaussian-blurs
    the traced y(x) in pixels (0 disables). Returns (n, 2) fractional (x, y)
    anchors, matching the `LINK_ANCHORS` convention fed to `perp_slices`.
    """
    flux = np.clip(np.asarray(flux, dtype="float64"), 0.0, None)  # (nx, ny)
    nx, ny = flux.shape
    ix0 = int(round(x_start_frac * (nx - 1)))
    ix1 = int(round(x_end_frac * (nx - 1)))
    step = 1 if ix1 >= ix0 else -1
    xs = np.arange(ix0, ix1 + step, step)
    y_idx = np.arange(ny, dtype="float64")

    ys = np.empty(xs.size, dtype="float64")
    for k, ix in enumerate(xs):
        col = flux[ix]
        total = col.sum()
        if method == "max":
            ys[k] = float(np.argmax(col))
        elif method == "median":
            ys[k] = float(np.interp(0.5, np.cumsum(col) / total, y_idx)) if total > 0 else float(np.argmax(col))
        else:
            raise ValueError("RIDGE_METHOD must be 'max' or 'median'")

    if smooth and smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        ys = gaussian_filter1d(ys, float(smooth), mode="nearest")

    return np.column_stack([xs / (nx - 1), ys / (ny - 1)])


# Start/end x from the first/last LINK_ANCHORS; the y-path in between is traced.
link_anchors = ridge_anchors(
    flux_c, LINK_ANCHORS[0][0], LINK_ANCHORS[-1][0], method=RIDGE_METHOD, smooth=RIDGE_SMOOTH,
)
print(f"ridge anchors: {len(link_anchors)} pts (method={RIDGE_METHOD}, smooth={RIDGE_SMOOTH})")

s_arcsec, centers, coords, seg_ends = perp_slices(
    link_anchors, HALF_WIDTH_ARCSEC, N_SLICES, N_PERP, flux_c.shape, pix_arcsec,
)

# Sample brightness / spectral index / its uncertainty along every slice.
def sample(arr):
    return ndi_map_coordinates(
        np.asarray(arr, dtype="float64"), coords, order=1, mode="nearest"
    ).reshape(N_SLICES, N_PERP)


# ---------------------------------------------------------------------------
# Sample-wise uncertainty propagation for the 1D profiles. Instead of
# propagating the per-pixel std maps (which would require assuming the pixels
# along a stripe are independent), we resample *each* posterior sample along the
# link and take the spread of the resulting profiles. This captures the full
# spatial correlation between pixels, and for the flux-weighted alpha also the
# flux/alpha covariance, since every sample is weighted by its own brightness.
# ---------------------------------------------------------------------------
def ref_maps(s):
    """Per-sample cropped brightness cube [mJy/arcsec^2] (all freqs) and spectral index."""
    flux_full = to_grid(obj(s)) * CONV_FACTOR
    alpha_full = to_grid(obj.spectral_index(s))
    flux_cube = crop_component(flux_full, REL_FOV, CENTER)  # (nfreq, nx, ny)
    alpha_ref = crop_component(alpha_full, REL_FOV, CENTER)
    return flux_cube, alpha_ref


# ---------------------------------------------------------------------------
# Sample loop: for every GeoVI sample build the stripe profiles, then take the
# mean and std ACROSS samples -- the correct posterior uncertainty (reduce each
# sample to its stripe scalar, then spread over samples). Per stripe:
#   alpha_wmean : flux-weighted mean spectral index (ref-freq brightness weights)
#   flux_smean  : stripe-mean sky brightness at each of the 6 frequencies
#   curv        : spectral curvature from a log-parabola fit of the stripe-mean
#                 spectrum,  ln S = a + alpha * u + c * u^2,  u = ln(nu/nu_ref);
#                 c > 0 flattening (blend), c < 0 steepening (single aged pop.).
# ---------------------------------------------------------------------------
u_freq = np.log(np.asarray(sky_mf.freq, dtype="float64") / sky_mf.freq[1])

alpha_wmean_k, flux_smean_k, curv_k = [], [], []
for s in samples_mf:
    flux_cube, alpha_ref = ref_maps(s)
    flux_per_freq = [sample(flux_cube[fi]) for fi in range(len(freq_mhz))]
    s_freq = np.stack([f.mean(axis=1) for f in flux_per_freq])  # (nfreq, n_slices)
    flux_smean_k.append(s_freq)

    # spectral index per stripe: flux-weighted (ref-freq brightness weights) or
    # plain mean over the perpendicular samples, per ALPHA_WEIGHTED.
    alpha_k = sample(alpha_ref)
    if ALPHA_WEIGHTED:
        w_k = np.clip(flux_per_freq[1], 0.0, None)
        wsum_k = np.where(w_k.sum(axis=1) > 0, w_k.sum(axis=1), np.nan)
        alpha_wmean_k.append((w_k * alpha_k).sum(axis=1) / wsum_k)
    else:
        alpha_wmean_k.append(alpha_k.mean(axis=1))

    # spectral curvature from the stripe-mean spectrum.
    ln_s = np.log(np.clip(s_freq, 1e-12, None))
    curv_k.append(np.polyfit(u_freq, ln_s, 2)[0])

alpha_wmean_k = np.stack(alpha_wmean_k)   # (n_samples, n_slices)
flux_smean_k = np.stack(flux_smean_k)     # (n_samples, nfreq, n_slices)
curv_k = np.stack(curv_k)                 # (n_samples, n_slices)

# Posterior mean and 1-sigma spread across the samples (ddof=1).
alpha_wmean = alpha_wmean_k.mean(axis=0)
alpha_wmean_err = alpha_wmean_k.std(axis=0, ddof=1)
flux_smean = flux_smean_k.mean(axis=0)          # (nfreq, n_slices)
flux_smean_err = flux_smean_k.std(axis=0, ddof=1)
curv = curv_k.mean(axis=0)
curv_err = curv_k.std(axis=0, ddof=1)

# Distance along the filament measured from the LEFT starting anchor: d = 0 at
# the first stripe, which is where the cross (cutout) and the grey vertical lines
# (profile panels) are drawn. The distance axis unit is independent of the
# surface-brightness unit (which stays mJy/arcsec^2).
DIST_UNIT = "arcmin"  # "arcsec" or "arcmin" for the distance axis
_DIST_SCALE = {"arcsec": 1.0, "arcmin": 1.0 / 60.0}[DIST_UNIT]
_DIST_UNIT_LABEL = {"arcsec": '["]', "arcmin": "[']"}[DIST_UNIT]

center = 0
dist = (s_arcsec - s_arcsec[center]) * _DIST_SCALE

# %%
# ---------------------------------------------------------------------------
# Helper to draw the perpendicular slice "rungs" (red), the centerline and the
# central cross, in the array index frame (element (i, j) shown at x=i, y=j).
# ---------------------------------------------------------------------------
def draw_slices(ax, color="red", lw=1.0, cross_color="black", cross=True):
    ax.plot(centers[:, 0], centers[:, 1], color=color, lw=0.6, zorder=5)
    for seg in seg_ends:
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, zorder=5)
    if cross:
        ax.scatter(
            *centers[center], marker="x", color=cross_color, s=45, lw=1.5, zorder=6
        )


def zoom_to_filament(ax):
    pts = seg_ends.reshape(-1, 2)
    (xmin, ymin), (xmax, ymax) = pts.min(0), pts.max(0)
    pad = 0.10 * max(xmax - xmin, ymax - ymin)  # small margin around the stripes
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)


# %%
# ---------------------------------------------------------------------------
# Plot 1: link profiles. Flux-weighted spectral index (mean +/- std) on top,
# stripe-mean sky brightness per frequency (mean only) below, and the link
# cutout with the perpendicular slices on the right. All uncertainties are the
# spread across the GeoVI samples.
# ---------------------------------------------------------------------------
os.makedirs(odir, exist_ok=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(freq_mhz)))
LINK_RED = plt.cm.coolwarm(1.0)  # coolwarm red end: stripes + spectral-index curve
LINK_BLUE = plt.cm.coolwarm(0.0)  # coolwarm blue end: spectral-curvature curve

# Align the top cutout with the profile panels below: the first/last stripe
# centres and the start/end of the curves are inset by the same 5% of the span
# on each side, so the leftmost/rightmost stripes line up with the curve ends.
X_MARGIN = 0.05  # fraction of the span left free on each side (cutout + curves)

_xc0, _xc1 = float(centers[0, 0]), float(centers[-1, 0])
_xm = X_MARGIN * (_xc1 - _xc0)
img_x0, img_x1 = _xc0 - _xm, _xc1 + _xm
_yv = seg_ends[..., 1]
_ypad = X_MARGIN * float(_yv.max() - _yv.min())  # same fractional margin as x
img_y0, img_y1 = float(_yv.min()) - _ypad, float(_yv.max()) + _ypad
img_aspect = (img_y1 - img_y0) / (img_x1 - img_x0)


REF_IDX = 1  # reference-frequency index (ref_freq = freq[1])


def plot_link_profiles(name, brightness_std=False, curvature=False, ref_only=False):
    """Link-profile figure: cutout on top, then profile panels below.

    Panels are: spectral index, [spectral curvature if `curvature`], brightness.
    `brightness_std` toggles the std band on the brightness panel; `ref_only`
    shows only the reference-frequency brightness instead of all frequencies.
    """
    prof_aspect = 0.5 * 2 / 3  # height / width of each profile panel (2/3 shorter)
    n_prof = 3 if curvature else 2
    fig = plt.figure(figsize=(8, 8 * (img_aspect + n_prof * prof_aspect)), dpi=200)
    gs = fig.add_gridspec(
        1 + n_prof, 1, height_ratios=[img_aspect] + [prof_aspect] * n_prof, hspace=0.06,
    )
    ax_img = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharex=ax_a) if curvature else None
    ax_i = fig.add_subplot(gs[-1], sharex=ax_a)

    # link cutout with the perpendicular slices on top (same width as the panels)
    ax_img.imshow(
        flux_c.T, origin="lower", cmap="gray_r",
        norm="log", vmin=1e-3, vmax=float(np.nanmax(flux_c)), aspect="auto",
    )
    draw_slices(ax_img, color=LINK_RED, lw=0.7, cross=True, cross_color="black")
    ax_img.set_xlim(img_x0, img_x1)
    ax_img.set_ylim(img_y0, img_y1)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # spectral index (flux-weighted; mean +/- std across samples)
    ax_a.fill_between(
        dist, alpha_wmean - alpha_wmean_err, alpha_wmean + alpha_wmean_err,
        color=LINK_RED, alpha=0.2, lw=0,
    )
    ax_a.plot(dist, alpha_wmean, color=LINK_RED, lw=1.0)
    ax_a.set_ylabel(r"spectral index $\alpha$")
    ax_a.axvline(0.0, color="0.8", lw=0.8, zorder=0)
    ax_a.tick_params(labelbottom=False)

    # spectral curvature (mean +/- std across samples), coolwarm blue end
    if curvature:
        ax_c.fill_between(
            dist, curv - curv_err, curv + curv_err, color=LINK_BLUE, alpha=0.2, lw=0,
        )
        ax_c.plot(dist, curv, color=LINK_BLUE, lw=1.0)
        ax_c.set_ylabel(r"spectral curvature $c$")
        ax_c.axvline(0.0, color="0.8", lw=0.8, zorder=0)
        ax_c.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        # symmetric y-limits centred on zero (same |vmin| = |vmax|)
        cmax = 1.05 * float(np.nanmax(np.abs([curv - curv_err, curv + curv_err])))
        ax_c.set_ylim(-cmax, cmax)
        ax_c.tick_params(labelbottom=False)

    # stripe-mean sky brightness (log; mean, optionally +/- std). `ref_only`
    # keeps just the reference frequency; otherwise every frequency is shown.
    freq_indices = [REF_IDX] if ref_only else range(len(freq_mhz))
    for fi in freq_indices:
        mhz = freq_mhz[fi]
        ax_i.plot(dist, flux_smean[fi], "-", lw=1.0, color=colors[fi], label=f"{mhz} MHz")
        if brightness_std:
            lo = np.clip(flux_smean[fi] - flux_smean_err[fi], 1e-12, None)
            ax_i.fill_between(
                dist, lo, flux_smean[fi] + flux_smean_err[fi],
                color=colors[fi], alpha=0.2, lw=0,
            )
    ax_i.set_yscale("log")
    ax_i.set_ylabel(r"$I_\mathrm{mean}$ [mJy / arcsec$^2$]")
    ax_i.set_xlabel(f"distance along ESO137-007 {_DIST_UNIT_LABEL}")
    ax_i.axvline(0.0, color="0.8", lw=0.8, zorder=0)
    ax_i.legend(fontsize=8, ncol=2)

    # same 5% side margin as the cutout, so the curve ends align with the stripes.
    _dm = X_MARGIN * (dist[-1] - dist[0])
    ax_a.set_xlim(dist[0] - _dm, dist[-1] + _dm)

    fig.savefig(os.path.join(odir, name), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


plot_link_profiles("c2_profiles.png", brightness_std=True)

# %%
# ---------------------------------------------------------------------------
# Plot 2: same as Plot 1 but with an extra spectral-curvature panel (coolwarm
# blue) between the spectral index and the brightness.  c > 0 flattening
# (blend), c < 0 steepening (aged). Uncertainties are the spread over samples.
# ---------------------------------------------------------------------------
plot_link_profiles("c2_profiles_curvature.png", brightness_std=True, curvature=True)

# %%

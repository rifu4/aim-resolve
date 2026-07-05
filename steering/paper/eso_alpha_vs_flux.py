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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from matplotlib.patches import Patch

import aim_resolve as aim
from aim_resolve.model.util import is_val, to_shape

# %%
# ---------------------------------------------------------------------------
# Load the reconstruction (same run as `eso_analysis.py` / `eso_components.py`).
# ---------------------------------------------------------------------------
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"
mf_it = 6

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
CONV_FACTOR = 1000 * AS2RAD**2

odir = "/scratch/users/rfuchs/packages/aim-resolve/steering/paper/alpha_vs_flux"

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

os.makedirs(odir, exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# Spectral index vs sky brightness (per pixel), separately for each component
# (c1 = ESO137-006, c2 = ESO137-007): every pixel above the flux floor as a
# single-color scatter. Adapted from `eso_mf_zoom_talk.py`.
# ---------------------------------------------------------------------------
av_comps = [sky_mf.objects[0], sky_mf.objects[1]]
av_labels = ["ESO137-006 (c1)", "ESO137-007 (c2)"]
av_ref = [samples_mf.mean(c.ref_freq_model) * CONV_FACTOR for c in av_comps]
av_alpha = [samples_mf.mean(c.spectral_index) for c in av_comps]
av_flux_min = [1e-2, 5e-3]  # mJy/arcsec^2 floor for the scatter

for idx, (c_ref, c_alpha, lab, f_min) in enumerate(zip(av_ref, av_alpha, av_labels, av_flux_min), start=1):
    mask = (c_ref > f_min) & np.isfinite(c_alpha)
    x = np.asarray(c_ref)[mask].ravel()  # brightness
    y = np.asarray(c_alpha)[mask].ravel()  # spectral index

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.scatter(x, y, s=2, alpha=0.35, color="k", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel(r"sky brightness [mJy / arcsec$^2$]")
    ax.set_ylabel(r"spectral index $\alpha$")
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(odir, f"c{idx}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved c{idx}.png")

# %%
# ---------------------------------------------------------------------------
# Region analysis of c1 (ESO137-006). Regions are read from regions.yml
# (any number of entries — e.g. "left lobe", "link", "right lobe"); each region
# gets its own colour (left -> coolwarm blue, right -> red, link(s) -> orange).
# Produced: spectral index vs sky brightness coloured by region, with a zoomed
# region-map inset. Adapted from eso_mf_physics.py.
# regions.yml is defined for c1 only, so this is not produced for c2.
# ---------------------------------------------------------------------------
REGIONS_YML = "/scratch/users/rfuchs/packages/aim-resolve/steering/paper/regions.yml"


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


regions_config = aim.yaml_load(REGIONS_YML)
region_names = list(regions_config.keys())
n_regions = len(region_names)


# Fixed colour per region role: left lobe -> coolwarm blue, right lobe ->
# coolwarm red, link(s) -> orange; anything else falls back to grey.
def region_color(name):
    n = name.lower()
    if "link" in n:
        return mcolors.to_rgba("orange")
    if "left" in n:
        return plt.cm.coolwarm(0.0)
    if "right" in n:
        return plt.cm.coolwarm(1.0)
    return mcolors.to_rgba("grey")


region_colors = [region_color(name) for name in region_names]


# Draw / legend order: right lobe, then left lobe, then link(s) (drawn last, on
# top). Independent of the order the regions appear in regions.yml.
def region_order(name):
    n = name.lower()
    if "right" in n:
        return 0
    if "left" in n:
        return 1
    if "link" in n:
        return 2
    return 3


order = sorted(range(n_regions), key=lambda i: region_order(region_names[i]))

# Draw order: threads/link(s) FIRST (behind the lobes in the map and scatter),
# the rest in the same relative order as `order`. The legend still uses `order`
# (threads last). Stable sort keeps the non-link relative order.
draw_order = sorted(order, key=lambda i: "link" not in region_names[i].lower())

c_ref = np.asarray(samples_mf.mean(sky_mf.objects[0].ref_freq_model) * CONV_FACTOR)
c_alpha = np.asarray(samples_mf.mean(sky_mf.objects[0].spectral_index))
mask = (c_ref > 1e-2) & np.isfinite(c_alpha)

region_id = assign_regions_to_pixels(mask, regions_config, brightness=c_ref)

# Region-map RGB (white background, grey for in-mask-but-unassigned pixels).
rgb_image = np.ones(c_ref.shape + (3,))
for rid in draw_order:
    rgb_image[region_id == rid] = mcolors.to_rgb(region_colors[rid])
rgb_image[mask & (region_id == -1)] = mcolors.to_rgb("grey")

# Per-pixel scatter data and the region-map bounding box.
x = c_ref[mask].ravel()
y = c_alpha[mask].ravel()
r = region_id[mask].ravel()

# Bounding box of the coloured regions (with a 5% margin) for the region map.
ii, jj = np.where(region_id >= 0)
pad_x = 0.05 * np.ptp(ii) if ii.size else 1.0
pad_y = 0.05 * np.ptp(jj) if ii.size else 1.0
x0, x1 = ii.min() - pad_x, ii.max() + pad_x
y0, y1 = jj.min() - pad_y, jj.max() + pad_y
map_aspect = (y1 - y0) / (x1 - x0)  # displayed height / width of the region map

# Legend handles in drawing order, with display names (link(s) -> "threads").
def region_label(name):
    n = name.lower()
    if "right" in n:
        return "right lobe"
    if "left" in n:
        return "left lobe"
    if "link" in n:
        return "threads"
    return name


legend_handles = [
    Patch(facecolor=region_colors[rid], label=region_label(region_names[rid]))
    for rid in order
]


def plot_regions(name, xlabel, scatter):
    """Region map on top (full width) and a colour-by-region scatter below."""
    fig_w, sc_aspect = 8.0, 0.7
    fig, (ax_map, ax_sc) = plt.subplots(
        2, 1, dpi=200,
        figsize=(fig_w, fig_w * (map_aspect + sc_aspect)),
        gridspec_kw={"height_ratios": [map_aspect, sc_aspect], "hspace": 0.08 / 3},
    )

    # region map on top, same width as the scatter, zoomed to the regions.
    ax_map.imshow(rgb_image.transpose(1, 0, 2), origin="lower", aspect="auto")
    ax_map.contour(
        c_ref.T, levels=[1e-2, 1e-1, 1, 10], colors="black", linewidths=0.4, origin="lower"
    )
    ax_map.set_xlim(x0, x1)
    ax_map.set_ylim(y0, y1)
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    # colour-by-region scatter below, with the region legend.
    scatter(ax_sc)
    ax_sc.set_xscale("log")
    ax_sc.set_xlabel(xlabel)
    ax_sc.set_ylabel(r"spectral index $\alpha$")
    ax_sc.grid(alpha=0.2)
    ax_sc.legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.savefig(os.path.join(odir, name), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def scatter_regions(ax):
    for rid in draw_order:
        sel = r == rid
        if np.any(sel):
            ax.scatter(
                x[sel], y[sel], s=1, alpha=0.5,
                color=region_colors[rid], edgecolors="none",
            )


plot_regions(
    "c1_regions.png",
    r"sky brightness [mJy / arcsec$^2$]", scatter_regions,
)

# %%
# ---------------------------------------------------------------------------
# Plot 4 (c2, ESO137-007): alpha vs flux coloured by distance ALONG the
# head-tail galaxy link. The link geometry is reproduced exactly from
# eso_c2_profiles.py (same REL_FOV/CENTER crop, LINK_ANCHORS, ridge trace, and
# the cross at the LEFT start anchor). Every c2 pixel above the flux floor is
# projected onto the link polyline; its arc-length position from the cross
# (converted to the chosen distance unit) sets the colour.
# ---------------------------------------------------------------------------
# --- link geometry copied from eso_c2_profiles.py (self-contained) ----------
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


def ridge_anchors(flux, x_start_frac, x_end_frac, *, method="max", smooth=0.0):
    """Trace the link ridge as (x, y) fractional anchors, one per x-pixel column."""
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
            raise ValueError("method must be 'max' or 'median'")

    if smooth and smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        ys = gaussian_filter1d(ys, float(smooth), mode="nearest")

    return np.column_stack([xs / (nx - 1), ys / (ny - 1)])


def link_arclength_of_pixels(P, verts):
    """Along-link arc length and perpendicular distance of each pixel (pixels).

    `P` (M, 2) and `verts` (V, 2) are in the same (axis0, axis1) pixel frame.
    Each pixel is projected onto every segment (clamped to the segment); the
    nearest foot gives its along-link arc length (from `verts[0]`) and the
    distance to that foot is the perpendicular distance from the link.
    Returns (arc_length, perp_distance), both (M,).
    """
    A, B = verts[:-1], verts[1:]
    AB = B - A
    seg_len2 = (AB ** 2).sum(1)
    seg_len = np.sqrt(seg_len2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    best_d2 = np.full(len(P), np.inf)
    best_s = np.zeros(len(P))
    for k in range(len(A)):
        t = np.clip(((P - A[k]) @ AB[k]) / max(seg_len2[k], 1e-12), 0.0, 1.0)
        foot = A[k] + t[:, None] * AB[k]
        d2 = ((P - foot) ** 2).sum(1)
        upd = d2 < best_d2
        best_d2[upd] = d2[upd]
        best_s[upd] = cum[k] + t[upd] * seg_len[k]
    return best_s, np.sqrt(best_d2)


# --- c2 link definition (identical to eso_c2_profiles.py) -------------------
C2_REL_FOV = (0.25, 0.09)
C2_CENTER = ("0.18deg", "0.31deg")
C2_LINK_ANCHORS = [(0.048, 0.12), (0.21, 0.28), (0.41, 0.58), (0.55, 0.65), (0.86, 0.7)]
C2_RIDGE_METHOD = "max"
C2_RIDGE_SMOOTH = 15.0
C2_FLUX_MIN = 5e-3  # same floor as the c2 scatter above
C2_FLUX_MAX = 5.0  # mJy/arcsec^2 ceiling: drop brighter (head) pixels
C2_HALF_WIDTH_ARCSEC = 80.0  # only keep pixels within this perp. distance of the link
C2_DIST_UNIT = "arcmin"  # "arcsec" or "arcmin"
_C2_DIST_SCALE = {"arcsec": 1.0, "arcmin": 1.0 / 60.0}[C2_DIST_UNIT]
_C2_DIST_LABEL = {"arcsec": '["]', "arcmin": "[']"}[C2_DIST_UNIT]

obj2 = sky_mf.objects[1]


def to_grid2(val):
    return aim.map_signal(obj2.grid, sky_mf.grid)(val)


# Cropped ref-freq brightness + spectral index in the link frame (posterior mean).
c2_ref = crop_component(to_grid2(samples_mf.mean(obj2.ref_freq_model)) * CONV_FACTOR, C2_REL_FOV, C2_CENTER)
c2_alpha = crop_component(to_grid2(samples_mf.mean(obj2.spectral_index)), C2_REL_FOV, C2_CENTER)
nx2, ny2 = c2_ref.shape
pix2_arcsec = 2.0 * 3600.0 * C2_REL_FOV[0] / nx2

# Traced link (same ridge as the profiles), as polyline vertices in pixel coords.
c2_link = ridge_anchors(
    c2_ref, C2_LINK_ANCHORS[0][0], C2_LINK_ANCHORS[-1][0],
    method=C2_RIDGE_METHOD, smooth=C2_RIDGE_SMOOTH,
)
c2_verts = c2_link * np.array([nx2 - 1, ny2 - 1])  # (V, 2) = (axis0=x, axis1=y)

# Per-pixel scatter + along-link distance from the cross (start anchor, s = 0),
# keeping only pixels within C2_HALF_WIDTH_ARCSEC (perpendicular) of the link.
c2_mask = (c2_ref > C2_FLUX_MIN) & (c2_ref < C2_FLUX_MAX) & np.isfinite(c2_alpha)
px, py = np.where(c2_mask)  # axis0 (x), axis1 (y) — matches the verts frame
P = np.column_stack([px, py]).astype("float64")
s_pix, perp_pix = link_arclength_of_pixels(P, c2_verts)

near = perp_pix <= (C2_HALF_WIDTH_ARCSEC / pix2_arcsec)  # within the link corridor
c2_dist = (s_pix * pix2_arcsec * _C2_DIST_SCALE)[near]  # distance along link, from cross
c2_x = c2_ref[c2_mask].ravel()[near]      # brightness
c2_y = c2_alpha[c2_mask].ravel()[near]    # spectral index

fig, ax = plt.subplots(figsize=(7.4, 5), dpi=200)
sc = ax.scatter(c2_x, c2_y, c=c2_dist, s=1, alpha=0.75, cmap="viridis", edgecolors="none")
ax.set_xscale("log")
ax.set_xlabel(r"sky brightness [mJy / arcsec$^2$]")
ax.set_ylabel(r"spectral index $\alpha$")
ax.grid(alpha=0.2)
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(f"distance along ESO137-007 {_C2_DIST_LABEL}")
fig.savefig(os.path.join(odir, "c2_distance.png"), bbox_inches="tight")
plt.close(fig)
print("saved c2_distance.png")

# %%

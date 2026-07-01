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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import aim_resolve as aim

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
# (c1 = ESO137-006, c2 = ESO137-007). Two versions per component:
#   unbinned : every pixel above the flux floor as a single-color scatter,
#   binned   : spectral index binned, summed brightness per bin.
# Adapted from `eso_mf_zoom_talk.py` (single color, no per-region coloring).
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

    # --- unbinned ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.scatter(x, y, s=2, alpha=0.35, color="k", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel(r"sky brightness [mJy / arcsec$^2$]")
    ax.set_ylabel("spectral index")
    ax.set_title(f"{lab} — unbinned")
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(odir, f"c{idx}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved c{idx}.png")

    # --- binned by spectral index, summed brightness per bin ---
    n_bins = 100
    binned_brightness, bin_edges = np.histogram(y, bins=n_bins, weights=x)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = binned_brightness > 0

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.scatter(
        binned_brightness[valid], bin_centers[valid],
        s=20, alpha=0.6, color="k", edgecolors="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"summed sky brightness [mJy / arcsec$^2$]")
    ax.set_ylabel("spectral index")
    ax.set_title(f"{lab} — binned by spectral index")
    ax.grid(alpha=0.2)
    fig.savefig(os.path.join(odir, f"c{idx}_binned.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved c{idx}_binned.png")

# %%
# ---------------------------------------------------------------------------
# Region analysis of c1 (ESO137-006). Regions are read from regions.yml
# (any number of entries — e.g. "left lobe", "link", "right lobe"); each region
# gets its own colour (left -> coolwarm blue, right -> red, link(s) -> orange).
# Produced: spectral index vs sky brightness coloured by region (unbinned and
# binned), each with a zoomed region-map inset. Adapted from eso_mf_physics.py.
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

c_ref = np.asarray(samples_mf.mean(sky_mf.objects[0].ref_freq_model) * CONV_FACTOR)
c_alpha = np.asarray(samples_mf.mean(sky_mf.objects[0].spectral_index))
mask = (c_ref > 1e-2) & np.isfinite(c_alpha)

region_id = assign_regions_to_pixels(mask, regions_config, brightness=c_ref)

# Region-map RGB (white background, grey for in-mask-but-unassigned pixels).
rgb_image = np.ones(c_ref.shape + (3,))
for rid in order:
    rgb_image[region_id == rid] = mcolors.to_rgb(region_colors[rid])
rgb_image[mask & (region_id == -1)] = mcolors.to_rgb("grey")

# Per-pixel scatter data and the region-map bounding box (shared by both plots).
x = c_ref[mask].ravel()
y = c_alpha[mask].ravel()
r = region_id[mask].ravel()

n_bins = 100
edges = np.linspace(np.nanmin(y), np.nanmax(y), n_bins + 1)
centers_bin = 0.5 * (edges[:-1] + edges[1:])

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
    ax_sc.set_ylabel("spectral index")
    ax_sc.grid(alpha=0.2)
    ax_sc.legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.savefig(os.path.join(odir, name), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def scatter_unbinned(ax):
    for rid in order:
        sel = r == rid
        if np.any(sel):
            ax.scatter(
                x[sel], y[sel], s=1, alpha=0.5,
                color=region_colors[rid], edgecolors="none",
            )


def scatter_binned(ax):
    for rid in order:
        sel = r == rid
        if not np.any(sel):
            continue
        binned, _ = np.histogram(y[sel], bins=edges, weights=x[sel])
        valid = binned > 0
        ax.scatter(
            binned[valid], centers_bin[valid], s=20, alpha=0.7,
            color=region_colors[rid], edgecolors="none",
        )


plot_regions(
    "c1_regions.png",
    r"sky brightness [mJy / arcsec$^2$]", scatter_unbinned,
)
plot_regions(
    "c1_regions_binned.png",
    r"summed sky brightness [mJy / arcsec$^2$]", scatter_binned,
)

# %%

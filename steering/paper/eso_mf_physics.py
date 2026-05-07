"""Multi-frequency physical analysis plots for ESO reconstructions."""

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %%
import pickle

import jax
import numpy as np

import aim_resolve as aim

jax.config.update("jax_enable_x64", True)


# %%
def box_markers(cfg, ps_map, grid, it):
    """Create marker dictionaries for point sources and object bounding boxes.

    Parameters
    ----------
    cfg : SetupKLConfig
        Configuration object containing sky model sections.
    ps_map : np.ndarray
        Point source detection map.
    grid : SignalGrid
        Signal grid for coordinate mapping.
    it : int
        Current iteration number.

    Returns
    -------
    dict
        Dictionary with 'ps_mrk' and 'oj_mrk' marker dictionaries.
    """
    import numpy as np

    from aim_resolve import draw_boxes

    px, py = np.argwhere(ps_map > 0).T
    ps_mrk = dict(x=px, y=py, s=10, c="white", marker="+")
    box_map = draw_boxes(cfg.sections, grid, it)
    ox, oy = np.argwhere(box_map > 0).T
    oj_mrk = dict(x=ox, y=oy, s=0.1, c="white", marker=",")

    return dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)


# %%
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

mf_rec = "3_rec_2z_6f"
mf_it = 5

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
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
min_mf = 2.5e-3
max_mf = np.max(samples_mf.mean(sky_mf.ref_freq_model) * CONV_FACTOR)
print("vmin:", min_mf, "\nvmax:", max_mf)

plot_dict = dict(
    name=None,
    odir=None,
    norm="log",
    vmin=min_mf,
    vmax=max_mf,
    cmap="inferno",
    cbar=True,
    ticks=0,
    dpi=300,
)

# %%
comps_mf = [sky_mf.objects[0], sky_mf.objects[1]]

comp_ref_mf = [samples_mf.mean(c.ref_freq_model) * CONV_FACTOR for c in comps_mf]
comp_alpha = [samples_mf.mean(c.spectral_index) for c in comps_mf]
min_cs, min_ca = 5e-3, 1e-2


print("plotting ...")
for c, a in zip(comp_ref_mf, comp_alpha):
    a = np.where(c > min_ca, a, np.nan)
    contours = {
        "array": c,
        "colors": "white",
        "levels": [1e-2, 1e-1],
        "linewidths": 0.5,
    }
    aim.plot_arrays(
        array=c,
        **plot_dict | dict(vmin=min_cs),
    )
    aim.plot_arrays(
        array=a,
        contour=contours,
        **plot_dict | dict(vmin=-3, vmax=0, norm="linear"),
    )

# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def assign_regions_to_pixels(mask, regions_config, brightness=None):
    """Assign region IDs to pixels based on multiple shapes per region and optional value ranges"""
    region_id = np.full(mask.shape, -1, dtype=int)
    colors = np.full((len(regions_config), 3), np.nan)

    # Convert brightness to numpy array if provided
    if brightness is not None:
        brightness = np.asarray(brightness)

    # Get pixel coordinates where mask is True
    y_coords, x_coords = np.where(mask)

    # Assign region ID to each pixel
    for idx, name in enumerate(regions_config):
        region = regions_config[name]
        region_mask = np.zeros(len(y_coords), dtype=bool)

        for shape_idx, (cen, ext) in enumerate(zip(region["center"], region["extend"])):
            c_y, c_x = cen

            if len(ext) == 1:  # Circle
                diameter = ext[0]
                radius = diameter / 2
                dist = np.sqrt((x_coords - c_x) ** 2 + (y_coords - c_y) ** 2)
                geom_mask = dist <= radius
            elif len(ext) == 2:  # Rectangle
                f_y, f_x = ext
                xmin = c_x - f_x / 2
                xmax = c_x + f_x / 2
                ymin = c_y - f_y / 2
                ymax = c_y + f_y / 2
                geom_mask = (
                    (x_coords >= xmin)
                    & (x_coords < xmax)
                    & (y_coords >= ymin)
                    & (y_coords < ymax)
                )
            else:
                raise ValueError("Unknown shape type")

            # Value range check if specified
            if "value" in region and brightness is not None:
                if shape_idx < len(region["value"]):
                    v_min, v_max = region["value"][shape_idx]
                    v_min, v_max = float(v_min), float(v_max)
                    brightness_values = brightness[y_coords, x_coords]
                    value_mask = (brightness_values >= v_min) & (
                        brightness_values <= v_max
                    )
                    geom_mask = geom_mask & value_mask

            region_mask |= geom_mask

        # Convert color name to RGB (using matplotlib)
        rgb_color = mcolors.to_rgb(region["color"])
        colors[idx] = rgb_color

        # Assign region ID to matched pixels
        region_id[y_coords[region_mask], x_coords[region_mask]] = idx

    return region_id, colors


regions_config = aim.yaml_load(
    "/scratch/users/rfuchs/packages/aim-resolve/steering/paper/regions.yml"
)
min_scatter = 5e-3


for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf[:1], comp_alpha[:1]), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)

    # Assign regions to pixels
    region_id, colors = assign_regions_to_pixels(mask, regions_config, brightness=c_ref)

    height, width = c_ref.shape

    # Create RGB image - initialize to white for low brightness points
    rgb_image = np.ones((height, width, 3))

    for i, color in enumerate(colors):
        region_mask = (region_id == i) & mask
        rgb_image[region_mask] = color

    # Points not in any region but above threshold (region_id == -1 and in mask) get grey
    unassigned_mask = (region_id == -1) & mask
    rgb_image[unassigned_mask] = mcolors.to_rgb("grey")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_image.transpose(1, 0, 2), origin="lower")

    # Add contours from component brightness
    ax.contour(
        c_ref.T, colors="white", levels=[1e-2, 1e-1], linewidths=0.5, origin="lower"
    )

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_title(f"Component {idx} (flexible regions)")

    # Add legend for region colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[i], label=name)
        for i, name in enumerate(regions_config.keys())
    ]
    legend_elements.append(Patch(facecolor=mcolors.to_rgb("grey"), label="unassigned"))
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=7,
        ncol=3,
        bbox_to_anchor=(0.5, -0.15),
        frameon=True,
    )

    plt.tight_layout()

# %%
# Scatter plot colored by flexible regions (Component 1 only)

print("plotting scatter with flexible regions (log scale) ...")

for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf[:1], comp_alpha[:1]), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)

    # Assign regions to pixels
    region_id, region_colors = assign_regions_to_pixels(
        mask, regions_config, brightness=c_ref
    )

    x = c_ref[mask].ravel()
    y = c_alpha[mask].ravel()
    regions = region_id[mask].ravel()

    plt.figure(figsize=(10, 6))
    for region_idx, (name, region) in enumerate(regions_config.items()):
        region_mask = regions == region_idx
        if np.any(region_mask):
            plt.scatter(
                x[region_mask],
                y[region_mask],
                s=2,
                alpha=0.5,
                edgecolors="none",
                color=region["color"],
                label=name,
            )

    # Plot unassigned points in grey
    unassigned_mask = regions == -1
    if np.any(unassigned_mask):
        plt.scatter(
            x[unassigned_mask],
            y[unassigned_mask],
            s=2,
            alpha=0.5,
            edgecolors="none",
            color="grey",
            label="unassigned",
        )

    plt.xscale("log")
    plt.xlabel("Sky brightness [mJy/arcsec$^2$]")
    plt.ylabel("Spectral index")
    plt.title(f"Component {idx} (flexible regions, log scale)")
    plt.legend(loc="lower center", fontsize=7, ncol=3, bbox_to_anchor=(0.5, -0.15))
    plt.grid(alpha=0.2)

# %%

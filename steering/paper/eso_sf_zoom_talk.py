"""Single-frequency zoom-in talk plots for ESO reconstructions."""

# %%
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# %%
import pickle

import jax
import aim_resolve as aim
import numpy as np

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
    px, py = np.argwhere(ps_map == 1).T
    ps_mrk = dict(x=px, y=py, s=10, c="white", marker="+")
    box_map = aim.draw_boxes(cfg.sections, grid, it)
    ox, oy = np.argwhere(box_map == 1).T
    oj_mrk = dict(x=ox, y=oy, s=0.1, c="white", marker=",")

    return dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)


# %%
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

sf_rec = "4_rec_3z_1"
sf_it = 5

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
CONV_FACTOR = 1000 * AS2RAD**2


opt_yml = f"{dir}/opt/{sf_rec}/opt.yml"
print("load:", opt_yml)

optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)
optim_cfg.sections["data.0"]["fname"] = "/Users/rf/Development/data/eso_986-1137mhz.npz"

sky_sf = optim_cfg.instantiate_sec(f"sky.{sf_it}")
print("sky components:", [c.prefix for c in sky_sf.models])

with open(f"{dir}/opt/{sf_rec}/last.pkl", "rb") as f:
    samples_sf, *_ = pickle.load(f)
samples_sf = aim.MySamples.from_samples(samples_sf)
print("samples:", len(samples_sf))

setup_cfg = aim.SetupKLConfig.from_file(opt_yml)
sky_ps = sky_sf.points[0]
ps_map = aim.map_signal(sky_ps.points.grid, sky_ps.grid)(np.ones(sky_ps.shape))
markers_sf = box_markers(setup_cfg, ps_map, sky_sf.grid, sf_it)
print("markers:", [f"{k}: {len(v['x'])}" for k, v in markers_sf.items()])

# %%
min_sf = 1e-4
max_sf = np.max(samples_sf.mean(sky_sf) * CONV_FACTOR)
print("vmin:", min_sf, "\nvmax:", max_sf)

plot_dict = dict(
    name=None,
    odir=None,
    norm="log",
    vmin=min_sf,
    vmax=max_sf,
    cmap="inferno",
    cbar=False,
    ticks=0,
    dpi=300,
)

# %%
sky_mean_sf = samples_sf.mean(sky_sf) * CONV_FACTOR

pot_mean_sf = samples_sf.mean(sky_sf.points_and_objects) * CONV_FACTOR


print("plotting ...")
aim.plot_arrays(
    array=sky_mean_sf,
    marker=markers_sf,
    callback=lambda fig, ax: fig.text(0.085, 0.90, "1062 MHz", fontsize=15, c="white"),
    **plot_dict,
)
aim.plot_arrays(
    array=sky_mean_sf,
    callback=lambda fig, ax: fig.text(0.085, 0.90, "1062 MHz", fontsize=15, c="white"),
    **plot_dict,
)
aim.plot_arrays(
    array=pot_mean_sf,
    callback=lambda fig, ax: fig.text(0.085, 0.90, "1062 MHz", fontsize=15, c="black"),
    **plot_dict,
)

# %%
comp_mean_sf = [
    samples_sf.mean(c) * CONV_FACTOR for c in [sky_sf.objects[0], sky_sf.objects[1]]
]


print("plotting ...")
for c in comp_mean_sf:
    aim.plot_arrays(
        array=c,
        figsize=(20,20),
        **plot_dict,
    )

# %%
sky_pt_sf = sky_sf.copy()

pt_models = []
for m in sky_pt_sf.models:
    if m not in [sky_sf.background, sky_sf.objects[0], sky_sf.objects[1]]:
        pt_models.append(m)

sky_pt_sf.models = pt_models
pt_mean_sf = samples_sf.mean(sky_pt_sf) * CONV_FACTOR


print("plotting ...")
aim.plot_arrays(
    array=pt_mean_sf,
    **plot_dict,
)

# %%
zoom = 2


smp_val_sf = []
for smp in samples_sf:
    val = sky_sf(smp) * CONV_FACTOR
    if zoom > 1:
        grd = sky_sf.grid
        val = aim.map_signal(grd, grd.update(space=grd.spc // 2))(val)
    smp_val_sf.append(val)


def callback(fig, axes):
    fig.text(0.045, 0.950, "Sample 1", fontsize=15, c="white")
    fig.text(0.51, 0.950, "Sample 2", fontsize=15, c="white")
    fig.text(0.045, 0.485, "Sample 3", fontsize=15, c="white")
    fig.text(0.51, 0.485, "Sample 4", fontsize=15, c="white")


print("plotting ...")
aim.plot_arrays(
    smp_val_sf,
    rows=2,
    grid_kwargs=dict(
        hspace=-0.71, wspace=-0.7, width_ratios=[1, 1], height_ratios=[1, 1]
    ),
    callback=callback,
    **plot_dict,
)

# %%
zoom = 2

mean, std = samples_sf.mean_and_std(sky_sf)
skyz_mean_sf = mean * CONV_FACTOR
skyz_runc_sf = std / mean
if zoom:
    grd = sky_sf.grid
    skyz_mean_sf = aim.map_signal(grd, grd.update(space=grd.spc // 2))(skyz_mean_sf)
    skyz_runc_sf = aim.map_signal(grd, grd.update(space=grd.spc // 2))(skyz_runc_sf)


print("plotting ...")
aim.plot_arrays(
    array=skyz_mean_sf,
    callback=lambda fig, ax: fig.text(
        0.085, 0.90, "Posterior mean", fontsize=15, c="white"
    ),
    **plot_dict,
)
aim.plot_arrays(
    array=skyz_runc_sf,
    contour={
        "array": skyz_mean_sf,
        "colors": "white",
        "levels": [1e-2, 1e-1],
        "linewidths": 0.5,
    },
    callback=lambda fig, ax: fig.text(
        0.085, 0.90, "Relative uncertainty", fontsize=15, c="black"
    ),
    **plot_dict | dict(vmin=None, vmax=None),
)

# %%
import nifty.re as jft

tiles_sf = sky_sf.tiles[0]

tiles_val = jft.mean(tuple((tiles_sf(s, map=False)) * CONV_FACTOR for s in samples_sf))

print(tiles_val.shape)

print("plotting ...")
tiles_arrays = []
tiles_vmin = []
tiles_vmax = []

for i,t in enumerate(tiles_val):
    t += 1e-9
    tiles_arrays.append(t)
    ts_max = np.max(t)
    ts_min = t.max() * 1e-3

    tiles_vmin.append(ts_min)
    tiles_vmax.append(ts_max)

aim.plot_arrays(
    array=tiles_arrays,
    cols=8,
    **plot_dict
    | dict(vmin=tiles_vmin, vmax=tiles_vmax, norm="log"),
)
# %%

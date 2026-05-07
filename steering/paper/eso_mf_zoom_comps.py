# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %%
import pickle

import numpy as np

import aim_resolve as aim


# %%
def box_markers(cfg, ps_map, grid, it):
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
    cbar=False,
    ticks=0,
    dpi=300,
)

# %%
sky_ref_mf = samples_mf.mean(sky_mf.ref_freq_model) * CONV_FACTOR

pot_ref_mf = samples_mf.mean(sky_mf.points_and_objects.ref_freq_model) * CONV_FACTOR

bg_ref_mf = samples_mf.mean(sky_mf.background.ref_freq_model) * CONV_FACTOR

pot_alpha = samples_mf.mean(sky_mf.points_and_objects.spectral_index)
pot_alpha = np.where(pot_ref_mf > 5e-3, pot_alpha, np.nan)
amin, amax = -3, 3

contours = {
    "array": sky_ref_mf,
    "colors": "white",
    "levels": [1e-2, 1e-1],
    "linewidths": 0.25,
}

print("plotting ...")
aim.plot_arrays(
    array=sky_ref_mf,
    marker=markers_mf,
    callback=lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c="white"),
    **plot_dict,
)
aim.plot_arrays(
    array=sky_ref_mf,
    callback=lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c="white"),
    **plot_dict,
)
aim.plot_arrays(
    array=bg_ref_mf,
    callback=lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c="black"),
    **plot_dict,
)
aim.plot_arrays(
    array=pot_ref_mf,
    callback=lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c="black"),
    **plot_dict,
)
aim.plot_arrays(
    array=pot_alpha,
    # callback = lambda fig, ax: fig.text(0.085, 0.90, 'Spectral Index', fontsize=15, c='black'),
    contour=contours,
    **plot_dict | dict(vmin=amin, vmax=amax, norm="linear", cmap="coolwarm"),
)

# %%
comps_mf = [sky_mf.objects[0], sky_mf.objects[1]]

comp_ref_mf = [samples_mf.mean(c.ref_freq_model) * CONV_FACTOR for c in comps_mf]
comp_alpha = [samples_mf.mean(c.spectral_index) for c in comps_mf]
min_cs, min_ca = 2.5e-3, 5e-3


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
        **plot_dict | dict(vmin=-3, vmax=3, norm="linear", cmap="coolwarm"),
    )

# %%
tiles = sky_mf.tiles[0]
tiles_rfm = tiles.ref_freq_model
tiles_sim = tiles.spectral_index
tiles_rfm.set_out_grid(tiles.tiles.grid)
tiles_sim.set_out_grid(tiles.tiles.grid)

tiles_ref_mf = samples_mf.mean(tiles_rfm) * CONV_FACTOR
tiles_alpha = samples_mf.mean(tiles_sim)
min_ts, min_ta = 1e-3, 2.5e-3

print(tiles_ref_mf.shape)
print(tiles_alpha.shape)

print("plotting ...")
tiles_arrays = []
tiles_vmin = []
tiles_vmax = []
tiles_norm = []
tiles_cmap = []
tiles_contour = []

for t, a in zip(tiles_ref_mf, tiles_alpha):
    ts_max = np.max(t)
    ts_min = t.max() * 1e-2
    ta_min = t.max() * 5e-2

    a = np.where(t > ta_min, a, np.nan)
    contours = {
        "array": t,
        "colors": "white",
        "levels": [1e-2, 1e-1],
        "linewidths": 0.5,
    }

    tiles_arrays.extend([t, a])
    tiles_vmin.extend([ts_min, -3])
    tiles_vmax.extend([ts_max, 3])
    tiles_norm.extend(["log", "linear"])
    tiles_cmap.extend(["inferno", "coolwarm"])
    tiles_contour.extend([{}, contours])

aim.plot_arrays(
    array=tiles_arrays,
    cols=16,
    contour=tiles_contour,
    **plot_dict
    | dict(vmin=tiles_vmin, vmax=tiles_vmax, norm=tiles_norm, cmap=tiles_cmap),
)

# %%

print("plotting sorted tiles ...")
tiles_peak = np.array([np.max(t) for t in tiles_ref_mf])
tiles_order = np.argsort(tiles_peak)[::-1]

sorted_tiles_ref = [tiles_ref_mf[i] for i in tiles_order]
sorted_tiles_alpha = [tiles_alpha[i] for i in tiles_order]
sorted_tiles_peak = tiles_peak[tiles_order]

sorted_tiles_arrays = []
sorted_tiles_vmin = []
sorted_tiles_vmax = []
sorted_tiles_norm = []
sorted_tiles_cmap = []
sorted_tiles_contour = []

for group_start in range(0, len(sorted_tiles_ref), 8):
    group_ref = sorted_tiles_ref[group_start : group_start + 8]
    group_alpha = sorted_tiles_alpha[group_start : group_start + 8]
    group_vmax = sorted_tiles_peak[group_start]
    group_vmin = group_vmax * 1e-2

    for t, a in zip(group_ref, group_alpha):
        ta_min = t.max() * 5e-2
        a = np.where(t > ta_min, a, np.nan)
        contours = {
            "array": t,
            "colors": "white",
            "levels": [1e-2, 1e-1],
            "linewidths": 0.5,
        }

        sorted_tiles_arrays.extend([t, a])
        sorted_tiles_vmin.extend([group_vmin, -3])
        sorted_tiles_vmax.extend([group_vmax, 3])
        sorted_tiles_norm.extend(["log", "linear"])
        sorted_tiles_cmap.extend(["inferno", "coolwarm"])
        sorted_tiles_contour.extend([{}, contours])

aim.plot_arrays(
    array=sorted_tiles_arrays,
    cols=16,
    contour=sorted_tiles_contour,
    **plot_dict
    | dict(
        vmin=sorted_tiles_vmin,
        vmax=sorted_tiles_vmax,
        norm=sorted_tiles_norm,
        cmap=sorted_tiles_cmap,
    ),
)

# %%
# plot_dict.pop('cbar')


print("plotting transposed sorted tiles ...")
sorted_groups_ref = [
    sorted_tiles_ref[i : i + 8] for i in range(0, len(sorted_tiles_ref), 8)
]
sorted_groups_alpha = [
    sorted_tiles_alpha[i : i + 8] for i in range(0, len(sorted_tiles_alpha), 8)
]
sorted_groups_peak = [sorted_tiles_peak[i] for i in range(0, len(sorted_tiles_peak), 8)]

transposed_tiles_arrays = []
transposed_tiles_vmin = []
transposed_tiles_vmax = []
transposed_tiles_norm = []
transposed_tiles_cmap = []
transposed_tiles_contour = []
transposed_tiles_cbar = []
transposed_tiles_cbar_kwargs = []

for row_idx in range(8):
    for group_idx, (group_ref, group_alpha, group_peak) in enumerate(
        zip(sorted_groups_ref, sorted_groups_alpha, sorted_groups_peak)
    ):
        t = group_ref[row_idx]
        a = group_alpha[row_idx]

        ta_min = t.max() * 0.05
        a = np.where(t > ta_min, a, np.nan)
        contours = {
            "array": t,
            "colors": "white",
            "levels": [1e-2, 1e-1],
            "linewidths": 0.5,
        }

        transposed_tiles_arrays.extend([t, a])
        transposed_tiles_vmin.extend([group_peak * 0.01, -3])
        transposed_tiles_vmax.extend([group_peak, 3])
        transposed_tiles_norm.extend(["log", "linear"])
        transposed_tiles_cmap.extend(["inferno", "coolwarm"])
        transposed_tiles_contour.extend([{}, contours])

        show_cbar = row_idx == 7
        transposed_tiles_cbar.extend([show_cbar, show_cbar])
        transposed_tiles_cbar_kwargs.extend(
            [
                {"loc": "bottom"} if show_cbar else {},
                {"loc": "bottom"} if show_cbar else {},
            ]
        )

aim.plot_arrays(
    array=transposed_tiles_arrays,
    cols=16,
    contour=transposed_tiles_contour,
    cbar=transposed_tiles_cbar,
    cbar_kwargs=transposed_tiles_cbar_kwargs,
    **plot_dict
    | dict(
        vmin=transposed_tiles_vmin,
        vmax=transposed_tiles_vmax,
        norm=transposed_tiles_norm,
        cmap=transposed_tiles_cmap,
    ),
)
# aim.plot_arrays(
#     array = tiles_arrays[::2],
#     cols = 8,
#     contour = tiles_contour[::2],
#     **plot_dict | dict(vmin=tiles_vmin[::2], vmax=tiles_vmax[::2], norm=tiles_norm[::2], cmap=tiles_cmap[::2]),
# )
# aim.plot_arrays(
#     array = tiles_arrays[1:][::2],
#     cols = 8,
#     contour = tiles_contour[1:][::2],
#     **plot_dict | dict(vmin=tiles_vmin[1:][::2], vmax=tiles_vmax[1:][::2], norm=tiles_norm[1:][::2], cmap=tiles_cmap[1:][::2]),
# )

# %%

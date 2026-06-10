# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %%
import pickle

import numpy as np

import aim_resolve as aim

from os.path import splitext


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


def array2fits(array, grid, file_name, overwrite, direction=None):
    import astropy.io.fits as pyfits
    from astropy.time import Time

    dom = grid
    if direction is not None:
        pcx, pcy = direction.phase_center
    h = pyfits.Header()
    h["BUNIT"] = "Jy/sr"
    h["CTYPE1"] = "RA---SIN"
    h["CRVAL1"] = pcx * 180 / np.pi if direction is not None else 0.0
    h["CDELT1"] = -dom.distances[0] * 180 / np.pi
    h["CRPIX1"] = dom.shape[0] / 2
    h["CUNIT1"] = "deg"
    h["CTYPE2"] = "DEC---SIN"
    h["CRVAL2"] = pcy * 180 / np.pi if direction is not None else 0.0
    h["CDELT2"] = dom.distances[1] * 180 / np.pi
    h["CRPIX2"] = dom.shape[1] / 2
    h["CUNIT2"] = "deg"
    # h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]
    if direction is not None:
        h["EQUINOX"] = direction.equinox
    hdu = pyfits.PrimaryHDU(array.T, header=h)
    hdulist = pyfits.HDUList([hdu])
    base, ext = splitext(file_name)
    hdulist.writeto(base + ext, overwrite=overwrite)


# %%
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

mf_rec = "4_rec_3z_1_6f_2"
mf_it = 6

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
min_mf = 1e-3
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
zoom = 2

pot_ref_mean = samples_mf.mean(sky_mf.ref_freq_model)
pot_alpha_mean = samples_mf.mean(sky_mf.spectral_index)

grd = sky_mf.grid
pot_ref_mz = aim.map_signal(grd, grd.update(space=grd.spc // 2))(pot_ref_mean) * CONV_FACTOR
pot_alpha_mz = aim.map_signal(grd, grd.update(space=grd.spc // 2))(pot_alpha_mean)

min_cs, min_ca = 1e-3, 5e-3
pot_alpha_m0 = np.where(pot_ref_mz > 5e-3, pot_alpha_mz, np.nan) 
contours = {
    "array": pot_ref_mz,
    "colors": "white",
    "levels": [1e-2, 1e-1],
    "linewidths": 0.5,
}

print("plotting ...")
aim.plot_arrays(
    array=[pot_ref_mz, pot_alpha_m0],
    contour=[{}, contours.copy()],
    rows=1,
    figsize=(10,10),
    **plot_dict | dict(
        vmin=[min_cs, -4], 
        vmax=[max_mf, 0], 
        norm=["log", "linear"], 
        cmap=["inferno", "coolwarm"], 
        cbar=True, 
        cbar_kwargs={"loc": "bottom"}),
        grid_kwargs=dict(hspace=0, wspace=-0.2, width_ratios=[1, 1]),
)
# aim.plot_arrays(
#     array=pot_ref_mz,
#     figsize=(10,10),
#     **plot_dict | dict(vmin=min_cs, vmax=max_mf, cbar=True, cbar_kwargs={"loc": "bottom"}),
# )
# aim.plot_arrays(
#     array=pot_alpha_m0,
#     contour=contours.copy(),
#     figsize=(10,10),
#     **plot_dict | dict(vmin=-4, vmax=0, norm="linear", cmap="coolwarm", cbar=True, cbar_kwargs={"loc": "bottom"}),
# )

# %%
comps_mf = [sky_mf.objects[0], sky_mf.objects[1]]

min_cs, min_ca = 1e-3, 5e-3

# grid_kwargs=[dict(hspace=-0.62, wspace=0, width_ratios=[1, 1]), dict(hspace=-0.9, wspace=0, width_ratios=[1, 1])]
# grid_kwargs=[dict(hspace=-0.45, wspace=0, width_ratios=[1, 1]), dict(hspace=-0.8, wspace=0, width_ratios=[1, 1])]
grid_kwargs=[dict(hspace=-0.6, wspace=-0.3, width_ratios=[1, 1]), dict(hspace=-0.9, wspace=-0.3, width_ratios=[1, 1])]


for i, c in enumerate(comps_mf):
    c_m, c_s = samples_mf.mean_and_std(c.ref_freq_model)
    a_m, a_s = samples_mf.mean_and_std(c.spectral_index)
    print("comp shape:", c_m.shape)

    c_m0 = c_m * CONV_FACTOR
    c_s0 = c_s * CONV_FACTOR
    a_m0 = np.where(c_m0 > min_ca, a_m, np.nan)
    a_s0 = np.where(c_m0 > min_ca, a_s, np.nan)
    contours = {
        "array": c_m0,
        "colors": "white",
        "levels": [1e-2, 1e-1],
        "linewidths": 0.5,
    }
    print(f"plotting comp {i}...")
    # aim.plot_arrays(
    #     array=[c_m0, a_m0, c_s0 / c_m0, a_s0 / np.abs(a_m0)],
    #     contour=[{}, contours.copy(), contours.copy(), contours.copy()],
    #     rows=2,
    #     figsize=(10,10),
    #     **plot_dict | dict(
    #         vmin=[min_cs, -4, None, None],
    #         vmax=[max_mf, 0, None, 0.2],
    #         norm=["log", "linear", "log", "linear"],
    #         cmap=["inferno", "coolwarm", "inferno", "coolwarm"],
    #         cbar=True,
    #         # cbar_kwargs=2*[{"loc": "left"}, {"loc": "right"}],
    #         cbar_kwargs={"loc": "bottom"},
    #         grid_kwargs=grid_kwargs[i],
    #     ),
    # )
    aim.plot_arrays(
        array=[c_m0, c_s0 / c_m0, a_m0, a_s0 / np.abs(a_m0)],
        contour=[{}, contours.copy(), contours.copy(), contours.copy()],
        rows=2,
        figsize=(10,10),
        **plot_dict | dict(
            vmin=[min_cs, None, -4, None],
            vmax=[max_mf, None, 0, 0.2],
            norm=2*["log"] + 2*["linear"],
            cmap=2*["inferno"] + 2*["coolwarm"],
            cbar=True,
            cbar_kwargs={"loc": "bottom"},
            grid_kwargs=grid_kwargs[i],
        ),
    )
    # aim.plot_arrays(
    #     array=c_m,
    #     figsize=(10,10),
    #     **plot_dict | dict(vmin=min_cs, vmax=max_mf, cbar=True, cbar_kwargs={"loc": "bottom"}),
    # )
    # aim.plot_arrays(
    #     array=a_m0,
    #     contour=contours.copy(),
    #     figsize=(10,10),
    #     **plot_dict | dict(vmin=-4, vmax=0, norm="linear", cmap="coolwarm", cbar=True, cbar_kwargs={"loc": "bottom"}),
    # )
    # aim.plot_arrays(
    #     array=c_s / c_m,
    #     contour=contours.copy(),
    #     figsize=(10,10),
    #     **plot_dict | dict(vmin=None, vmax=None, norm="log", cbar=True, cbar_kwargs={"loc": "bottom"}),
    # )
    # aim.plot_arrays(
    #     array=a_s0 / np.abs(a_m0),
    #     contour=contours.copy(),
    #     figsize=(10,10),
    #     **plot_dict | dict(vmin=None, vmax=0.2, norm="linear", cmap="coolwarm", cbar=True, cbar_kwargs={"loc": "bottom"}),
    # )


# %%
import nifty.re as jft

tiles = sky_mf.tiles[0]
tiles_rfm = tiles.ref_freq_model
tiles_sim = tiles.spectral_index

print(tiles_rfm)

tiles_ref_mf = jft.mean(tuple((tiles_rfm(s, map=False)) * CONV_FACTOR for s in samples_mf))
tiles_alpha = jft.mean(tuple(tiles_sim(s, map=False) for s in samples_mf))
# samples_mf.mean(tiles_rfm) * CONV_FACTOR
# tiles_alpha = samples_mf.mean(tiles_sim)
min_ts, min_ta = 1e-3, 5e-3

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
    t += 1e-10
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
        t += 1e-10
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

        ta_min = t.max() * 0.01
        a = np.where(t > ta_min, a, np.nan)
        contours = {
            "array": t,
            "colors": "white",
            "levels": [1e-2, 1e-1],
            "linewidths": 0.5,
        }
        t += 1e-10
        transposed_tiles_arrays.extend([t, a])
        transposed_tiles_vmin.extend([group_peak * 1e-4, -3])
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
    **plot_dict
    | dict(
        vmin=transposed_tiles_vmin,
        vmax=transposed_tiles_vmax,
        norm=transposed_tiles_norm,
        cmap=transposed_tiles_cmap,
        cbar=transposed_tiles_cbar,
        cbar_kwargs=transposed_tiles_cbar_kwargs,
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

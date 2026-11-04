# %%
# ---------------------------------------------------------------------------
# Truth sky with components from a single-frequency zoom run (alternative to
# make_truth_comp.py, same output): the decomposition of the run's last stage
# (posterior mean of every component), cut to the 0.5 deg field.
#   O0, O2   object components
#   tiles    the tiles of t0 whose 96 x 96 fine-pixel footprint lies inside
#   points   the point sources of p0 whose 3 x 3 sub-pixel block lies inside (the run's sub-pixel fluxes)
#   BG       the rest of the run's sky in the field: its background plus the parts of components that are
#            not modelled here (tiles / points partly or not inside)
# Objects O1 (ESO 137-007) and O3 lie largely outside the field and are left out entirely (only their edges
# reach into it: a bright source cut off at the field edge). The total is the run's sky without O1 / O3.
# The run's sky is scaled by FLUX_SCALE = 2: the single-frequency data are on half the flux scale of the
# multi-frequency data (V_SF = 0.5 V_MF), and the synthetic noise is that of the real data. The run's sky is
# apparent (no primary beam in the single-frequency stages) and taken as the intrinsic truth (~10 % too faint
# at the edge of the field; irrelevant for the test). Spectra as in make_truth_comp.py: the diffuse
# components follow the alpha / beta fields of make_truth.py, every point source has its own alpha / beta.
#
# Output (as make_truth_comp.py): OUT_COMP/truth/<variant>_comp/ truth.npz, components.yml, FITS, overview.png
#
# usage: python make_truth_sfrun.py [RUN_DIR]   (default: REF_DIR/opt/4_rec_3z_3)
# ---------------------------------------------------------------------------
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import astropy.io.fits as pyfits
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import (
    ALPHA_MEAN, ALPHA_STD, BETA_MEAN, BETA_STD, CROP, FAC, N, OUT, PREFIX, REF_DIR, REF_MF, REF_N, SEED_TRUTH,
    STEERING, truth_dir,
)

import aim_resolve as aim
from aim_resolve.model.map import map_signal

RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else f"{REF_DIR}/opt/4_rec_3z_3"
FLUX_SCALE = 2.0
AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2 (save_fits.py)
PIX_AS = 7.03125 / 3  # fine pixel [arcsec]
OMEGA_PIX = (PIX_AS * AS2RAD) ** 2  # fine-pixel solid angle [sr]
EDGE = 3  # fine-pixel margin to the field edge for points and tiles
LO, HI = CROP.start, CROP.stop  # fine-pixel range of the field in the 3072^2 grid
F = (slice(LO, HI), slice(LO, HI))


def coarse(i):
    """Fine-pixel index of the 3072^2 grid -> coarse-pixel coordinate relative to the centre."""
    return (i + 0.5) / FAC - REF_N / (2 * FAC)


# %%
# ---------------------------------------------------------------------------
# Posterior mean of every component of the run's last stage on the 3072^2 grid.
# ---------------------------------------------------------------------------
cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{RUN}/opt.yml"), aim.get_builders)
it = max(int(k.split(".")[1]) for k in cfg.sections if k.startswith("sky.") and k.split(".")[1].isdigit())
sky = cfg.instantiate_sec(f"sky.{it}")
with open(f"{RUN}/last.pkl", "rb") as f:
    samples, *_ = pickle.load(f)
print(f"run {RUN} | stage {it} | components {[m.prefix for m in sky.models]} | {len(samples)} samples")


def full(m):
    """Posterior mean of component m on the sky grid (objects are placed from their own grid)."""
    v = np.squeeze(np.mean([np.asarray(m(x)) for x in samples], 0)).astype("float64")
    return v if v.shape == sky.grid.shape else np.asarray(map_signal(m.grid, sky.grid)(v))


EXCLUDE = ("o1", "o3")  # objects largely outside the field
comp = {m.prefix.split(".")[0]: full(m) for m in sky.models}
total = np.squeeze(np.mean([np.asarray(sky(x)) for x in samples], 0)).astype("float64")
print("sum of components / total sky:", f"{sum(comp.values()).sum() / total.sum():.6f}")
for k in EXCLUDE:
    print(f"left out {k}: {comp[k][F].sum() * FLUX_SCALE * (7.03125 / 3 * np.pi / 180 / 3600) ** 2:.4f} Jy in the field")
    total = total - comp[k]

# %%
# ---------------------------------------------------------------------------
# Selection inside the field (geometry of the run = that of the reference run's FITS tables).
# ---------------------------------------------------------------------------
hdul = pyfits.open(f"{REF_MF}/fits/{PREFIX}_1054MHz_posterior_mean.fits")
run_cfg = {k: v for d in yaml.safe_load_all(open(f"{RUN}/opt.yml")) if d for k, v in d.items()}
tile_centers = run_cfg[f"sky_t0.{it}"]["tile_grid"]["center"]
tiles, tile_fp = [], np.zeros(sky.grid.shape, dtype=bool)
for row in hdul["TILEPOS"].data:
    t, x0, y0, nx, ny = int(row["TILE"]), int(row["X0_PIX"]) - 1, int(row["Y0_PIX"]) - 1, int(row["NX"]), int(row["NY"])
    if not (x0 >= LO + EDGE and y0 >= LO + EDGE and x0 + nx <= HI - EDGE and y0 + ny <= HI - EDGE):
        continue
    center = [int(round(coarse(x0 + nx / 2 - 0.5))), int(round(coarse(y0 + ny / 2 - 0.5)))]
    assert center in tile_centers, (t, center)
    tile_fp[x0:x0 + nx, y0:y0 + ny] = True
    tiles.append(dict(tile=t, center=center))
coords = np.array(run_cfg[f"sky_p0.{it}"]["point_grid"]["coordinates"])
rng = np.random.default_rng(SEED_TRUTH + 100)
points, pt_fp = [], np.zeros(sky.grid.shape, dtype=bool)
for row in hdul["POINTS"].data:
    i, j = int(round(row["X_PIX"] - 1)), int(round(row["Y_PIX"] - 1))
    if not (LO + EDGE <= i - 1 and i + 1 < HI - EDGE and LO + EDGE <= j - 1 and j + 1 < HI - EDGE):
        continue
    cx, cy = coarse(i), coarse(j)
    assert np.min(np.abs(coords - [cx, cy]).max(axis=1)) < 1e-6, (row["NAME"], cx, cy)
    pt_fp[i - 1:i + 2, j - 1:j + 2] = True
    points.append(dict(name=str(row["NAME"]), i=i - LO, j=j - LO,
                       block=[[cx + dx / FAC, cy + dy / FAC] for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                       alpha=float(rng.normal(ALPHA_MEAN, ALPHA_STD)), beta=float(rng.normal(BETA_MEAN, BETA_STD))))

i0 = {k: FLUX_SCALE * v[F] for k, v in dict(o0=comp["o0"], o2=comp["o2"], tiles=np.where(tile_fp, comp["t0"], 0.0),
                                             pts=np.where(pt_fp, comp["p0"], 0.0), total=total).items()}
i0["bg"] = i0["total"] - i0["o0"] - i0["o2"] - i0["tiles"] - i0["pts"]
assert i0["bg"].min() > -1e-6 * i0["total"].max(), i0["bg"].min()
i0["bg"] = i0["bg"].clip(0, None)
for p in points:
    p["flux_mjy"] = float(i0["pts"][p["i"] - 1:p["i"] + 2, p["j"] - 1:p["j"] + 2].sum() * OMEGA_PIX * 1e3)
for d in tiles:
    x0 = (N // (2 * FAC) + d["center"][0]) * FAC - 48
    y0 = (N // (2 * FAC) + d["center"][1]) * FAC - 48
    d["flux_jy"] = float(i0["tiles"][x0:x0 + 96, y0:y0 + 96].sum() * OMEGA_PIX)
jy = OMEGA_PIX
print(f"field (x{FLUX_SCALE:g}): total {i0['total'].sum() * jy:.3f} Jy | O0 {i0['o0'].sum() * jy:.3f} | "
      f"O2 {i0['o2'].sum() * jy:.4f} | tiles {i0['tiles'].sum() * jy:.4f} {[round(d['flux_jy'], 4) for d in tiles]} | "
      f"points {i0['pts'].sum() * jy:.4f} {[round(p['flux_mjy'], 2) for p in points]} mJy | BG {i0['bg'].sum() * jy:.3f}")

# %%
# ---------------------------------------------------------------------------
# Write both variants (alpha / beta fields of make_truth.py, point spectra as make_truth_comp.py).
# ---------------------------------------------------------------------------
geometry = dict(
    objects={k: dict(center=run_cfg[f"sky_{k}.{it}"]["grid"]["center"], space=run_cfg[f"sky_{k}.{it}"]["grid"]["space"])
             for k in ("o0", "o2")},
    tile_centers=[d["center"] for d in tiles],
    point_coordinates=[c for p in points for c in p["block"]],
)
i0_diffuse = i0["o0"] + i0["o2"] + i0["tiles"] + i0["bg"]
for variant in ("main", "beta0"):
    t1 = np.load(f"{OUT}/truth/{variant}/truth.npz")
    nu_ref, nu_eff, alpha, beta = float(t1["nu_ref"]), t1["nu_eff"], t1["alpha"], t1["beta"]
    alpha_pts, beta_pts = np.zeros((N, N)), np.zeros((N, N))
    for p in points:
        blk = (slice(p["i"] - 1, p["i"] + 2), slice(p["j"] - 1, p["j"] + 2))
        alpha_pts[blk] = p["alpha"]
        beta_pts[blk] = p["beta"] if variant == "main" else 0.0
    pmask = i0["pts"] > i0_diffuse
    alpha_eff, beta_eff = np.where(pmask, alpha_pts, alpha), np.where(pmask, beta_pts, beta)
    vdir = truth_dir(f"{variant}_comp")
    os.makedirs(vdir, exist_ok=True)
    np.savez(f"{vdir}/truth.npz", nu_ref=nu_ref, nu_eff=nu_eff, i0=i0_diffuse + i0["pts"], i0_diffuse=i0_diffuse,
             i0_obj=i0["o0"], i0_bg=i0["bg"], i0_o2=i0["o2"], i0_tiles=i0["tiles"], i0_pts=i0["pts"],
             alpha=alpha_eff, beta=beta_eff, alpha_diffuse=alpha, beta_diffuse=beta, alpha_pts=alpha_pts,
             beta_pts=beta_pts, seed=SEED_TRUTH, source_run=RUN, flux_scale=FLUX_SCALE)
    with open(f"{vdir}/components.yml", "w") as f:
        yaml.safe_dump(dict(geometry, points=[{k: v for k, v in p.items() if k != "block"} for p in points],
                            tiles=tiles, source_run=RUN, flux_scale=FLUX_SCALE), f, sort_keys=False,
                       default_flow_style=None, width=120)
    for f_hz in nu_eff:
        u = np.log(f_hz / nu_ref)
        s = i0_diffuse * np.exp(alpha * u + beta * u**2) + i0["pts"] * np.exp(alpha_pts * u + beta_pts * u**2)
        pyfits.PrimaryHDU((s * CONV_FACTOR).astype("float32").T).writeto(
            f"{vdir}/{PREFIX}_truth_{round(f_hz / 1e6)}MHz.fits", overwrite=True)
    print(f"saved {variant}_comp")

# %%
# overview: total I0 with the component boxes (as plot_from_fits.py sky_rgb_box) and the background alone
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, (arr, title) in zip(axes, [((i0_diffuse + i0["pts"]), f"truth from {os.path.basename(RUN)} (x{FLUX_SCALE:g})"),
                                   (i0["bg"], "background component")]):
    im = ax.imshow(arr.T * CONV_FACTOR, origin="lower", cmap="inferno", norm=matplotlib.colors.LogNorm(1e-3, 10))
    box = dict(fill=False, ec="white", lw=1.8)
    for g in geometry["objects"].values():
        x0, y0 = [(N // (2 * FAC) + g["center"][d] - g["space"][d] // 2) * FAC for d in range(2)]
        ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), g["space"][0] * FAC, g["space"][1] * FAC, **box))
    for c in geometry["tile_centers"]:
        x0, y0 = [(N // (2 * FAC) + c[d]) * FAC - 48 for d in range(2)]
        ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), 96, 96, **box))
    ax.scatter([p["i"] for p in points], [p["j"] for p in points], s=60, facecolors="none", edgecolors="white", lw=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="I0 [mJy/arcsec$^2$]")
fig.savefig(f"{truth_dir('main_comp')}/overview.png", dpi=110, bbox_inches="tight")
print("saved:", f"{truth_dir('main_comp')}/overview.png")

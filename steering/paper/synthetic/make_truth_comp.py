# %%
# ---------------------------------------------------------------------------
# Truth sky with the component structure of the reference run (version 2 of the
# synthetic test): the truth of make_truth.py (ESO 137-006 = O0 + diffuse
# background BG, same alpha / beta fields) plus the components of the reference
# run that lie fully inside the 0.5 deg field:
#   tiles   the tiles of t0 whose 96 x 96 fine-pixel footprint lies inside
#   points  the point sources of p0 whose 3 x 3 sub-pixel block lies inside,
#           each with its whole flux density in the central sub-pixel
#   O2      object o2: empty in the multi-frequency reference (its flux went to BG after the transition
#           from the old single-frequency model), so it is taken from the single-frequency stage REF_SF
#           (posterior mean of o2.5), put on the multi-frequency flux scale (median MF / SF ratio of the
#           total sky in its box, ~2: SF apparent sky and flux convention) and subtracted from BG there
# O2 and the tiles follow the alpha / beta fields of the diffuse emission; every
# point source gets its own alpha ~ N(ALPHA_MEAN, ALPHA_STD) and, in the "main"
# variant, beta ~ N(BETA_MEAN, BETA_STD) (beta = 0 in "beta0").
#
# Output OUT_COMP/truth/<variant>_comp/: truth.npz (model array order, Jy/sr; read by
# synth_common.TruthSky), FITS at the effective model frequencies, the alpha / beta
# FITS of the diffuse emission and an overview plot. The component geometry
# (object grids, tile centres, point coordinates in the reference run's convention)
# is stored for make_configs.py --comp.
#
# usage: python make_truth_comp.py   (after make_truth.py)
# ---------------------------------------------------------------------------
import os
import sys

import astropy.io.fits as pyfits
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import (
    ALPHA_MEAN, ALPHA_STD, BETA_MEAN, BETA_STD, CROP, FAC, N, OUT, PREFIX, REF_MF, REF_N, REF_SF, SEED_TRUTH,
    STEERING, truth_dir,
)

AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2 (save_fits.py)
PIX_AS = 7.03125 / 3  # fine pixel [arcsec]
OMEGA_PIX = (PIX_AS * AS2RAD) ** 2  # fine-pixel solid angle [sr]
EDGE = 3  # fine-pixel margin to the field edge for points and tiles
LO, HI = CROP.start, CROP.stop  # reference fine-pixel range of the field

fn = f"{REF_MF}/fits/{PREFIX}_1054MHz_posterior_mean.fits"
print("load:", fn)
hdul = pyfits.open(fn)
ref_cfg = {k: v for d in yaml.safe_load_all(open(f"{REF_MF}/opt.yml")) if d for k, v in d.items()}


def coarse(i):
    """Reference fine-pixel index -> coarse-pixel coordinate relative to the centre (point_grid convention)."""
    return (i + 0.5) / FAC - REF_N / (2 * FAC)


# %%
# ---------------------------------------------------------------------------
# Tiles fully inside the field (TILECUBE + TILEPOS; orientation checked against the TILES map).
# ---------------------------------------------------------------------------
tilepos, cube = hdul["TILEPOS"].data, np.asarray(hdul["TILECUBE"].data, dtype="float64")
tiles_map = np.asarray(hdul["TILES"].data[0, 0].T, dtype="float64")
tile_centers_ref = np.array(ref_cfg["sky_t0.6"]["tile_grid"]["center"])
i0_tiles = np.zeros((N, N))
tiles = []
for row in tilepos:
    t, x0, y0, nx, ny = int(row["TILE"]), int(row["X0_PIX"]) - 1, int(row["Y0_PIX"]) - 1, int(row["NX"]), int(row["NY"])
    if not (x0 >= LO + EDGE and y0 >= LO + EDGE and x0 + nx <= HI - EDGE and y0 + ny <= HI - EDGE):
        continue
    ref = tiles_map[x0:x0 + nx, y0:y0 + ny]
    cand = [cube[t], cube[t].T]
    err = [np.abs(c - ref).max() / np.abs(ref).max() for c in cand]
    tile = cand[int(np.argmin(err))] / CONV_FACTOR
    center = [int(round(coarse(x0 + nx / 2 - 0.5))), int(round(coarse(y0 + ny / 2 - 0.5)))]
    assert any(np.all(tile_centers_ref == center, axis=1)), (t, center)
    i0_tiles[x0 - LO:x0 - LO + nx, y0 - LO:y0 - LO + ny] += tile
    tiles.append(dict(tile=t, center=center, flux_jy=float(tile.sum() * OMEGA_PIX), match=float(min(err))))
print(f"tiles inside: {len(tiles)}", [(d["tile"], d["center"], round(d["flux_jy"], 4), f"{d['match']:.0e}") for d in tiles])

# %%
# ---------------------------------------------------------------------------
# O2 from the single-frequency stage, on the multi-frequency flux scale.
# ---------------------------------------------------------------------------
def sf_object(name):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import pickle

    import aim_resolve as aim

    cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{REF_SF}/opt.yml"), aim.get_builders)
    comp, sky = cfg.instantiate_sec(f"sky_{name}.5"), cfg.instantiate_sec("sky.5")
    with open(f"{REF_SF}/last.pkl", "rb") as f:
        samples, *_ = pickle.load(f)
    obj = np.mean([np.asarray(comp(x)) for x in samples], 0)
    tot = np.squeeze(np.mean([np.asarray(sky(x)) for x in samples], 0))
    h = hdul[name.upper()].header
    x0, y0 = int(round(REF_N / 2 + 1 - h["CRPIX1"])), int(round(REF_N / 2 + 1 - h["CRPIX2"]))
    box = (slice(x0, x0 + obj.shape[0]), slice(y0, y0 + obj.shape[1]))
    mf_tot = np.asarray(hdul[0].data[0, 0].T, dtype="float64")[box] / CONV_FACTOR
    sf_tot = tot[box]
    m = sf_tot > np.percentile(sf_tot, 50)
    scale = float(np.median(mf_tot[m] / sf_tot[m]))
    img = np.zeros((N, N))
    img[x0 - LO:x0 - LO + obj.shape[0], y0 - LO:y0 - LO + obj.shape[1]] = scale * obj
    print(f"{name} (single-frequency stage): flux {obj.sum() * OMEGA_PIX:.4f} Jy x MF/SF scale {scale:.2f} "
          f"-> {img.sum() * OMEGA_PIX:.4f} Jy at nu_ref")
    return img, scale


i0_o2, o2_scale = sf_object("o2")

# %%
# ---------------------------------------------------------------------------
# Point sources whose 3 x 3 block lies inside: the whole flux density in the central sub-pixel.
# ---------------------------------------------------------------------------
pts_tab = hdul["POINTS"].data
coords_ref = np.array(ref_cfg["sky_p0.6"]["point_grid"]["coordinates"])
rng = np.random.default_rng(SEED_TRUTH + 100)
points = []
for row in pts_tab:
    i, j = int(round(row["X_PIX"] - 1)), int(round(row["Y_PIX"] - 1))  # reference fine pixel (model order)
    if not (LO + EDGE <= i - 1 and i + 1 < HI - EDGE and LO + EDGE <= j - 1 and j + 1 < HI - EDGE):
        continue
    cx, cy = coarse(i), coarse(j)
    block = [[cx + dx / FAC, cy + dy / FAC] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    assert np.min(np.abs(coords_ref - [cx, cy]).max(axis=1)) < 1e-6, (row["NAME"], cx, cy)
    points.append(dict(name=str(row["NAME"]), i=i - LO, j=j - LO, flux_mjy=float(row["FLUX"]), block=block,
                       alpha=float(rng.normal(ALPHA_MEAN, ALPHA_STD)), beta=float(rng.normal(BETA_MEAN, BETA_STD))))
print(f"points inside: {len(points)}", [(p["name"], round(p["flux_mjy"], 2)) for p in points])

# %%
# ---------------------------------------------------------------------------
# Write both variants on top of the version-1 truth (same O0, BG, alpha, beta fields).
# ---------------------------------------------------------------------------
geometry = dict(
    objects={k: dict(center=ref_cfg[f"sky_{k}.6"]["grid"]["center"], space=ref_cfg[f"sky_{k}.6"]["grid"]["space"])
             for k in ("o0", "o2")},
    tile_centers=[d["center"] for d in tiles],
    point_coordinates=[c for p in points for c in p["block"]],
)
for variant in ("main", "beta0"):
    t1 = np.load(f"{OUT}/truth/{variant}/truth.npz")
    nu_ref, nu_eff, alpha, beta = float(t1["nu_ref"]), t1["nu_eff"], t1["alpha"], t1["beta"]
    # O2's flux is in the multi-frequency BG: take it out there (BG kept >= 5 % of its value)
    i0_bg = np.maximum(t1["i0_bg"] - i0_o2, 0.05 * t1["i0_bg"])
    i0_diffuse = t1["i0_obj"] + i0_bg + i0_o2 + i0_tiles
    i0_pts, alpha_pts, beta_pts = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    for p in points:
        i0_pts[p["i"], p["j"]] = p["flux_mjy"] * 1e-3 / OMEGA_PIX
        alpha_pts[p["i"], p["j"]] = p["alpha"]
        beta_pts[p["i"], p["j"]] = p["beta"] if variant == "main" else 0.0
    # alpha / beta of the total sky where a point source dominates its pixel (for maps and scatter plots)
    pmask = i0_pts > i0_diffuse
    alpha_eff, beta_eff = np.where(pmask, alpha_pts, alpha), np.where(pmask, beta_pts, beta)

    vdir = truth_dir(f"{variant}_comp")
    os.makedirs(vdir, exist_ok=True)
    np.savez(f"{vdir}/truth.npz", nu_ref=nu_ref, nu_eff=nu_eff, i0=i0_diffuse + i0_pts, i0_diffuse=i0_diffuse,
             i0_obj=t1["i0_obj"], i0_bg=i0_bg, i0_o2=i0_o2, i0_tiles=i0_tiles, i0_pts=i0_pts,
             alpha=alpha_eff, beta=beta_eff, alpha_diffuse=alpha, beta_diffuse=beta, alpha_pts=alpha_pts,
             beta_pts=beta_pts, seed=SEED_TRUTH)
    with open(f"{vdir}/components.yml", "w") as f:
        yaml.safe_dump(dict(geometry, points=[{k: v for k, v in p.items() if k != "block"} for p in points],
                            tiles=tiles), f, sort_keys=False, default_flow_style=None, width=120)
    for f_hz in nu_eff:
        u = np.log(f_hz / nu_ref)
        s = i0_diffuse * np.exp(alpha * u + beta * u**2) + i0_pts * np.exp(alpha_pts * u + beta_pts * u**2)
        pyfits.PrimaryHDU((s * CONV_FACTOR).astype("float32").T).writeto(
            f"{vdir}/{PREFIX}_truth_{round(f_hz / 1e6)}MHz.fits", overwrite=True)
    jy = OMEGA_PIX
    print(f"saved {variant}_comp: total flux at nu_ref {(i0_diffuse + i0_pts).sum() * jy:.3f} Jy "
          f"(O0 {t1['i0_obj'].sum() * jy:.3f}, BG {i0_bg.sum() * jy:.3f}, O2 {i0_o2.sum() * jy:.4f}, "
          f"BG change in the O2 box {(i0_bg - t1['i0_bg']).sum() * jy:+.4f}, "
          f"tiles {i0_tiles.sum() * jy:.4f}, points {i0_pts.sum() * jy:.4f} Jy)")

# %%
# overview: total I0 with the component boxes as in plot_from_fits.py (sky_rgb_box): outlines of the
# object cutouts and the tiles, circles for the point sources
fig, ax = plt.subplots(figsize=(8, 7.5))
tot = (i0_diffuse + i0_pts) * CONV_FACTOR
im = ax.imshow(tot.T, origin="lower", cmap="inferno", norm=matplotlib.colors.LogNorm(1e-3, 10))
box = dict(fill=False, ec="white", lw=1.8)
for k, g in geometry["objects"].items():  # object cutouts (model grids of the reference run)
    x0, y0 = [(N // (2 * FAC) + g["center"][d] - g["space"][d] // 2) * FAC for d in range(2)]
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), g["space"][0] * FAC, g["space"][1] * FAC, **box))
ts = np.array(ref_cfg["sky_t0.6"]["tile_grid"]["space"]) * FAC
for c in geometry["tile_centers"]:  # tiles (96 x 96 fine pixels)
    x0, y0 = [(N // (2 * FAC) + c[d]) * FAC - ts[d] // 2 for d in range(2)]
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), ts[0], ts[1], **box))
ax.scatter([p["i"] for p in points], [p["j"] for p in points], s=60, facecolors="none", edgecolors="white", lw=1)
plt.colorbar(im, ax=ax, label="I0 at nu_ref [mJy/arcsec$^2$]")
ax.set_title(f"truth with components: objects O0 / O2, {len(tiles)} tiles, {len(points)} point sources")
fig.savefig(f"{truth_dir('main_comp')}/overview.png", dpi=110, bbox_inches="tight")
print("saved:", f"{truth_dir('main_comp')}/overview.png")

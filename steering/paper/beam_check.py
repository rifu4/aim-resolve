# %%
# ---------------------------------------------------------------------------
# Checks of the primary-beam correction (beam in the forward model):
#   2  apparent-sky invariance: both runs fit the same data, so the new
#      (intrinsic) reconstruction times the model's beam must reproduce the old
#      (apparent, no beam) reconstruction where the data constrain it
#   3  beam prediction vs actual difference: intrinsic - apparent should equal
#      -d ln PB / d ln nu (alpha) and -curvature(ln PB) (beta); alpha vs radius
#      for point sources (POINTS table) and tiles (TILES layer), old and new, with the expected beam curve
#   4  pointing centre (metadata of the data)
#   5  uSARA / AIRI (Dabbech et al.): primary-beam corrected or not, from the
#      flux ratio to our apparent and intrinsic maps vs radius
#
# Definitions as in plot_from_fits.py / the model: alpha = straight-line slope of
# ln S over the 6 model frequencies (the model's alpha), beta = quadratic term of
# the log-parabola fit (curvature_from_cube), u = ln(nu / nu_ref), per sample.
# The beam is the model's: JimBeam averaged over the data channels of each bin
# with the channel weights (build_primary_beam, as in the likelihood).
#
# Output: runs/fast_vi_1f_1024_1z_b/opt/beam_check/
# ---------------------------------------------------------------------------
import glob
import json
import os
from types import SimpleNamespace

import astropy.io.fits as pyfits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

from aim_resolve.fast_resolve.beam import build_primary_beam
from aim_resolve.model.grid import SignalGrid

STEERING = "/scratch/users/rfuchs/packages/aim-resolve/steering"
OPT = f"{STEERING}/runs/fast_vi_1f_1024_1z_b/opt"
RUNS = {"old": "4_rec_3z_4_6f_1_it_1_it_1", "new": "4_rec_3z_1_6f_1"}  # old: no beam (apparent), new: beam in the model
DATA_MF = "/scratch/users/rfuchs/data/eso_961-1503mhz.npz"
CHAN_WEIGHTS = f"{STEERING}/runs/synth_05deg/prep/prep.npz"  # nu_data, w_chan of DATA_MF
DABBECH = {f"{m} {mhz}": f"{STEERING}/paper/fits/ESO137_{mhz}MHz_{m}.fits" for m in ["uSARA", "AIRI"] for mhz in [1053, 1399]}
PREFIX, REF_IDX = "eso137", 1
FLUX_MIN = 1e-2  # mJy/arcsec^2, signal mask (as plot_from_fits.py)
PB_MASK = 0.2  # PB(highest frequency) below which the correction is not trusted
APER = 4  # half width [fine pixels] of the point-source aperture
odir = f"{OPT}/beam_check"
os.makedirs(odir, exist_ok=True)
summary = {}

# %%
# ---------------------------------------------------------------------------
# Posterior samples (total sky, PRIMARY) of both runs: (n_samples, nfreq, N, N).
# ---------------------------------------------------------------------------
def load_run(run):
    fdir = f"{OPT}/{run}/fits"
    mhz = sorted({int(os.path.basename(f).split("_")[1][:-3]) for f in glob.glob(f"{fdir}/{PREFIX}_*MHz_posterior_mean.fits")})
    tags = sorted({os.path.basename(f).split("posterior_")[1][:-5]
                   for f in glob.glob(f"{fdir}/{PREFIX}_{mhz[0]}MHz_posterior_sample_*.fits")})
    freq = np.array([pyfits.getheader(f"{fdir}/{PREFIX}_{m}MHz_posterior_mean.fits")["CRVAL3"] for m in mhz])
    cube = np.stack([np.stack([pyfits.getdata(f"{fdir}/{PREFIX}_{m}MHz_posterior_{t}.fits", ext=0)[0, 0].T
                               for m in mhz]) for t in tags]).astype("float64")
    hdr = pyfits.getheader(f"{fdir}/{PREFIX}_{mhz[REF_IDX]}MHz_posterior_mean.fits")
    pts = pyfits.getdata(f"{fdir}/{PREFIX}_{mhz[REF_IDX]}MHz_posterior_mean.fits", extname="POINTS")
    tpos = pyfits.getdata(f"{fdir}/{PREFIX}_{mhz[REF_IDX]}MHz_posterior_mean.fits", extname="TILEPOS")
    # component spectra: point sources from the POINTS table, tiles from the TILES layer (not the total sky,
    # which contains the background) -> (n_samples, nfreq, n_sources)
    pt_flux = np.stack([np.stack([pyfits.getdata(f"{fdir}/{PREFIX}_{m}MHz_posterior_{t}.fits", extname="POINTS")["FLUX"]
                                  for m in mhz]) for t in tags]).astype("float64")
    tile_flux = np.zeros((len(tags), len(mhz), len(tpos)))
    for i, t in enumerate(tags):
        for j, m in enumerate(mhz):
            tl = np.nan_to_num(pyfits.getdata(f"{fdir}/{PREFIX}_{m}MHz_posterior_{t}.fits", extname="TILES")[0, 0].T)
            tile_flux[i, j] = [tl[x0:x0 + nx, y0:y0 + ny].sum() for x0, y0, nx, ny in
                               zip(tpos["X0_PIX"], tpos["Y0_PIX"], tpos["NX"], tpos["NY"])]
    return dict(freq=freq, cube=cube, hdr=hdr, pts=pts, tpos=tpos, pbcor=hdr.get("PBCOR", False),
                pt_flux=pt_flux, tile_flux=tile_flux)


runs = {k: load_run(r) for k, r in RUNS.items()}
freq = runs["new"]["freq"]
assert np.allclose(freq, runs["old"]["freq"]), "model frequencies differ"
u = np.log(freq / freq[REF_IDX])
N = runs["new"]["cube"].shape[-1]
print("model freqs [MHz]:", np.round(freq / 1e6, 2).tolist(), "| samples:",
      {k: r["cube"].shape[0] for k, r in runs.items()}, "| grid", N)
# straight-line slope and quadratic log-parabola coefficient as linear projections over the frequency axis
P_LIN = (u - u.mean()) / np.sum((u - u.mean()) ** 2)
P_QUAD = np.linalg.pinv(np.stack([u**2, u, np.ones_like(u)], axis=1))[0]


def alpha_beta(ln_cube, axis=-3):
    """alpha (straight-line slope) and beta (log-parabola) over the frequency axis."""
    return np.tensordot(P_LIN, ln_cube, axes=(0, axis)), np.tensordot(P_QUAD, ln_cube, axes=(0, axis))


# %%
# ---------------------------------------------------------------------------
# The model's beam (bin-averaged over the data channels, as in the likelihood).
# ---------------------------------------------------------------------------
grid = SignalGrid.build(space=(N // 3, N // 3), factor=3, fov=float(np.deg2rad(2.0)))
cw = np.load(CHAN_WEIGHTS)
obs_like = SimpleNamespace(freq=cw["nu_data"], weight=cw["w_chan"][None, None, :])
pb = build_primary_beam(grid, freq, obs_like)  # (nfreq, N, N)
ln_pb = np.log(np.clip(pb, 1e-12, None))
d_alpha_beam, d_beta_beam = alpha_beta(-ln_pb)  # intrinsic - apparent
pix_deg = np.rad2deg(grid.dis[0] / grid.fac)
xx, yy = np.meshgrid(np.arange(N) - N / 2, np.arange(N) - N / 2, indexing="ij")
r_arcmin = np.hypot(xx, yy) * pix_deg * 60
pb_ok = pb[-1] > PB_MASK
print(f"beam: PB(top freq) = {PB_MASK} at r = {r_arcmin[pb[-1] > PB_MASK].max():.1f} arcmin | "
      f"PB(1054) at 16' / 31': {pb[REF_IDX].flat[np.argmin(np.abs(r_arcmin - 16))]:.3f} / "
      f"{pb[REF_IDX].flat[np.argmin(np.abs(r_arcmin - 31))]:.3f}")

# %%
# ---------------------------------------------------------------------------
# alpha / beta per sample: old (apparent), new intrinsic, new apparent (x beam).
# ---------------------------------------------------------------------------
res = {}
for name, cube, extra in [("old", runs["old"]["cube"], 0.0), ("new_int", runs["new"]["cube"], 0.0),
                          ("new_app", runs["new"]["cube"], ln_pb)]:
    ln = np.log(np.clip(cube, 1e-12, None)) + extra
    a, b = alpha_beta(ln)
    flux = np.exp(ln[:, REF_IDX])
    res[name] = dict(alpha=a.mean(0), alpha_std=a.std(0, ddof=1), beta=b.mean(0), beta_std=b.std(0, ddof=1),
                     flux=flux.mean(0))
    del ln, a, b
signal = (res["old"]["flux"] > FLUX_MIN) & (res["new_app"]["flux"] > FLUX_MIN)
mask = signal & pb_ok
print(f"signal pixels: {signal.sum()} | of which PB(top) > {PB_MASK}: {mask.sum()}")

# %%
# ---------------------------------------------------------------------------
# Check 2 + 3: maps and radial trends of the differences.
# ---------------------------------------------------------------------------
diff = {
    "alpha": dict(app=res["new_app"]["alpha"] - res["old"]["alpha"], int=res["new_int"]["alpha"] - res["old"]["alpha"],
                  pred=d_alpha_beam),
    "beta": dict(app=res["new_app"]["beta"] - res["old"]["beta"], int=res["new_int"]["beta"] - res["old"]["beta"],
                 pred=d_beta_beam),
}
summary["invariance"] = {}
r_bins = np.arange(0, 66, 3)
for q, d in diff.items():
    rows = []
    for lo, hi in zip(r_bins[:-1], r_bins[1:]):
        m = signal & (r_arcmin >= lo) & (r_arcmin < hi)
        if m.sum() < 30:
            continue
        rows.append(dict(r=0.5 * (lo + hi), n=int(m.sum()), app_median=float(np.median(d["app"][m])),
                         app_mad=float(1.4826 * np.median(np.abs(d["app"][m] - np.median(d["app"][m])))),
                         int_median=float(np.median(d["int"][m])), pred_median=float(np.median(d["pred"][m]))))
    summary["invariance"][q] = rows
    print(f"\n{q}: new - old vs radius (signal pixels)   [apparent: should be 0 | intrinsic: should equal the beam prediction]")
    print(f"{'r [\']':>6} {'n':>7} {'app med':>8} {'app MAD':>8} {'int med':>8} {'pred med':>9}")
    for rw in rows:
        print(f"{rw['r']:6.1f} {rw['n']:7d} {rw['app_median']:+8.3f} {rw['app_mad']:8.3f} {rw['int_median']:+8.3f} {rw['pred_median']:+9.3f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for c, q in enumerate(["alpha", "beta"]):
    rows = summary["invariance"][q]
    rr = [x["r"] for x in rows]
    ax = axes[0, c]
    ax.plot(rr, [x["int_median"] for x in rows], "o-", label="new intrinsic - old (observed)")
    ax.plot(rr, [x["pred_median"] for x in rows], "k--", label="beam prediction (intrinsic - apparent)")
    ax.plot(rr, [x["app_median"] for x in rows], "s-", label="new apparent (x beam) - old: should be 0")
    ax.fill_between(rr, [x["app_median"] - x["app_mad"] for x in rows], [x["app_median"] + x["app_mad"] for x in rows],
                    color="C2", alpha=0.2)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.axvline(r_arcmin[pb_ok].max(), color="r", ls=":", lw=0.8, label=f"PB(top) = {PB_MASK}")
    ax.set_xlabel("distance from the pointing centre [arcmin]")
    ax.set_ylabel(f"$\\Delta\\{q}$ (median over signal pixels)")
    ax.set_ylim(-1.5, 5 if q == "alpha" else 8)
    ax.legend(fontsize=8)
    ax = axes[1, c]
    m = signal
    ax.hexbin(diff[q]["pred"][m], diff[q]["int"][m], gridsize=80, bins="log", extent=(-1, 5, -3, 6), cmap="Blues")
    ax.plot([-1, 5], [-1, 5], "k--", lw=0.8)
    ax.set_xlabel(f"beam prediction $\\Delta\\{q}$")
    ax.set_ylabel(f"new intrinsic - old $\\Delta\\{q}$")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-3, 6)
fig.tight_layout()
fig.savefig(f"{odir}/check2_3_radial.png", dpi=120)
plt.close(fig)

# maps: full field (binned 4x for display) and zooms on the two galaxies
def zoom(name):
    # ESO137-006 (CST1 inside) and ESO137-007, as plot_from_fits.py COMPONENTS (centre, size in deg)
    cen, size = {"ESO137-006": ((0.0, -0.05), (0.32, 0.16)), "ESO137-007": ((0.18, 0.31), (0.5, 0.18))}[name]
    c = np.array(cen) / pix_deg + N / 2
    h = np.array(size) / pix_deg / 2
    return slice(int(c[0] - h[0]), int(c[0] + h[0])), slice(int(c[1] - h[1]), int(c[1] + h[1]))


for region in ["full", "ESO137-006", "ESO137-007"]:
    sl = (slice(None, None, 4), slice(None, None, 4)) if region == "full" else zoom(region)
    fig, axes = plt.subplots(2, 5, figsize=(26, 9.5 if region == "full" else 6.5))
    for r_, q in enumerate(["alpha", "beta"]):
        lim = (-2.5, 0.5) if q == "alpha" else (-1.5, 1.5)
        panels = [
            (res["old"][q], f"old {q} (no beam = apparent)", "RdYlBu_r" if q == "alpha" else "coolwarm", lim),
            (res["new_app"][q], f"new {q} x beam (apparent)", "RdYlBu_r" if q == "alpha" else "coolwarm", lim),
            (diff[q]["app"], "new apparent - old (should be 0)", "PuOr", (-1, 1)),
            (diff[q]["int"], "new intrinsic - old", "PuOr", (-3, 3)),
            (diff[q]["pred"], "beam prediction", "PuOr", (-3, 3)),
        ]
        for c, (arr, t, cmap, (vmin, vmax)) in enumerate(panels):
            ax = axes[r_, c]
            im = ax.imshow(np.where(signal, arr, np.nan)[sl].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(pb[-1][sl].T, levels=[PB_MASK, 0.5, 0.8], colors="0.4", linewidths=0.5)
            ax.set_title(t, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{region}: signal pixels (flux > {FLUX_MIN:g} mJy/arcsec$^2$ in both); grey: PB(1469 MHz) = {PB_MASK}, 0.5, 0.8")
    fig.tight_layout()
    fig.savefig(f"{odir}/check2_maps_{region}.png", dpi=110)
    plt.close(fig)

# %%
# ---------------------------------------------------------------------------
# Check 3: alpha vs radius for point sources (aperture spectrum) and tiles (tile spectrum).
# ---------------------------------------------------------------------------
def source_alpha(flux):
    """alpha, beta of each component spectrum (n_samples, nfreq, n_src) -> mean, std over the samples."""
    a, b = alpha_beta(np.log(np.clip(flux, 1e-30, None)), axis=1)  # (n_samples, n_src)
    return a.mean(0), a.std(0, ddof=1), b.mean(0), b.std(0, ddof=1)


pts, tpos = runs["new"]["pts"], runs["new"]["tpos"]
pt_boxes = [(slice(int(round(x)) - APER, int(round(x)) + APER + 1), slice(int(round(y)) - APER, int(round(y)) + APER + 1))
            for x, y in zip(pts["X_PIX"], pts["Y_PIX"])]
tile_boxes = [(slice(x0, x0 + nx), slice(y0, y0 + ny)) for x0, y0, nx, ny in zip(tpos["X0_PIX"], tpos["Y0_PIX"], tpos["NX"], tpos["NY"])]
src = {}
for kind, boxes, cx, cy in [("points", pt_boxes, pts["X_PIX"], pts["Y_PIX"]),
                            ("tiles", tile_boxes, tpos["X0_PIX"] + tpos["NX"] / 2, tpos["Y0_PIX"] + tpos["NY"] / 2)]:
    r_src = np.hypot(cx - N / 2, cy - N / 2) * pix_deg * 60
    pb_src = np.array([pb[:, sl[0], sl[1]].mean(axis=(-2, -1)) for sl in boxes])  # (n_src, nfreq)
    pred = alpha_beta(-np.log(pb_src), axis=1)
    src[kind] = dict(r=r_src, pred_alpha=pred[0], pred_beta=pred[1], pb_top=pb_src[:, -1])
    key = "pt_flux" if kind == "points" else "tile_flux"
    for run_name, run_key in [("old", "old"), ("new_int", "new")]:
        am, asd, bm, bsd = source_alpha(runs[run_key][key])
        src[kind][run_name] = dict(alpha=am, alpha_std=asd, beta=bm, beta_std=bsd)
    src[kind]["flux"] = runs["new"][key][:, REF_IDX].mean(0)

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True)
rr = np.linspace(0, 62, 200)
r_pix = np.argmin(np.abs(r_arcmin[N // 2:, N // 2] - rr[:, None]), axis=1) + N // 2
pred_curve_a = d_alpha_beam[r_pix, N // 2]
pred_curve_b = d_beta_beam[r_pix, N // 2]
for c, kind in enumerate(["points", "tiles"]):
    s = src[kind]
    for r_, q, pc in [(0, "alpha", pred_curve_a), (1, "beta", pred_curve_b)]:
        ax = axes[r_, c]
        ax.errorbar(s["r"], s["old"][q], yerr=s["old"][f"{q}_std"], fmt="o", ms=4, color="C3", label="old (no beam)")
        ax.errorbar(s["r"], s["new_int"][q], yerr=s["new_int"][f"{q}_std"], fmt="s", ms=4, color="C0", label="new (beam in model)")
        ax.plot(s["r"], s["old"][q] + s[f"pred_{q}"], "x", color="k", ms=5, label="old + beam prediction")
        med_new = np.median(s["new_int"][q][s["r"] < 30])
        ax.plot(rr, med_new - pc, "C3--", lw=0.8, label="expected trend without beam")
        ax.axvline(r_arcmin[pb_ok].max(), color="r", ls=":", lw=0.8)
        ax.set_ylabel(f"$\\{q}$")
        ax.set_ylim(-6, 2) if q == "alpha" else ax.set_ylim(-4, 6)
        ax.set_title(f"{kind}: {q} of the component spectrum")
        ax.grid(alpha=0.2)
    axes[1, c].set_xlabel("distance from the pointing centre [arcmin]")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{odir}/check3_alpha_vs_radius.png", dpi=120)
plt.close(fig)
summary["sources"] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else {a: b.tolist() for a, b in vv.items()})
                          for kk, vv in v.items()} for k, v in src.items()}
for kind in ["points", "tiles"]:
    s = src[kind]
    ok = s["pb_top"] > PB_MASK
    for run_name in ["old", "new_int"]:
        sl_, ic = np.polyfit(s["r"][ok], s[run_name]["alpha"][ok], 1)
        print(f"{kind} ({ok.sum()} with PB > {PB_MASK}): {run_name:7s} alpha vs r: slope {sl_ * 10:+.3f} per 10' | "
              f"median alpha {np.median(s[run_name]['alpha'][ok]):+.2f}")
    resid = s["new_int"]["alpha"] - (s["old"]["alpha"] + s["pred_alpha"])
    print(f"{kind}: new - (old + beam prediction): median {np.median(resid[ok]):+.3f}, MAD {1.4826 * np.median(np.abs(resid[ok] - np.median(resid[ok]))):.3f}")

# %%
# ---------------------------------------------------------------------------
# Check 4: pointing centre (metadata).
# ---------------------------------------------------------------------------
with np.load(DATA_MF, allow_pickle=True) as d:
    cols = list(d["auxtable_FIELD_0000"])
    dirs = {c: np.rad2deg(np.asarray(d[f"auxtable_FIELD_{cols.index(c) + 1:04d}"], float)).reshape(-1)[:2].tolist()
            for c in cols if c.endswith("_DIR")}
    tables = sorted({k.split("_")[1] for k in d.files if k.startswith("auxtable")})
summary["pointing"] = dict(field_dirs_deg=dirs, aux_tables=tables, fits_obs=[runs["new"]["hdr"]["OBSRA"], runs["new"]["hdr"]["OBSDEC"]])
print("\npointing: FIELD", dirs, "| tables:", tables)

# %%
# ---------------------------------------------------------------------------
# Check 5: uSARA / AIRI vs our apparent (old) and intrinsic (new) maps, per point source.
# ---------------------------------------------------------------------------
our = {"old (apparent)": runs["old"]["cube"].mean(0), "new (intrinsic)": runs["new"]["cube"].mean(0)}
our_wcs = WCS(runs["new"]["hdr"]).celestial
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
summary["dabbech"] = {}
for c, mhz in enumerate([1053, 1399]):
    fi = int(np.argmin(np.abs(freq / 1e6 - mhz)))
    ax = axes[c]
    for m in ["uSARA", "AIRI"]:
        fn = DABBECH[f"{m} {mhz}"]
        img = np.asarray(pyfits.getdata(fn), dtype="float64").squeeze()  # (dec, ra), Jy/pixel
        w = WCS(pyfits.getheader(fn)).celestial
        h = pyfits.getheader(fn)
        pix_area = abs(h["CDELT1"] * h["CDELT2"]) * 3600**2
        aper_as = (2 * APER + 1) * pix_deg * 3600
        na = int(round(aper_as / 2 / (abs(h["CDELT1"]) * 3600)))
        for lab, cube_mean in our.items():
            ratios, rs = [], []
            for k in range(len(pts)):
                px, py = w.world_to_pixel_values(pts["RA"][k], pts["DEC"][k])
                px, py = int(round(float(px))), int(round(float(py)))
                if not (na <= px < img.shape[1] - na and na <= py < img.shape[0] - na):
                    continue
                f_d = img[py - na:py + na + 1, px - na:px + na + 1].sum() * 1e3  # mJy
                sl = pt_boxes[k]
                f_o = cube_mean[fi, sl[0], sl[1]].sum() * (pix_deg * 3600) ** 2  # mJy
                if f_o > 0 and f_d > 0:
                    ratios.append(f_d / f_o)
                    rs.append(src["points"]["r"][k])
            rs, ratios = np.array(rs), np.array(ratios)
            ax.plot(rs, ratios, "o" if "old" in lab else "s", ms=4, label=f"{m} / {lab}", alpha=0.7)
            ok = rs < 50
            sl_ = np.polyfit(rs[ok], np.log(ratios[ok]), 1)[0] if ok.sum() > 3 else np.nan
            summary["dabbech"][f"{m} {mhz} / {lab}"] = dict(n=int(ratios.size), median=float(np.median(ratios)), dlnratio_dr_per10=float(sl_ * 10))
            print(f"{m} {mhz} MHz / {lab}: {ratios.size} sources, median ratio {np.median(ratios):.2f}, d ln(ratio)/dr {sl_ * 10:+.3f} per 10'")
    rr2 = np.linspace(0, 60, 100)
    pb_c = pb[fi][np.argmin(np.abs(r_arcmin[N // 2:, N // 2] - rr2[:, None]), axis=1) + N // 2, N // 2]
    ax.plot(rr2, pb_c, "k--", lw=0.8, label="PB (if Dabbech apparent, ratio to ours intrinsic)")
    ax.set_yscale("log")
    ax.set_xlabel("distance from the pointing centre [arcmin]")
    ax.set_ylabel("flux ratio (aperture)")
    ax.set_title(f"{mhz} MHz: point sources")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(f"{odir}/check5_dabbech_ratio.png", dpi=120)
plt.close(fig)

with open(f"{odir}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("\nsaved:", odir)

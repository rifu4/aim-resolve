# %%
# ---------------------------------------------------------------------------
# Truth sky of the 1 deg synthetic test: the two galaxies ESO 137-006 (object O0)
# and ESO 137-007 (object O1) plus the diffuse background, taken from the single-
# frequency zoom run RUN (posterior mean of each component on the reference grid),
# cut to the central 1 deg x 1 deg. The small components of the run (O2, O3, the
# tiles and the point sources) are left out entirely. O1 lies fully inside
# (its emission reaches 0.36 deg from the centre, its model box 0.49 deg).
#
# The run's sky is scaled by FLUX_SCALE = 2 (the single-frequency data are on half
# the flux scale of the multi-frequency data) and taken as the intrinsic truth (the
# single-frequency stages have no primary beam). Spectra: smoothed Gaussian random
# fields alpha / beta on the 1 deg field (as make_truth.py, new realisation for the
# larger field); variants "main" (beta field) and "beta0" (beta = 0, same alpha).
#
# Output OUT_1DEG/truth/<variant>/: truth.npz (model array order, Jy/sr; keys as
# make_truth_comp.py: i0, i0_bg, i0_o0, i0_o1, alpha, beta, ...), components.yml
# (object grids), FITS at the 8 model frequencies (mf_ext_8f.yml), overview.png.
#
# --fits: take the components from the FITS of a multi-frequency run instead (posterior mean at 1054 MHz,
#         extensions BG, O0, O1; default run REF_MF = 4_rec_3z_1_6f_1): already on the multi-frequency flux
#         scale and intrinsic (primary beam in its forward model), so no flux scaling
#
# --seed S: seed of the alpha / beta fields (default SEED_TRUTH)
#
# usage: python make_truth_1deg.py [--fits] [--seed S] [RUN_DIR]   (default: REF_DIR/opt/4_rec_3z_3, with --fits REF_MF)
# ---------------------------------------------------------------------------
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import astropy.io.fits as pyfits
import matplotlib
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import (
    ALPHA_MEAN, ALPHA_STD, BETA_MEAN, BETA_STD, FAC, OUT, PREFIX, REF_DIR, REF_IDX, REF_MF, REF_N, SEED_TRUTH, SMOOTH_ARCSEC,
    STEERING,
)

import aim_resolve as aim
from aim_resolve.model.map import map_signal

FITS = "--fits" in sys.argv
args = [a for a in sys.argv[1:] if a != "--fits"]
SEED = SEED_TRUTH
if "--seed" in args:
    i = args.index("--seed")
    SEED = int(args[i + 1])
    del args[i:i + 2]
RUN = os.path.abspath(args[0]) if args else (REF_MF if FITS else f"{REF_DIR}/opt/4_rec_3z_3")
OUT_1DEG = f"{STEERING}/runs/synth_1deg"
FOV_DEG = 1.0
N_COARSE = 512  # coarse pixels of the field (7.03125 arcsec)
N = N_COARSE * FAC  # 1536 fine pixels (2.34375 arcsec)
LO = (REF_N - N) // 2
F = (slice(LO, LO + N), slice(LO, LO + N))
KEEP = ("bg", "o0", "o1")  # components of the truth
FLUX_SCALE = 1.0 if FITS else 2.0
AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2
PIX_AS = 7.03125 / 3
OMEGA_PIX = (PIX_AS * AS2RAD) ** 2

# %%
# ---------------------------------------------------------------------------
# Components of the run's last stage (posterior mean) on the reference grid, cut to the field.
# ---------------------------------------------------------------------------
run_cfg = {k: v for d in yaml.safe_load_all(open(f"{RUN}/opt.yml")) if d for k, v in d.items()}
it = max(int(k.split(".")[1]) for k in run_cfg if k.startswith("sky.") and k.split(".")[1].isdigit())
i0, flux_full = {}, {}
if FITS:
    hdul = pyfits.open(f"{RUN}/fits/{PREFIX}_1054MHz_posterior_mean.fits")
    for k in KEEP:
        name = "BG" if k == "bg" else k.upper()
        h, v = hdul[name].header, np.asarray(hdul[name].data[0, 0].T, dtype="float64") / CONV_FACTOR
        full = np.zeros((REF_N, REF_N))
        x0, y0 = int(round(REF_N / 2 + 1 - h["CRPIX1"])), int(round(REF_N / 2 + 1 - h["CRPIX2"]))
        full[x0:x0 + v.shape[0], y0:y0 + v.shape[1]] = v
        i0[k] = full[F]
        print(f"{k} ({name}): {i0[k].sum() * OMEGA_PIX:.3f} Jy in the field ({i0[k].sum() / full.sum():.4f} of its total)")
else:
    cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{RUN}/opt.yml"), aim.get_builders)
    sky = cfg.instantiate_sec(f"sky.{it}")
    with open(f"{RUN}/last.pkl", "rb") as f:
        samples, *_ = pickle.load(f)
    for m in sky.models:
        k = m.prefix.split(".")[0]
        v = np.squeeze(np.mean([np.asarray(m(x)) for x in samples], 0)).astype("float64")
        v = v if v.shape == sky.grid.shape else np.asarray(map_signal(m.grid, sky.grid)(v))
        flux_full[k] = v.sum() * OMEGA_PIX * FLUX_SCALE
        if k in KEEP:
            i0[k] = FLUX_SCALE * v[F]
            print(f"{k}: {i0[k].sum() * OMEGA_PIX:.3f} Jy in the field ({i0[k].sum() * OMEGA_PIX / flux_full[k]:.4f} of its total)")
        else:
            print(f"{k}: left out ({v[F].sum() * OMEGA_PIX * FLUX_SCALE:.4f} Jy in the field)")
i0_tot = sum(i0.values())
print(f"truth: {i0_tot.sum() * OMEGA_PIX:.3f} Jy in the 1 deg field")

# %%
# ---------------------------------------------------------------------------
# alpha / beta: smoothed Gaussian random fields on the 1 deg field (as make_truth.py).
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)


def grf(mean, std):
    f = gaussian_filter(rng.standard_normal((N, N)), SMOOTH_ARCSEC / PIX_AS, mode="wrap")
    return mean + std * (f - f.mean()) / f.std()


alpha = grf(ALPHA_MEAN, ALPHA_STD)
beta_field = grf(BETA_MEAN, BETA_STD)
mf8 = {k: v for d in yaml.safe_load_all(open(f"{STEERING}/nifty/config/mf_ext_8f.yml")) if d for k, v in d.items()}
nu_eff = np.asarray(mf8["freq"], dtype="float64")
nu_ref = float(nu_eff[REF_IDX])
geometry = dict(seed=SEED, objects={k: dict(center=run_cfg[f"sky_{k}.{it}"]["grid"]["center"], space=run_cfg[f"sky_{k}.{it}"]["grid"]["space"])
                         for k in ("o0", "o1")}, fov_deg=FOV_DEG, n_coarse=N_COARSE, source_run=RUN, flux_scale=FLUX_SCALE)
for variant, beta in [("main", beta_field), ("beta0", np.zeros_like(beta_field))]:
    vdir = f"{OUT_1DEG}/truth/{variant}"
    os.makedirs(vdir, exist_ok=True)
    np.savez(f"{vdir}/truth.npz", nu_ref=nu_ref, nu_eff=nu_eff, i0=i0_tot, i0_bg=i0["bg"], i0_o0=i0["o0"], i0_o1=i0["o1"],
             alpha=alpha, beta=beta, seed=SEED, smooth_arcsec=SMOOTH_ARCSEC, fov_deg=FOV_DEG)
    with open(f"{vdir}/components.yml", "w") as f:
        yaml.safe_dump(geometry, f, sort_keys=False, default_flow_style=None, width=120)
    for f_hz in nu_eff:
        u = np.log(f_hz / nu_ref)
        pyfits.PrimaryHDU((i0_tot * np.exp(alpha * u + beta * u**2) * CONV_FACTOR).astype("float32").T).writeto(
            f"{vdir}/{PREFIX}_truth_{round(f_hz / 1e6)}MHz.fits", overwrite=True)
    print(f"saved {variant}: {vdir}")

# %%
# overview: total I0 with the object boxes (as plot_from_fits.py sky_rgb_box), the background alone, alpha and beta
fig, axes = plt.subplots(2, 2, figsize=(16, 15))
box = dict(fill=False, ec="white", lw=1.8)
for ax, (arr, title, kw, label) in zip(axes.ravel(), [
    (i0_tot * CONV_FACTOR, f"truth I0 at {nu_ref / 1e6:.0f} MHz (bg + O0 + O1, from {os.path.basename(RUN)}, x{FLUX_SCALE:g})",
     dict(cmap="inferno", norm=matplotlib.colors.LogNorm(1e-3, 10)), "I0 [mJy/arcsec$^2$]"),
    (i0["bg"] * CONV_FACTOR, "background component", dict(cmap="inferno", norm=matplotlib.colors.LogNorm(1e-3, 10)),
     "I0 [mJy/arcsec$^2$]"),
    (alpha, "truth alpha", dict(cmap="RdYlBu_r", vmin=-2.5, vmax=0.5), "alpha"),
    (beta_field, "truth beta (variant main; beta0: 0)", dict(cmap="coolwarm", vmin=-1, vmax=1), "beta"),
]):
    im = ax.imshow(arr.T, origin="lower", **kw)
    for g in geometry["objects"].values():
        x0, y0 = [(N_COARSE // 2 + g["center"][d] - g["space"][d] // 2) * FAC for d in range(2)]
        ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), g["space"][0] * FAC, g["space"][1] * FAC, **box))
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, label=label, shrink=0.85)
fig.suptitle(f"synthetic truth, {FOV_DEG:g} deg x {FOV_DEG:g} deg field ({N} x {N} fine pixels), alpha / beta seed {SEED}")
fig.tight_layout()
fig.savefig(f"{OUT_1DEG}/truth/overview.png", dpi=90, bbox_inches="tight")
print("saved:", f"{OUT_1DEG}/truth/overview.png")

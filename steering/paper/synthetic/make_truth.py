# %%
# ---------------------------------------------------------------------------
# Phase 1 of the synthetic-data test (../synthetic_test_plan.md): truth sky.
#
# I0 = ESO 137-006 (object O0) + diffuse background (BG) of the reference run (posterior mean at nu_ref,
# intrinsic sky since the beam is in its forward model), placed on the
# 0.5 deg field (768^2 fine pixels). alpha / beta are smoothed
# Gaussian random fields:
#     S(x, nu) = I0(x) exp(alpha u + beta u^2),  u = ln(nu / nu_ref).
# Variants: "main" (beta field) and "beta0" (beta = 0, same alpha).
#
# Output OUT/truth/<variant>/: truth.npz (model array order, Jy/sr), FITS at
# the effective model frequencies and the alpha / beta FITS (save_fits.py
# conventions), and an overview plot.
# ---------------------------------------------------------------------------
import os
import sys

import astropy.io.fits as pyfits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import (
    ALPHA_MEAN, ALPHA_STD, BETA_MEAN, BETA_STD, CROP, N, OBJECT, OUT, PREFIX, REF_IDX, REF_MF, REF_N, SEED_TRUTH,
    SMOOTH_ARCSEC,
)

AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2 (save_fits.py)
PIX_AS = 7.03125 / 3  # fine pixel [arcsec]

prep = np.load(f"{OUT}/prep/prep.npz")
nu_eff = prep["nu_eff"]
nu_ref = nu_eff[REF_IDX]

# %%
# ---------------------------------------------------------------------------
# I0: object O0 of the reference run on the field (model array order, Jy/sr).
# ---------------------------------------------------------------------------
fn = f"{REF_MF}/fits/{PREFIX}_1054MHz_posterior_mean.fits"
print("load:", fn)
hdul = pyfits.open(fn)
h = hdul[OBJECT].header
obj = np.asarray(hdul[OBJECT].data[0, 0].T, dtype="float64") / CONV_FACTOR
off = (int(round(REF_N / 2 + 1 - h["CRPIX1"])) - CROP.start, int(round(REF_N / 2 + 1 - h["CRPIX2"])) - CROP.start)
if off[0] < 0 or off[1] < 0 or off[0] + obj.shape[0] > N or off[1] + obj.shape[1] > N:
    raise ValueError(f"{OBJECT} ({obj.shape} at {off}) does not fit into the {N}^2 field")
i0_obj = np.zeros((N, N))
i0_obj[off[0]:off[0] + obj.shape[0], off[1]:off[1] + obj.shape[1]] = obj
# diffuse background of the reference run, so the truth has no empty (zero) regions;
# the reconstruction still uses a single correlated field for the total sky
i0_bg = np.asarray(hdul["BG"].data[0, 0].T, dtype="float64")[CROP, CROP] / CONV_FACTOR
i0 = i0_obj + i0_bg
jy = (PIX_AS * AS2RAD) ** 2
print(f"{OBJECT}: {obj.shape} at field pixel {off} | flux at nu_ref {i0_obj.sum() * jy:.3f} Jy")
print(f"BG: flux at nu_ref {i0_bg.sum() * jy:.3f} Jy, {i0_bg.min():.2e}-{i0_bg.max():.2e} Jy/sr")

# %%
# ---------------------------------------------------------------------------
# alpha / beta: smoothed Gaussian random fields.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED_TRUTH)


def grf(mean, std):
    f = gaussian_filter(rng.standard_normal((N, N)), SMOOTH_ARCSEC / PIX_AS, mode="wrap")
    return mean + std * (f - f.mean()) / f.std()


alpha = grf(ALPHA_MEAN, ALPHA_STD)
beta_field = grf(BETA_MEAN, BETA_STD)


def header(freq, ref_hdr=hdul[0].header):
    hd = pyfits.Header()
    for k in ["OBSRA", "OBSDEC", "CTYPE1", "CRVAL1", "CDELT1", "CUNIT1", "CTYPE2", "CRVAL2", "CDELT2", "CUNIT2",
              "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4", "EQUINOX"]:
        if k in ref_hdr:
            hd[k] = ref_hdr[k]
    hd["ORIGIN"] = "aim-resolve synthetic truth"
    hd["CRPIX1"] = N / 2 + 1
    hd["CRPIX2"] = N / 2 + 1
    hd["CTYPE3"], hd["CRVAL3"], hd["CDELT3"], hd["CRPIX3"], hd["CUNIT3"] = "FREQ", float(freq), 1.0, 1.0, "Hz"
    return hd


def write(arr, fn, freq, bunit):
    hd = header(freq)
    hd["BUNIT"] = bunit
    pyfits.PrimaryHDU(np.asarray(arr, dtype="float32").T[None, None], header=hd).writeto(fn, overwrite=True)


# %%
# ---------------------------------------------------------------------------
# Write both variants.
# ---------------------------------------------------------------------------
for variant, beta in [("main", beta_field), ("beta0", np.zeros_like(beta_field))]:
    vdir = f"{OUT}/truth/{variant}"
    os.makedirs(vdir, exist_ok=True)
    for f_hz in nu_eff:
        u = np.log(f_hz / nu_ref)
        write(i0 * np.exp(alpha * u + beta * u**2) * CONV_FACTOR,
              f"{vdir}/{PREFIX}_truth_{round(f_hz / 1e6)}MHz.fits", f_hz, "mJy/arcsec^2")
    write(alpha, f"{vdir}/{PREFIX}_truth_spectral_index.fits", nu_ref, "")
    write(beta, f"{vdir}/{PREFIX}_truth_curvature.fits", nu_ref, "")
    np.savez(f"{vdir}/truth.npz", nu_ref=nu_ref, nu_eff=nu_eff, i0=i0, i0_obj=i0_obj, i0_bg=i0_bg, alpha=alpha, beta=beta,
             obj_offset=np.array(off), seed=SEED_TRUTH, smooth_arcsec=SMOOTH_ARCSEC)
    print(f"saved {variant}: truth.npz and FITS at", [round(f / 1e6) for f in nu_eff], "MHz")

    m = i0_obj > np.percentile(i0_obj[i0_obj > 0], 50)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    vmax = i0.max()
    for ax, (arr, t, kw) in zip(axes, [
        (np.log10(np.clip(i0, vmax * 1e-4, None)), "I0 (log, Jy/sr)", dict(cmap="inferno")),
        (np.where(m, alpha, np.nan), "alpha (brighter half of O0)", dict(cmap="RdYlBu_r", vmin=-2.5, vmax=0.5)),
        (np.where(m, beta, np.nan), "beta (brighter half of O0)", dict(cmap="coolwarm", vmin=-1, vmax=1)),
    ]):
        im = ax.imshow(arr.T, origin="lower", **kw)
        ax.set_title(t)
        plt.colorbar(im, ax=ax, shrink=0.8)
    axes[3].hist(alpha[m], 60, alpha=0.6, label="alpha")
    axes[3].hist(beta[m], 60, alpha=0.6, label="beta")
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(f"{vdir}/truth_overview.png", dpi=110)
    plt.close(fig)
    print(f"  alpha in mask: {alpha[m].mean():+.3f} +- {alpha[m].std():.3f} | beta {beta[m].mean():+.3f} +- {beta[m].std():.3f}")

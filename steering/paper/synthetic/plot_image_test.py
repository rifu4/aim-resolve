# %%
# ---------------------------------------------------------------------------
# Plots of the image-domain test of the spectral model with curvature
# (make_image_test.py): reconstruction vs truth for I0 (at nu_ref), alpha and
# beta in the high-flux mask, and the spectral deviations delta, which are zero
# in the truth (exact log-parabola).
#
# usage: python plot_image_test.py [run_dir] [truth_variant] [sky_section]
# Output: <run_dir>/compare/image_test_maps.png, image_test_deviations.png
# ---------------------------------------------------------------------------
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import OUT, REF_IDX, STEERING, TruthSky

import aim_resolve as aim

run = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else f"{OUT}/image_curv/A_img"
truth_name = sys.argv[2] if len(sys.argv) > 2 else "main"
sky_sec = sys.argv[3] if len(sys.argv) > 3 else "sky.0"
odir = f"{run}/compare"
os.makedirs(odir, exist_ok=True)
AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2
FLUX_MIN = 1e-2  # mJy/arcsec^2 (mask, as plot_from_fits.py)

cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
sky = cfg.instantiate_sec(sky_sec)
bg = sky.background
freq = np.asarray(sky.freq, dtype="float64")
with open(f"{run}/last.pkl", "rb") as f:
    samples, *_ = pickle.load(f)

i0_k = np.stack([np.asarray(bg.ref_freq_model(x)) for x in samples]) * CONV_FACTOR
a_k = np.stack([np.asarray(bg.spectral_index(x)) for x in samples])
b_k = np.stack([np.asarray(bg.spectral_curvature(x)) for x in samples])
d_k = np.stack([np.asarray(bg.spectral_deviations(x)) for x in samples])  # (k, nfreq, N, N), log-space
print(f"samples {len(samples)} | i0 {i0_k.shape} | deviations {d_k.shape}")

t = TruthSky(truth_name)
u = np.log(freq / freq[REF_IDX])
t_cube = np.stack([t(f) for f in freq])
t_i0 = t_cube[REF_IDX] * CONV_FACTOR
t_alpha = np.tensordot(u, np.log(t_cube) - np.log(t_cube[REF_IDX]), axes=1) / np.sum(u**2)  # anchored slope (model alpha)
t_beta = t.beta  # exact u^2 coefficient

rec = dict(i0=i0_k.mean(0), alpha=a_k.mean(0), beta=b_k.mean(0))
std = dict(i0=i0_k.std(0, ddof=1), alpha=a_k.std(0, ddof=1), beta=b_k.std(0, ddof=1))
mask = rec["i0"] > FLUX_MIN
print(f"mask: {mask.sum()} pixels with I0 > {FLUX_MIN:g} mJy/arcsec^2")


def masked(a):
    return np.where(mask, a, np.nan)


def stat(d):
    d = d[mask]
    return f"median {np.median(d):+.3f}, MAD {1.4826 * np.median(np.abs(d - np.median(d))):.3f}"


# %%
# I0, alpha, beta: truth | reconstruction | rec - truth | posterior std (masked)
fig, axes = plt.subplots(3, 4, figsize=(21, 14.5))
rel_i0 = (rec["i0"] - t_i0) / t_i0
rows = [
    ("I0 [mJy/arcsec$^2$]", t_i0, rec["i0"], rel_i0, std["i0"] / rec["i0"],
     dict(cmap="inferno", norm=LogNorm(vmin=1e-2, vmax=10)), 0.2, "(rec - truth) / truth", "relative std"),
    ("alpha", t_alpha, rec["alpha"], rec["alpha"] - t_alpha, std["alpha"],
     dict(cmap="RdYlBu_r", vmin=-2.5, vmax=0.5), 0.2, "rec - truth", "posterior std"),
    ("beta", t_beta, rec["beta"], rec["beta"] - t_beta, std["beta"],
     dict(cmap="coolwarm", vmin=-1, vmax=1), 0.5, "rec - truth", "posterior std"),
]
for r, (name, tr, rc, d, sd, kw, dlim, dlab, slab) in enumerate(rows):
    log_label = ""
    for c, (arr, title, kwc) in enumerate([
        (tr, f"truth {name}{log_label}", kw),
        (rc, f"reconstruction {name}{log_label}", kw),
        (d, f"{dlab}: {stat(d)}", dict(cmap="PuOr", vmin=-dlim, vmax=dlim)),
        (sd, slab, dict(cmap="viridis", vmin=0, vmax=dlim)),
    ]):
        ax = axes[r, c]
        im = ax.imshow(masked(arr).T, origin="lower", **kwc)
        ax.contour(rec["i0"].T, levels=[FLUX_MIN, 0.1, 1], colors="k", linewidths=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"image-domain test ({os.path.basename(run)}): I0 > {FLUX_MIN:g} mJy/arcsec$^2$; contours 0.01, 0.1, 1")
fig.tight_layout()
fig.savefig(f"{odir}/image_test_maps.png", dpi=100)
plt.close(fig)
for name, d in [("I0 (relative)", rel_i0), ("alpha", rec["alpha"] - t_alpha), ("beta", rec["beta"] - t_beta)]:
    print(f"{name:14s} error in mask: {stat(d)}")

# %%
# deviations delta (log-space), zero in the truth: posterior mean per frequency, masked and over the whole field
d_mean = d_k.mean(0)
dlim = float(np.nanpercentile(np.abs(d_mean[:, mask]), 99))
fig, axes = plt.subplots(2, freq.size, figsize=(4 * freq.size, 8.2))
for i in range(freq.size):
    for r, (arr, tag) in enumerate([(masked(d_mean[i]), "mask"), (d_mean[i], "whole field")]):
        ax = axes[r, i]
        im = ax.imshow(arr.T, origin="lower", cmap="PuOr", vmin=-dlim, vmax=dlim)
        if r == 1:
            ax.contour(rec["i0"].T, levels=[FLUX_MIN], colors="k", linewidths=0.4)
        vals = d_mean[i][mask] if r == 0 else d_mean[i]
        ax.set_title(f"delta {freq[i] / 1e6:.0f} MHz, {tag}\nrms {np.sqrt(np.mean(vals**2)):.2e}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("spectral deviations delta (posterior mean, ln S units); truth: 0 everywhere")
fig.tight_layout()
fig.savefig(f"{odir}/image_test_deviations.png", dpi=100)
plt.close(fig)
rms_mask = np.sqrt(np.mean(d_mean[:, mask] ** 2, axis=1))
print("delta rms in the mask per frequency:", np.array2string(rms_mask, precision=4))
print("delta rms over the field per frequency:", np.array2string(np.sqrt(np.mean(d_mean**2, axis=(1, 2))), precision=4))
print("saved:", f"{odir}/image_test_maps.png", f"{odir}/image_test_deviations.png")

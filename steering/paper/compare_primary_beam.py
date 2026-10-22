# %%
# ---------------------------------------------------------------------------
# Compare the MeerKAT L-band primary beam from SARAO's `katbeam` package with
# the one used for the primary-beam correction (resolve.library.meerkat_beam,
# saved by save_primary_beam.py), at the 6 model frequencies on the full sky
# grid (2 deg, 3072 x 3072, pointing centre at the 0-based pixel N / 2).
# ---------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
from katbeam import JimBeam
from matplotlib.colors import LogNorm

dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"
mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"
odir = f"{dir}/opt/{mf_rec}/primary_beam"
os.makedirs(odir, exist_ok=True)

FREQ_HZ = np.array([992153320.3125, 1053594726.5625, 1114618164.0625,
                    1329454101.5625, 1398998058.6408134, 1468637695.3125])  # sky.6 freq
N, FOV_DEG = 3072, 2.0
PB_MIN = 0.1  # cutoff used in save_fits.py

pix_deg = FOV_DEG / N
x = (np.arange(N) - N / 2) * pix_deg  # offsets from the pointing centre [deg], axis 0 -> RA
xx, yy = np.meshgrid(x, x, indexing="ij")

pb_kat = np.stack([JimBeam("MKAT-AA-L-JIM-2020").I(xx, yy, f / 1e6) for f in FREQ_HZ])
pb_res = np.load(f"{dir}/opt/{mf_rec}/fits/eso137_primary_beam.npy").astype("float64")
assert pb_res.shape == pb_kat.shape
np.save(f"{odir}/eso137_primary_beam_katbeam.npy", pb_kat.astype("float32"))

diff = pb_res - pb_kat
for f, d, k in zip(FREQ_HZ, diff, pb_kat):
    fwhm = 2 * x[N // 2:][np.argmax(k[N // 2:, N // 2] < 0.5)] * 60
    print(f"{f / 1e6:7.1f} MHz | FWHM (katbeam, RA cut) {fwhm:5.1f} arcmin | "
          f"max |resolve - katbeam| {np.abs(d).max():.2e}")

ext = [x[0] + pix_deg / 2, x[-1] + pix_deg / 2] * 2  # image extent [deg]


def beam_grid(cube, title, name):
    """2 x 3 panels of the beam, log colour scale with PB = 0.5 and PB_MIN contours."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, f, pb in zip(axs.flat, FREQ_HZ, cube):
        im = ax.imshow(pb.T, origin="lower", extent=ext, cmap="viridis", norm=LogNorm(1e-3, 1))
        cs = ax.contour(x, x, pb.T, levels=[PB_MIN, 0.5], colors=["white", "black"], linewidths=1)
        ax.clabel(cs, fmt={PB_MIN: f"{PB_MIN}", 0.5: "0.5"}, fontsize=8)
        ax.set_title(f"{f / 1e6:.0f} MHz")
        ax.invert_xaxis()  # RA increases to the left
        ax.set_aspect("equal")
    for ax in axs[-1]:
        ax.set_xlabel(r"$\Delta$RA [deg]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\Delta$Dec [deg]")
    fig.colorbar(im, ax=axs, shrink=0.8, label="primary beam (Stokes I)")
    fig.suptitle(title)
    fig.savefig(f"{odir}/{name}.png", dpi=150)
    print("saved:", f"{odir}/{name}.png")
    plt.close(fig)


beam_grid(pb_kat, "katbeam (SARAO) - MKAT-AA-L-JIM-2020", "pb_katbeam")
beam_grid(pb_res, "resolve.library.meerkat_beam - used for the PB correction", "pb_resolve")

# %%
# Comparison: radial cuts along RA (katbeam solid, resolve dashed) and the
# absolute difference map at each frequency.
colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(FREQ_HZ)))
fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
gs = fig.add_gridspec(2, 5, width_ratios=[2.2, 2.2, 1, 1, 1])
ax_lin, ax_log = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
r = x[N // 2:]
for f, c, k, p in zip(FREQ_HZ, colors, pb_kat, pb_res):
    for ax in (ax_lin, ax_log):
        ax.plot(r, k[N // 2:, N // 2], color=c, lw=2, label=f"{f / 1e6:.0f} MHz")
        ax.plot(r, p[N // 2:, N // 2], color="black", lw=1, ls="--")
for ax in (ax_lin, ax_log):
    ax.axhline(PB_MIN, color="0.6", lw=1, ls=":")
    ax.set_xlabel("offset from pointing centre [deg]")
    ax.grid(alpha=0.3)
ax_lin.set_ylabel("primary beam")
ax_log.set_yscale("log")
ax_log.set_ylim(1e-4, 1.2)
ax_lin.plot([], [], color="black", lw=1, ls="--", label="resolve (dashed)")
ax_lin.legend(title="katbeam (solid)", fontsize=8)
ax_lin.set_title("RA cut through the pointing centre")
ax_log.set_title("same, log scale")
for i, (f, d) in enumerate(zip(FREQ_HZ, diff)):
    ax = fig.add_subplot(gs[i % 2, 2 + i // 2])
    vmax = max(np.abs(d).max(), 1e-12)
    im = ax.imshow(d.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{f / 1e6:.0f} MHz\nmax |diff| = {np.abs(d).max():.1e}", fontsize=9)
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("resolve - katbeam")
fig.savefig(f"{odir}/pb_comparison.png", dpi=150)
print("saved:", f"{odir}/pb_comparison.png")

# %%
# ---------------------------------------------------------------------------
# Low-resolution truth with a beta field of another correlation length than alpha: brightness and alpha of the
# 1 deg truth "main" (one spectrum per pixel), but beta a new Gaussian random field with the same mean and
# standard deviation, smoothed with BETA_SMOOTH_ARCSEC instead of SMOOTH_ARCSEC. The model has one spatial
# power spectrum for alpha and beta, so this tests whether that limits beta. Averaged over 3 x 3 fine pixels and
# refitted as make_lowres_1deg.py.
#
# usage: python make_truth_bcl.py   Output: truth/main_bcl_lr/truth.npz (keys of main_lr)
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import BETA_MEAN, BETA_STD, FAC, SEED_TRUTH, truth_dir

BETA_SMOOTH_ARCSEC = 150.0
PIX_AS = 7.03125 / FAC
t = np.load(f"{truth_dir('1deg_main')}/truth.npz")
nu_ref = float(t["nu_ref"])
N = t["i0"].shape[0]
u = np.log(np.geomspace(950e6, 1520e6, 24) / nu_ref)


def block(a):
    n = a.shape[-1] // FAC
    return a.reshape(a.shape[:-2] + (n, FAC, n, FAC)).mean(axis=(-3, -1))


rng = np.random.default_rng([SEED_TRUTH, 200])
f = gaussian_filter(rng.standard_normal((N, N)), BETA_SMOOTH_ARCSEC / PIX_AS, mode="wrap")
beta_fine = BETA_MEAN + BETA_STD * (f - f.mean()) / f.std()
cube = block(t["i0"][None] * np.exp(t["alpha"][None] * u[:, None, None] + beta_fine[None] * u[:, None, None] ** 2))
b, a, _ = np.polyfit(u, np.log(np.clip(cube, 1e-300, None)).reshape(u.size, -1), 2)
out = dict(nu_ref=nu_ref, nu_eff=t["nu_eff"], source=truth_dir("1deg_main"), factor=FAC, alpha=a.reshape(cube.shape[1:]), beta=b.reshape(cube.shape[1:]))
out |= {f"i0_{c}": block(t[f"i0_{c}"]) for c in ("bg", "o0", "o1")}
out["i0"] = block(t["i0"])
tdir = truth_dir("1deg_main_bcl_lr")
os.makedirs(tdir, exist_ok=True)
np.savez(f"{tdir}/truth.npz", **out)
print(f"saved {tdir}/truth.npz: beta {np.median(out['beta']):+.2f} +- {out['beta'].std():.2f}, smoothed {BETA_SMOOTH_ARCSEC:g} arcsec")

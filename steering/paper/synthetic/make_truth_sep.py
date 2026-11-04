# %%
# ---------------------------------------------------------------------------
# Low-resolution truth with a spectrum per component: the brightness of the 1 deg truth "main", but
#     S(x, nu) = sum_k I0_k(x) exp(alpha_k(x) u + beta_k(x) u^2),   k = bg, o0, o1
# O1 keeps the alpha / beta fields of "main" (comparable to the earlier runs), bg and o0 get new independent
# fields with the same statistics. Every component is averaged over 3 x 3 fine pixels at 24 frequencies over
# the band and refitted (as make_lowres_1deg.py); alpha / beta are the quadratic fit of the total spectrum.
#
# usage: python make_truth_sep.py   Output: truth/main_sep_lr/truth.npz (keys of main_lr + alpha_k, beta_k)
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import ALPHA_MEAN, ALPHA_STD, BETA_MEAN, BETA_STD, FAC, SEED_TRUTH, SMOOTH_ARCSEC, truth_dir

PIX_AS = 7.03125 / FAC
t = np.load(f"{truth_dir('1deg_main')}/truth.npz")
nu_ref = float(t["nu_ref"])
N = t["i0"].shape[0]
nus = np.geomspace(950e6, 1520e6, 24)
u = np.log(nus / nu_ref)


def grf(rng, mean, std):
    f = gaussian_filter(rng.standard_normal((N, N)), SMOOTH_ARCSEC / PIX_AS, mode="wrap")
    return mean + std * (f - f.mean()) / f.std()


def block(a):
    n = a.shape[-1] // FAC
    return a.reshape(a.shape[:-2] + (n, FAC, n, FAC)).mean(axis=(-3, -1))


def fit(cube):
    ln = np.log(np.clip(cube, 1e-300, None)).reshape(u.size, -1)
    beta, alpha, _ = np.polyfit(u, ln, 2)
    return alpha.reshape(cube.shape[1:]), beta.reshape(cube.shape[1:])


fields = {"o1": (t["alpha"], t["beta"])}
for i, c in enumerate(("bg", "o0")):
    rng = np.random.default_rng([SEED_TRUTH, 100 + i])
    fields[c] = (grf(rng, ALPHA_MEAN, ALPHA_STD), grf(rng, BETA_MEAN, BETA_STD))

out = dict(nu_ref=nu_ref, nu_eff=t["nu_eff"], source=truth_dir("1deg_main"), factor=FAC)
total = 0
for c, (a, b) in fields.items():
    cube = block(t[f"i0_{c}"][None] * np.exp(a[None] * u[:, None, None] + b[None] * u[:, None, None] ** 2))
    total = total + cube
    alpha, beta = fit(cube)
    zero = block(t[f"i0_{c}"]) <= 0  # no emission: the fit is meaningless
    out |= {f"i0_{c}": block(t[f"i0_{c}"]), f"alpha_{c}": np.where(zero, 0.0, alpha), f"beta_{c}": np.where(zero, 0.0, beta)}
    print(f"{c}: {np.sum(~zero)} pixels with emission, alpha {np.median(alpha[~zero]):+.2f} +- {alpha[~zero].std():.2f}, "
          f"beta {np.median(beta[~zero]):+.2f} +- {beta[~zero].std():.2f}")
out["i0"] = sum(out[f"i0_{c}"] for c in fields)
out["alpha"], out["beta"] = fit(total)
tdir = truth_dir("1deg_main_sep_lr")
os.makedirs(tdir, exist_ok=True)
np.savez(f"{tdir}/truth.npz", **out)
print(f"saved {tdir}/truth.npz; the O1 mixing check: median |beta_total - beta_o1| where O1 dominates "
      f"{np.median(np.abs(out['beta'] - out['beta_o1'])[out['i0_o1'] > out['i0_bg'] + out['i0_o0']]):.3f}")

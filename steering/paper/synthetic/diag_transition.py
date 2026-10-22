# %%
# ---------------------------------------------------------------------------
# Diagnostic of the single- -> multi-frequency transition (mode: freq): does the
# multi-frequency model at the transitioned position reproduce the sky of the
# single-frequency stage? Uses the configs of image_curv/A_img_sfmf and the
# single-frequency state of image_curv/diag_transition (same stage re-run).
#
# usage: python diag_transition.py
# Output: OUT/image_curv/diag_transition/transition_check.png and printed numbers
# ---------------------------------------------------------------------------
import os
import pickle
import sys

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import OUT, REF_IDX, STEERING, TruthSky

import aim_resolve as aim
from aim_resolve.optimize.samples import domain_tree
from aim_resolve.transition import transition_freq

root = f"{OUT}/image_curv"
cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{root}/A_img_sfmf/config.yml"),
                                     aim.get_builders)
lh0, lh1 = cfg.instantiate_sec("lh.0"), cfg.instantiate_sec("lh.1")
sky0, sky1 = lh0["sky_model"], lh1["sky_model"]
with open(f"{root}/diag_transition/last.pkl", "rb") as f:
    samples_sf, *_ = pickle.load(f)

new = transition_freq(key=jax.random.PRNGKey(0), samples=samples_sf, it=1, lh_old=lh0, lh_new=lh1)
pos_sf, pos_mf = samples_sf.pos, new.pos

s_sf = np.asarray(sky0(pos_sf), dtype="float64")  # single-frequency sky at its position (N, N)
s_mf = np.asarray(sky1(pos_mf), dtype="float64")  # multi-frequency sky at the transitioned position (nfreq, N, N)
ref_sf = np.asarray(sky1.background.ref_freq_model(pos_mf), dtype="float64")
print("single-frequency sky", s_sf.shape, "| multi-frequency sky", s_mf.shape)

# 1) the same parameters, the same sky? (reference frequency)
ratio = s_mf[REF_IDX] / s_sf
bright = s_sf > np.percentile(s_sf, 90)
print(f"MF(nu_ref) / SF: median {np.median(ratio):.4f}, 5-95 % {np.percentile(ratio, 5):.4f}-{np.percentile(ratio, 95):.4f}, "
      f"brightest 10 %: median {np.median(ratio[bright]):.4f}")
print(f"MF ref_freq_model / SF: median {np.median(ref_sf / s_sf):.4f}")

# 2) which parameters were copied, which initialised anew
k_sf, k_mf = set(domain_tree(pos_sf)), set(domain_tree(pos_mf))
copied = sorted(k for k in k_mf if k.replace("bg.1", "bg.0") in k_sf)
fresh = sorted(k_mf - set(copied))
print("copied:", copied)
print("initialised anew:", fresh)
for k in copied:
    a, b = np.asarray(domain_tree(pos_sf)[k.replace("bg.1", "bg.0")]), np.asarray(domain_tree(pos_mf)[k])
    print(f"  {k:28s} identical: {np.array_equal(a, b)}  shape {a.shape}")

# 3) spectral parameters and the fit to the multi-frequency data at the start
alpha = np.asarray(sky1.background.spectral_index(pos_mf))
beta = np.asarray(sky1.background.spectral_curvature(pos_mf))
print(f"alpha at the start: {alpha.mean():+.3f} +- {alpha.std():.3f} | beta: {beta.mean():+.3f} +- {beta.std():.3f}")
sig0, sig1 = 1 / lh0["noise_std_inv"], 1 / lh1["noise_std_inv"]
d0, d1 = np.asarray(lh0["data"]), np.asarray(lh1["data"])
chi_sf = np.mean((d0 - s_sf) ** 2) / sig0**2
chi_mf = np.mean((d1 - s_mf) ** 2, axis=(1, 2)) / sig1**2
print(f"reduced chi^2: SF end {chi_sf:.2f} | MF start per frequency {np.array2string(chi_mf, precision=1)}")
t = TruthSky("main")
freq = np.asarray(sky1.freq, dtype="float64")
truth_spec_ratio = np.stack([t(f) for f in freq]) / t(freq[REF_IDX])
chi_flat = np.mean((d1 - s_sf[None] * 1.0) ** 2, axis=(1, 2)) / sig1**2
print(f"reduced chi^2 if the SF sky is used at every frequency (flat spectrum): {np.array2string(chi_flat, precision=1)}")

fig, axes = plt.subplots(1, 4, figsize=(21, 5))
vmax = np.percentile(s_sf, 99.9)
for ax, (arr, title, kw) in zip(axes, [
    (np.log10(np.clip(s_sf, vmax * 1e-4, None)), "single-frequency sky (end of stage)", dict(cmap="inferno")),
    (np.log10(np.clip(s_mf[REF_IDX], vmax * 1e-4, None)), "multi-frequency sky at nu_ref after the transition", dict(cmap="inferno")),
    (np.log10(ratio), "log10 MF(nu_ref) / SF", dict(cmap="PuOr", vmin=-1, vmax=1)),
    (alpha, "alpha at the start of the MF stage", dict(cmap="RdYlBu_r", vmin=-2.5, vmax=0.5)),
]):
    im = ax.imshow(arr.T, origin="lower", **kw)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(f"{root}/diag_transition/transition_check.png", dpi=100)
print("saved:", f"{root}/diag_transition/transition_check.png")

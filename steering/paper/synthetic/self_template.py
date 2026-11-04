# %%
# ---------------------------------------------------------------------------
# Self-template fit: does the data agree with a reconstruction? Needs no truth, so it also works on real runs.
# The posterior mean of every model component is cut into radial rings; each ring is a template T_j with a fixed
# shape. Per model bin k the visibilities are fitted with one free amplitude per template,
#     d_c = sum_j a_jk R_c(T_j(nu_c) PB_c) + noise,
# with the exact response per channel (bin image extrapolated with its local spectral slope, the channel's beam;
# as ChannelResidual). a = 1: the data want this flux. a = 0.9: 10 % less. The curvature of ln a over the bins is
# the beta offset the data ask for in that region.
#
# The error of a is sqrt(diag A^-1 chi^2), A the normal matrix of the bin, chi^2 the reduced chi^2 of the fit.
# A flag needs both a significant deviation (> NSIG sigma) and a relevant one (> MIN_DEV): with a prior, faint
# regions may legitimately sit slightly low. Templates in the same place share flux (see the correlations printed
# per component pair): a wrong split between two components within a ring is not seen, only the sum.
#
# usage: python self_template.py RUN_DIR [--stage 1] [--nthreads 32]
# Output: RUN_DIR/compare/self_template.npz and printed tables
# ---------------------------------------------------------------------------
import argparse
import os
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import STEERING

import nifty8 as ift

import aim_resolve as aim
from aim_resolve.fast_resolve.beam import beam_offsets, build_primary_beam
from aim_resolve.fast_resolve.response import ChannelResidual
from aim_resolve.resolve.observation import Observation

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--stage", default="1", help="stage of the sky, likelihood and data sections")
parser.add_argument("--nthreads", type=int, default=32)
args = parser.parse_args()
run = os.path.abspath(args.run)
os.chdir(STEERING)
EDGES = [0, 0.1, 0.2, 0.3, 0.35, 1.0]  # rings [deg]
MIN_SHARE = 1e-3  # rings with less of the component's flux are left out
NSIG, MIN_DEV = 5.0, 0.02

# %%
cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
sky = cfg.instantiate_sec(f"sky.{args.stage}")
sec = cfg.sections
freq = np.asarray(sky.freq, dtype="float64")
bounds = sec[f"lh.{args.stage}"].get("freq_bounds")
obs = Observation.load_npz(sec[f"data.{args.stage}"]["fname"]).to_resolve_obs().to_double_precision()
with open(f"{run}/last.pkl", "rb") as f:
    samples, _, mj = pickle.load(f)
print(f"{os.path.basename(run)}: major iteration {mj}, {len(samples)} samples, {freq.size} bins, {obs.freq.size} channels")

xx, yy = beam_offsets(sky.grid)
r = np.hypot(xx, yy)
templates, names = [], []
for m in sky.models:
    c = m.prefix.split(".")[0]
    cube = np.mean([np.asarray(m(x, map=True), dtype="float64") for x in samples], 0)
    slope = np.gradient(np.log(np.clip(cube, 1e-300, None)), np.log(freq), axis=0)
    tot = cube[0].sum()
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        ring = (r >= lo) & (r < hi)
        if cube[0][ring].sum() >= MIN_SHARE * tot:
            templates.append((cube, slope, ring))
            names.append(f"{c} r[{lo:g},{hi:g})")
print("templates:", names, flush=True)

# %%
# normal equations per bin (exact response per channel)
ch = ChannelResidual(obs, sky.grid, freq, None, bounds, nthreads=args.nthreads)
nu, kb = ch.nu, ch.chan_bin
K, J = freq.size, len(names)
A, b, dd, nvis = np.zeros((K, J, J)), np.zeros((K, J)), np.zeros(K), np.zeros(K)
v_all, w_all = np.asarray(obs.vis.val)[0], np.asarray(obs.weight.val)[0]
t0 = time.time()
for c in range(nu.size):
    k = kb[c]
    R = ch._response(c)
    pb = np.asarray(build_primary_beam(sky.grid, [nu[c]], channel_average=False), dtype="float64")
    P = []
    for cube, slope, ring in templates:
        s = cube[k] * (nu[c] / freq[k]) ** slope[k] * pb
        P.append(np.asarray(R(ift.makeField(R.domain, np.where(ring, s, 0.0))).val).reshape(-1))
    P = np.stack(P)
    w, v = w_all[:, c], v_all[:, c]
    A[k] += np.real((P.conj() * w) @ P.T)
    b[k] += np.real((P.conj() * w) @ v)
    dd[k] += np.sum(w * np.abs(v) ** 2)
    nvis[k] += 2 * np.sum(w > 0)
    if c % 100 == 0:
        print(f"channel {c}/{nu.size} {time.time() - t0:.0f} s", flush=True)

# %%
a, ea, corr = np.zeros((K, J)), np.zeros((K, J)), []
u = np.log(freq / freq[freq.size // 2])
for k in range(K):
    a[k] = np.linalg.solve(A[k], b[k])
    chi2 = (dd[k] - 2 * a[k] @ b[k] + a[k] @ A[k] @ a[k]) / (nvis[k] - J)
    inv = np.linalg.inv(A[k])
    ea[k] = np.sqrt(np.diag(inv) * chi2)
    corr.append(inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv))))
corr = np.mean(corr, 0)

print(f"\namplitudes a per bin [MHz], with the median error; d_beta / d_alpha: curvature / slope of ln a; * flagged")
print(f"{'template':18s} " + " ".join(f"{f / 1e6:6.0f}" for f in freq) + f" {'err':>7s} {'d_alpha':>8s} {'d_beta':>7s} {'+-':>6s}")
for j, n in enumerate(names):
    wfit = a[:, j] / ea[:, j]
    cf, cov = np.polyfit(u, np.log(np.clip(a[:, j], 1e-3, None)), 2, w=wfit, cov="unscaled")
    flag = "*" if np.any((np.abs(a[:, j] - 1) > NSIG * ea[:, j]) & (np.abs(a[:, j] - 1) > MIN_DEV)) else ""
    print(f"{n:18s} " + " ".join(f"{v:6.3f}" for v in a[:, j]) + f" {np.median(ea[:, j]):7.4f} {cf[1]:+8.3f} {cf[0]:+7.3f} {np.sqrt(cov[0, 0]):6.3f} {flag}")
pairs = [(names[i], names[j], corr[i, j]) for i in range(J) for j in range(i) if abs(corr[i, j]) > 0.5]
print("\nstrongly coupled templates (the data see only their combination):", [(p, q, round(float(c), 2)) for p, q, c in pairs] or "none")
os.makedirs(f"{run}/compare", exist_ok=True)
np.savez(f"{run}/compare/self_template.npz", a=a, ea=ea, names=names, freq=freq, corr=corr, A=A, b=b)
print("saved:", f"{run}/compare/self_template.npz")

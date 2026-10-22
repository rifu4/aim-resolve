# %%
# ---------------------------------------------------------------------------
# Phase 2 of the synthetic-data test (../synthetic_test_plan.md): synthetic
# visibilities with the real uvw, weights, flags and channels.
#
# For every data channel c the apparent sky S(x, nu_c) * PB(x, nu_c) is
# degridded with the same resolve operator as in the inference
# (build_exact_responses: InterferometryResponse, w-gridding, epsilon 1e-9,
# double precision), on the 1 deg field grid. Variants (synth_common.VARIANTS):
#   A  truth "main",  each channel at its exact frequency, beam per channel
#   C  truth "beta0", as A
#   B  truth "main",  S and beam at the model frequency of the channel's bin
#      (bin-averaged beam of build_primary_beam with freq_bounds), i.e. data
#      exactly as the multi-frequency model assumes
# Noise: Re and Im ~ N(0, (s_c / sqrt(w))^2), with s_c the robust noise level
# of the real data in channel c (longest 10 % baselines), one noise seed for
# all variants unless synth_common.NOISE_SEEDS sets another one (Ap). Flags (w = 0) stay flags (vis = 0).
#
# Validation (--check): runs on a channel subset only and compares
#   V1 the per-channel prediction (variant B, noiseless) with the inference's
#      own multi-frequency operator (freq_bounds) -> conventions, beam, units
#   V2 dirty images per bin of real vs synthetic data (A) -> realism, orientation
#   V3 noise level of synthetic vs real data on long baselines
#
# usage: python make_vis.py --data mf|sf [--variants A,C] [--check] [--nthreads 48]
# ---------------------------------------------------------------------------
import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import DATA_MF, DATA_SF, OUT, SYNTH_DATA, VARIANTS, TruthSky, field_grid, noise_seed

import nifty8 as ift
import resolve as rve

from aim_resolve.fast_resolve.beam import build_primary_beam
from aim_resolve.fast_resolve.response import build_exact_responses
from aim_resolve.resolve.observation import Observation

C_LIGHT = 299792458.0
EPSILON = 1e-9  # as build_exact_responses

parser = argparse.ArgumentParser()
parser.add_argument("--data", choices=["mf", "sf"], required=True)
parser.add_argument("--variants", default="A,C")
parser.add_argument("--check", action="store_true", help="validation on a channel subset only, nothing is saved")
parser.add_argument("--nthreads", type=int, default=48)
args = parser.parse_args()
variants = args.variants.split(",")
fname = DATA_MF if args.data == "mf" else DATA_SF
odir = f"{OUT}/vis_check" if args.check else SYNTH_DATA
os.makedirs(odir, exist_ok=True)

prep = np.load(f"{OUT}/prep/prep.npz")
nu_eff, bounds = prep["nu_eff"], prep["bounds"]
grid = field_grid()
sdom = ift.RGSpace(grid.shape, distances=grid.dis / grid.fac)

# %%
# ---------------------------------------------------------------------------
# Real data, loaded as in the pipeline (aim Observation -> resolve, double).
# ---------------------------------------------------------------------------
t0 = time.time()
obs = Observation.load_npz(fname).to_resolve_obs().to_double_precision()
nu_data = np.asarray(obs.freq, dtype="float64")
n_chan = nu_data.size
chans = np.arange(n_chan)
if args.check:
    chans = np.unique(np.linspace(0, n_chan - 1, 13).round().astype(int)) if n_chan > 13 else chans
    obs = obs.get_freqs_by_slice(chans) if hasattr(obs, "get_freqs_by_slice") else obs
print(f"data {os.path.basename(fname)}: {obs.vis.shape} (pol, row, chan) | {len(chans)} channels used "
      f"| loaded in {time.time() - t0:.0f} s")
vis_real = np.asarray(obs.vis.val)[0]  # (row, chan)
wgt = np.asarray(obs.weight.val)[0]
uvlen = np.hypot(obs.uvw[:, 0], obs.uvw[:, 1])
nu_used = np.asarray(obs.freq, dtype="float64")


def robust_noise_level(c):
    """Noise per real / imaginary part in units of 1 / sqrt(w), longest 10 % baselines."""
    lam = uvlen * nu_used[c] / C_LIGHT
    m = (lam > np.percentile(lam, 90)) & (wgt[:, c] > 0)
    z = vis_real[m, c] * np.sqrt(wgt[m, c])
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))  # noqa: E731
    return 0.5 * (mad(z.real) + mad(z.imag))


noise_level = np.array([robust_noise_level(c) for c in range(nu_used.size)])
print(f"real-data noise level s_c (x 1/sqrt(w)): {noise_level.min():.3f}-{noise_level.max():.3f}, "
      f"median {np.median(noise_level):.3f}")

# %%
# ---------------------------------------------------------------------------
# Per-channel prediction.
# ---------------------------------------------------------------------------
truth = {v: TruthSky(VARIANTS[v][0]) for v in variants}
binned = {v: VARIANTS[v][1] == "binned" for v in variants}
if args.data == "sf" and any(binned.values()):
    raise ValueError("the binned variant only applies to the multi-frequency data")
if any(binned.values()) or args.check:
    chan_bin = np.searchsorted(bounds, nu_used, side="right") - 1
    pb_bins = build_primary_beam(grid, nu_eff, obs, freq_bounds=bounds)  # the model's beam (bin-averaged)


def channel_response(c):
    oc = obs.get_freqs_by_slice(slice(c, c + 1))
    sky_dom = rve.default_sky_domain(sdom=sdom, fdom=rve.IRGSpace(oc.freq))
    R = rve.InterferometryResponse(oc, sky_dom, True, EPSILON, verbosity=0, nthreads=args.nthreads)
    return R @ ift.DomainChangerAndReshaper(R.domain[3], R.domain)


def predict(R, sky):
    return np.asarray(R(ift.makeField(R.domain, sky)).val).reshape(-1)


n_row = vis_real.shape[0]
vis_out = {v: np.zeros((n_row, nu_used.size), dtype=np.complex128) for v in variants}
vis_clean = {v: np.zeros((n_row, nu_used.size), dtype=np.complex128) for v in variants} if args.check else None
check_B = np.zeros((n_row, nu_used.size), dtype=np.complex128) if args.check else None
t0 = time.time()
for c in range(nu_used.size):
    R = channel_response(c)
    pb_c = build_primary_beam(grid, [nu_used[c]], channel_average=False)
    sig = np.where(wgt[:, c] > 0, noise_level[c] / np.sqrt(np.where(wgt[:, c] > 0, wgt[:, c], 1.0)), 0.0)
    noise = {}  # one noise realisation per seed: variants with the same seed share it
    for s in {noise_seed(v) for v in variants}:
        rng = np.random.default_rng([s, int(round(nu_used[c]))])
        noise[s] = sig * (rng.standard_normal(n_row) + 1j * rng.standard_normal(n_row))
    for v in variants:
        if binned[v]:
            k = chan_bin[c]
            sky = truth[v](nu_eff[k]) * pb_bins[k]
        else:
            sky = truth[v](nu_used[c]) * pb_c
        clean = predict(R, sky)
        vis_out[v][:, c] = np.where(wgt[:, c] > 0, clean + noise[noise_seed(v)], 0.0)
        if args.check:
            vis_clean[v][:, c] = clean
    if args.check:
        k = chan_bin[c]
        check_B[:, c] = predict(R, TruthSky("main")(nu_eff[k]) * pb_bins[k])
    if c % 20 == 0 or c == nu_used.size - 1:
        el = time.time() - t0
        print(f"  channel {c + 1}/{nu_used.size} ({nu_used[c] / 1e6:.1f} MHz) | {el / 60:.1f} min, "
              f"eta {el / (c + 1) * (nu_used.size - c - 1) / 60:.1f} min", flush=True)

# %%
# ---------------------------------------------------------------------------
# Save (full run) or validate (--check).
# ---------------------------------------------------------------------------
if not args.check:
    src = np.load(fname, allow_pickle=True)
    keys = [k for k in src.files if k != "vis"]
    base = os.path.basename(fname)[:-4]
    for v in variants:
        fn = f"{odir}/{base}_synth_{v}_s{noise_seed(v)}.npz"
        t0 = time.time()
        vis = vis_out[v][None].astype(np.complex64)  # (pol, row, chan) as the original
        np.savez(fn, vis=vis, **{k: src[k] for k in keys})
        print(f"saved: {fn} ({os.path.getsize(fn) / 1e9:.1f} GB, {time.time() - t0:.0f} s)")
    np.savez(f"{odir}/{base}_synth_noise_level.npz", freq=nu_used, noise_level=noise_level)
    sys.exit(0)

if args.data == "mf":
    print("\n=== V1: per-channel prediction vs the inference operator (variant B, noiseless) ===")
    R_inf = build_exact_responses(obs, grid, nu_eff, bounds)[0]
    model_B = np.stack([TruthSky("main")(f) for f in nu_eff]) * pb_bins
    vis_inf = np.asarray(R_inf(ift.makeField(R_inf.domain, model_B)).val)[0]
    rel = np.linalg.norm(vis_inf - check_B) / np.linalg.norm(vis_inf)
    print(f"relative difference |V_inference - V_per_channel| / |V_inference| = {rel:.2e}  (expect <~1e-6)")
    per_chan = np.linalg.norm(vis_inf - check_B, axis=0) / np.linalg.norm(vis_inf, axis=0)
    print("per channel:", np.array2string(per_chan, precision=1))
    show_freq = nu_eff
else:  # single-frequency stage: one model frequency, all channels
    R_inf = build_exact_responses(obs, grid)[0]
    show_freq = np.array([np.sum(nu_used * wgt.sum(axis=0)) / wgt.sum()])

print("\n=== V2: dirty images per bin, real vs synthetic (A) ===")
N_inv = ift.makeOp(obs.weight)


def dirty(vis):
    d = ift.makeField(obs.vis.domain, vis[None].astype(np.complex128))
    return np.asarray(R_inf.adjoint(N_inv(d)).val).reshape((-1,) + grid.shape)


v_show = "A" if "A" in variants else variants[0]
d_real, d_syn = dirty(vis_real), dirty(vis_out[v_show])
fig, axes = plt.subplots(2, show_freq.size, figsize=(4 * show_freq.size, 8), squeeze=False)
for k in range(show_freq.size):
    r, s = d_real[k], d_syn[k]
    cc = np.corrcoef(r.ravel(), s.ravel())[0, 1]
    pk_r, pk_s = np.unravel_index(np.argmax(r), r.shape), np.unravel_index(np.argmax(s), s.shape)
    print(f"bin {k} ({show_freq[k] / 1e6:.0f} MHz): corr {cc:.3f} | sum syn/real {s.sum() / r.sum():.3f} | "
          f"peak real {pk_r} syn {pk_s} | max syn/real {s.max() / r.max():.3f}")
    vmax = np.percentile(r, 99.9)
    for ax, a, t in [(axes[0, k], r, "real"), (axes[1, k], s, f"synthetic {v_show}")]:
        ax.imshow(a.T, origin="lower", cmap="inferno", vmin=-0.05 * vmax, vmax=vmax)
        ax.set_title(f"{t}, {show_freq[k] / 1e6:.0f} MHz", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
fig.savefig(f"{odir}/dirty_real_vs_synth_{args.data}.png", dpi=110)
print("saved:", f"{odir}/dirty_real_vs_synth_{args.data}.png")

print("\n=== V3: noise level on long baselines, real vs synthetic ===")
for c in range(0, nu_used.size, max(1, nu_used.size // 5)):
    lam = uvlen * nu_used[c] / C_LIGHT
    m = (lam > np.percentile(lam, 90)) & (wgt[:, c] > 0)
    for v in variants:
        z = (vis_out[v][m, c] - vis_clean[v][m, c]) * np.sqrt(wgt[m, c])
        zs = vis_out[v][m, c] * np.sqrt(wgt[m, c])
        print(f"  {nu_used[c] / 1e6:7.1f} MHz {v}: noise std (Re,Im) {z.real.std():.3f},{z.imag.std():.3f} | "
              f"robust std of data {1.4826 * np.median(np.abs(zs.real - np.median(zs.real))):.3f} "
              f"(real data {noise_level[c]:.3f})")

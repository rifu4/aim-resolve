# %%
# ---------------------------------------------------------------------------
# Is the beta error of a 1 deg run (O1 in A1deg: beta ~0.3 too negative) caused by the
# primary beam? Variant A's data have the exact beam and sky at every channel, the model
# has one bin-averaged beam and the sky at the effective frequency per bin. Per pixel,
# the apparent bin value the model sees is
#     r_k = <S(nu) PB(nu)>_k / <PB>_k    (channel-weighted averages over bin k),
# and the curvature of a quadratic fit of ln r_k in u_k = ln(nu_eff_k / nu_ref) minus the
# truth beta is the predicted beta error ("beam + sky binning"); with PB = 1 the same gives
# the sky binning alone. Both are compared with the measured beta error of the run
# (compare/posterior.npz of compare_synthetic.py) per component (dominant truth component)
# and in radial bins.
#
# usage: python check_beta_radius.py --run DIR --truth 1deg_main [--variant A1deg]
# Output: DIR/compare/beta_radius.png and printed tables
# ---------------------------------------------------------------------------
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import STEERING, SYNTH_DATA, noise_seed, truth_dir

import aim_resolve as aim
from aim_resolve.fast_resolve.beam import beam_offsets, default_beam_model
from resolve.library.meerkat_beam import JimBeam

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--truth", required=True)
parser.add_argument("--variant", default=None)
parser.add_argument("--n-chan", type=int, default=16, help="channels per bin used for the averages")
args = parser.parse_args()
run = os.path.abspath(args.run)
variant = args.variant or os.path.basename(run)
AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2
FLUX_MIN = 1e-2

# %%
# grid, model frequencies and bins, channel frequencies and weights
cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
sky = cfg.instantiate_sec("sky.1")
mf8 = {k: v for d in yaml.safe_load_all(open(f"{STEERING}/nifty/config/mf_ext_8f.yml")) if d for k, v in d.items()}
nu_eff = np.asarray(mf8["freq"], dtype="float64")
bounds = np.asarray(mf8["lh.6"]["freq_bounds"], dtype="float64")
d = np.load(f"{SYNTH_DATA}/eso_961-1503mhz_rows10pct_w_synth_{variant}_s{noise_seed(variant)}.npz", allow_pickle=True)
ch_freq = np.asarray(d["freq"], dtype="float64")
ch_wgt = np.asarray(d["weight"], dtype="float64").reshape(-1, ch_freq.size).sum(axis=0)
print(f"{ch_freq.size} channels, {ch_freq.min() / 1e6:.0f}-{ch_freq.max() / 1e6:.0f} MHz, bins {np.round(bounds / 1e6)}")

# pixels: reconstructed flux above FLUX_MIN, labelled by the dominant truth component
post = np.load(f"{run}/compare/posterior.npz")
t = np.load(f"{truth_dir(args.truth)}/truth.npz")
m = post["rec_flux"] > FLUX_MIN
comps = ("bg", "o0", "o1")
dom = np.argmax(np.stack([t[f"i0_{c}"] for c in comps]), axis=0)[m]
xx, yy = beam_offsets(sky.grid)
r = np.hypot(xx, yy)[m]
i0, a, b = t["i0"][m], t["alpha"][m], t["beta"][m]
nu_ref = float(t["nu_ref"])
beam = JimBeam(default_beam_model(ch_freq))
print(f"{m.sum()} pixels with rec flux > {FLUX_MIN:g} mJy/arcsec^2, r up to {r.max():.2f} deg")

# %%
# per bin: channel-weighted averages of S PB and PB at the pixels (n_chan channels per bin)
num, den, num1 = (np.zeros((nu_eff.size, m.sum())) for _ in range(3))
for k in range(nu_eff.size):
    upper = ch_freq <= bounds[k + 1] if k == nu_eff.size - 1 else ch_freq < bounds[k + 1]
    idx = np.nonzero((ch_freq >= bounds[k]) & upper & (ch_wgt > 0))[0]
    idx = idx[np.unique(np.linspace(0, idx.size - 1, min(args.n_chan, idx.size)).round().astype(int))]
    w = ch_wgt[idx] / ch_wgt[idx].sum()
    for wi, f in zip(w, ch_freq[idx], strict=True):
        u = np.log(f / nu_ref)
        s = i0 * np.exp(a * u + b * u**2)
        pb = beam.I(xx[m], yy[m], f / 1e6)
        num[k] += wi * s * pb
        den[k] += wi * pb
        num1[k] += wi * s
    print(f"bin {k}: {idx.size} channels, {ch_freq[idx].min() / 1e6:.0f}-{ch_freq[idx].max() / 1e6:.0f} MHz", flush=True)

uk = np.log(nu_eff / nu_ref)


def curvature(v):
    return np.polyfit(uk, np.log(v), 2)[0]


pred_pb = curvature(num / den) - b  # beam + sky binning
pred_sky = curvature(num1) - b  # sky binning alone (PB = 1)
meas = post["rec_beta_model"][m] - post["truth_beta"][m]
std = post["rec_beta_model_std"][m]

# %%
fmt = "{:22s} {:>7d} {:>8.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>8.2f}"
print(f"\n{'region':22s} {'n':>7s} {'r [deg]':>8s} {'meas':>9s} {'pred PB':>9s} {'pred sky':>9s} {'std':>8s}   (medians)")
rows = [(c, dom == i) for i, c in enumerate(comps)]
edges = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5]
rows += [(f"r [{lo}, {hi}) deg", (r >= lo) & (r < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
for name, sel in rows:
    if sel.sum():
        print(fmt.format(name, sel.sum(), np.median(r[sel]), np.median(meas[sel]), np.median(pred_pb[sel]),
                         np.median(pred_sky[sel]), np.median(std[sel])))
for i, c in enumerate(comps):
    sel = dom == i
    if sel.sum() > 10:
        cc = np.corrcoef(meas[sel], pred_pb[sel])[0, 1]
        print(f"{c}: correlation measured vs predicted (beam) beta error {cc:+.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, c in enumerate(comps):
    sel = dom == i
    axes[0].scatter(r[sel], meas[sel], s=1, alpha=0.3, label=f"{c} measured")
order = np.argsort(r)
axes[0].plot(r[order], pred_pb[order], ",", color="k", alpha=0.5)
axes[0].plot([], [], "k.", label="predicted (beam + sky binning)")
axes[0].set(xlabel="distance from pointing centre [deg]", ylabel=r"$\beta$ error", ylim=(-1.5, 1.5))
axes[0].axhline(0, color="0.5", lw=0.8)
axes[0].legend(markerscale=8, fontsize=8)
axes[1].hexbin(pred_pb, meas, gridsize=80, bins="log", extent=(-1, 1, -1.5, 1.5), cmap="viridis")
axes[1].plot([-1, 1], [-1, 1], "w--", lw=0.8)
axes[1].set(xlabel=r"predicted $\beta$ error (beam + sky binning)", ylabel=r"measured $\beta$ error")
fig.suptitle(f"{variant}: beta error vs primary beam, flux > {FLUX_MIN:g} mJy/arcsec$^2$")
fig.tight_layout()
fig.savefig(f"{run}/compare/beta_radius.png", dpi=100)
print("saved:", f"{run}/compare/beta_radius.png")

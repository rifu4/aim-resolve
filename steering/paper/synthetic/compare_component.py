# %%
# ---------------------------------------------------------------------------
# Comparison of a single component of a component run (make_configs.py --comp)
# with its truth component (make_truth_sfrun.py / make_truth_comp.py): I0 at
# nu_ref, alpha and beta of the component alone (its own spectral model), in the
# mask where the component's reconstructed brightness exceeds FLUX_MIN. Same
# layout as maps_mask.png of compare_synthetic.py (truth | reconstruction |
# rec - truth | posterior std). The truth alpha / beta are those of the truth
# component's spectrum at the run's model frequencies (anchored slope and u^2
# coefficient, as the model's alpha / beta).
#
# usage: python compare_component.py --run DIR --truth main_comp [--comp o0|o1|o2|bg|t0|p0] [--sky sky.1]
# Output: DIR/compare/maps_<comp>.png and printed statistics
# ---------------------------------------------------------------------------
import argparse
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import REF_IDX, STEERING, truth_dir

import aim_resolve as aim
from aim_resolve.model.map import map_signal

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--truth", required=True)
parser.add_argument("--comp", default="o0")
parser.add_argument("--sky", default="sky.1")
args = parser.parse_args()
TRUTH_KEY = dict(o0="i0_obj", o1="i0_o1", o2="i0_o2", bg="i0_bg", t0="i0_tiles", p0="i0_pts")
SPEC_KEY = dict(p0=("alpha_pts", "beta_pts"))  # spectra of the truth component (default: the diffuse fields)
AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2
FLUX_MIN = 1e-2

run = os.path.abspath(args.run)
cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
sky = cfg.instantiate_sec(args.sky)
model = next(m for m in sky.models if m.prefix.split(".")[0] == args.comp)
freq = np.asarray(sky.freq, dtype="float64")
u = np.log(freq / freq[REF_IDX])
with open(f"{run}/last.pkl", "rb") as f:
    samples, *_ = pickle.load(f)


def place(v):
    v = np.asarray(v, dtype="float64")
    return v if v.shape[-2:] == sky.grid.shape else np.asarray(map_signal(model.grid, sky.grid)(v))


i0_k = np.stack([place(model.ref_freq_model(x)) for x in samples]) * CONV_FACTOR
a_k = np.stack([place(model.spectral_index(x)) for x in samples])
b_k = np.stack([place(model.spectral_curvature(x)) for x in samples])

# truth component: its spectrum at the run's model frequencies -> anchored slope and u^2 coefficient
t = np.load(f"{truth_dir(args.truth)}/truth.npz")
ut = np.log(freq / float(t["nu_ref"]))
ka, kb = SPEC_KEY.get(args.comp, ("alpha_diffuse", "beta_diffuse") if "alpha_diffuse" in t.files else ("alpha", "beta"))
if f"alpha_{args.comp}" in t.files:  # a spectrum per component (make_truth_sep.py)
    ka, kb = f"alpha_{args.comp}", f"beta_{args.comp}"
key = f"i0_{args.comp}" if f"i0_{args.comp}" in t.files else TRUTH_KEY[args.comp]  # 1 deg truth: i0_o0, i0_o1, i0_bg
cube = np.stack([t[key] * np.exp(t[ka] * x + t[kb] * x**2) for x in ut])
ln = np.log(np.clip(cube, 1e-30, None))
t_i0 = cube[REF_IDX] * CONV_FACTOR
t_alpha = np.tensordot(u, ln - ln[REF_IDX], axes=1) / np.sum(u**2)
t_beta = np.polyfit(u, ln.reshape(u.size, -1), 2)[0].reshape(t_i0.shape)

rec = dict(i0=i0_k.mean(0), alpha=a_k.mean(0), beta=b_k.mean(0))
std = dict(i0=i0_k.std(0, ddof=1), alpha=a_k.std(0, ddof=1), beta=b_k.std(0, ddof=1))
mask = rec["i0"] > FLUX_MIN
print(f"{args.comp} of {os.path.basename(run)} (truth {args.truth}): {mask.sum()} pixels with I0 > {FLUX_MIN:g} mJy/arcsec^2")


def stat(d, s=None):
    d0 = d[mask]
    out = f"median {np.median(d0):+.3f}, MAD {1.4826 * np.median(np.abs(d0 - np.median(d0))):.3f}"
    if s is not None:
        out += f", |z|<1 {np.mean(np.abs(d0 / s[mask]) < 1):.2f}"
    return out


with np.errstate(divide="ignore", invalid="ignore"):
    rel_i0 = (rec["i0"] - t_i0) / t_i0
rows = [
    ("I0 [mJy/arcsec$^2$]", t_i0, rec["i0"], rel_i0, std["i0"] / rec["i0"], dict(cmap="inferno", norm=LogNorm(FLUX_MIN, 10)),
     0.2, "(rec - truth) / truth", "relative std"),
    ("alpha (model)", t_alpha, rec["alpha"], rec["alpha"] - t_alpha, std["alpha"], dict(cmap="RdYlBu_r", vmin=-2.5, vmax=0.5),
     0.5, "rec - truth", "posterior std"),
    ("beta (model)", t_beta, rec["beta"], rec["beta"] - t_beta, std["beta"], dict(cmap="coolwarm", vmin=-1, vmax=1),
     0.5, "rec - truth", "posterior std"),
]
fig, axes = plt.subplots(3, 4, figsize=(20, 14.7))
contour = np.where(np.isfinite(rec["i0"]), rec["i0"], 0)
for r, (name, tr, rc, d, sd, kw, dlim, dlab, slab) in enumerate(rows):
    for c, (arr, title, kwc) in enumerate([
        (tr, f"truth {name}", kw), (rc, f"reconstruction {name}", kw),
        (d, f"{dlab}: {stat(d, None if r == 0 else sd)}", dict(cmap="PuOr", vmin=-dlim, vmax=dlim)),
        (sd, slab, dict(cmap="viridis", vmin=0, vmax=dlim)),
    ]):
        ax = axes[r, c]
        im = ax.imshow(np.where(mask, arr, np.nan).T, origin="lower", **kwc)
        ax.contour(contour.T, levels=[FLUX_MIN, 1e-1, 1], colors="k", linewidths=0.4)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    print(f"{name:20s} {stat(d, None if r == 0 else sd)}")
fig.suptitle(f"component {args.comp} of {os.path.basename(run)} (truth {args.truth}): {args.comp} I0 > {FLUX_MIN:g} mJy/arcsec$^2$")
fig.tight_layout()
os.makedirs(f"{run}/compare", exist_ok=True)
fig.savefig(f"{run}/compare/maps_{args.comp}.png", dpi=100)
print("saved:", f"{run}/compare/maps_{args.comp}.png")

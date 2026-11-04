# %%
# ---------------------------------------------------------------------------
# Low-resolution version of a 1 deg synthetic run (fast diagnostic, ~9x fewer pixels):
# all model grids with factor 1 instead of FAC = 3 (7.03 arcsec pixels instead of 2.34),
# on the same synthetic data with the baselines beyond the grid's Nyquist limit removed.
#   data    rows with |uv| nu_max / c > U_FRAC / (2 pixel) dropped (all channels of a row, so
#           every channel keeps the same rows), SF and MF file of the source variant
#           (--cut-only: instead the files of the low-resolution variant itself, generated from the
#           low-resolution truth by make_vis.py --field 1deglr, cut in place; nothing else is done)
#   truth   the source truth averaged over 3 x 3 fine pixels: every component's brightness
#           at 24 frequencies over the band, alpha / beta of the total from a quadratic fit
#           of the averaged spectrum (truth/<truth>_lr)
#   config  the source run's config with factor 1, the uv-cut data, kernels *_512 and the
#           run directory <variant>lr (offsets: set by prefit_comp.py --get-offsets)
#
# --u-frac F: cut at F times the Nyquist limit (default 0.95; with 0.95 the fit stalled at
#           reduced chi^2 ~ 9); other values give the variant <variant>lr_u<100 F>, with its own data,
#           kernels and run directory
#
# usage: python make_lowres_1deg.py [--cut-only] [--u-frac F] [variant, default A1deg]
# Afterwards: the kernels (instantiate lh.0 / lh.1), prefit_comp.py --truth-comps --get-offsets, the run.
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import FAC, OUT_1DEG, STEERING, SYNTH_DATA, VARIANTS, data_variant, noise_seed, truth_dir

CUT_ONLY = "--cut-only" in sys.argv
argv = [a for a in sys.argv[1:] if a != "--cut-only"]
U_FRAC = 0.95  # fraction of the Nyquist limit 1 / (2 pixel) kept
if "--u-frac" in argv:
    i = argv.index("--u-frac")
    U_FRAC = float(argv[i + 1])
    del argv[i:i + 2]
src = argv[0] if argv else "A1deg"
tag = "" if U_FRAC == 0.95 else f"_u{round(100 * U_FRAC)}"
dst = f"{src}lr{tag}"
RUNS = f"{OUT_1DEG}/r10_sf5_mf15"
C_LIGHT = 299792458.0
PIX_RAD = np.deg2rad(1.0) / 512  # 1 deg / 512 pixels (7.03 arcsec)
U_MAX = U_FRAC / (2 * PIX_RAD)  # [wavelengths]
assert VARIANTS.get(dst) == (f"{VARIANTS[src][0]}_lr", VARIANTS[src][1]), f"add {dst} to VARIANTS in synth_common.py"

# %%
# uv-cut data
for stage, band in (("sf", "986-1137mhz"), ("mf", "961-1503mhz")):
    out = f"{SYNTH_DATA}/eso_{band}_rows10pct_w_synth_{data_variant(dst, stage)}_s{noise_seed(dst)}.npz"
    fn = out if CUT_ONLY else f"{SYNTH_DATA}/eso_{band}_rows10pct_w_synth_{data_variant(src, stage)}_s{noise_seed(src)}.npz"
    d = dict(np.load(fn, allow_pickle=True))  # in memory: with --cut-only the file is overwritten
    uvlen = np.hypot(d["antpos0"][:, 0], d["antpos0"][:, 1]) * float(np.max(d["freq"])) / C_LIGHT
    keep = uvlen <= U_MAX
    n = keep.size
    arr = {}
    for k in d:
        a = d[k]
        if k in ("vis", "weight"):
            a = a[:, keep, :]
        elif k.startswith("antpos") and a.shape[0] == n:
            a = a[keep]
        arr[k] = a
    np.savez(out, **arr)
    w = np.asarray(d["weight"], dtype="float64").sum(axis=(0, 2))
    print(f"{os.path.basename(out)}: rows {n} -> {keep.sum()} (|uv| <= {U_MAX / 1e3:.1f} klambda at the highest channel), "
          f"weight kept {w[keep].sum() / w.sum():.3f}", flush=True)
if CUT_ONLY:
    sys.exit()

# %%
# truth averaged over 3 x 3 fine pixels
t = np.load(f"{truth_dir(VARIANTS[src][0])}/truth.npz")
nu_ref = float(t["nu_ref"])
comps = [k for k in t.files if k.startswith("i0_")]
nus = np.geomspace(950e6, 1520e6, 24)
u = np.log(nus / nu_ref)


def block(a):
    n = a.shape[-1] // FAC
    return a.reshape(a.shape[:-2] + (n, FAC, n, FAC)).mean(axis=(-3, -1))


spec = np.exp(t["alpha"][None] * u[:, None, None] + t["beta"][None] * u[:, None, None] ** 2)  # (nu, X, Y)
cube = {c: block(t[c][None] * spec) for c in comps}
total = sum(cube.values())
ln = np.log(np.clip(total, 1e-300, None)).reshape(u.size, -1)
beta, alpha, ln_i0 = np.polyfit(u, ln, 2)
shape = total.shape[1:]
i_ref = int(np.argmin(np.abs(u)))
assert abs(u[i_ref]) < 0.02, u[i_ref]
lr = {c: block(t[c]) for c in comps}  # brightness at nu_ref
res = np.abs(np.exp(np.polyval(np.stack([beta, alpha, ln_i0]), u[:, None])) / total.reshape(u.size, -1) - 1).max()
tdir = truth_dir(VARIANTS[dst][0])
os.makedirs(tdir, exist_ok=True)
np.savez(f"{tdir}/truth.npz", nu_ref=nu_ref, nu_eff=t["nu_eff"], i0=sum(lr.values()), **lr, alpha=alpha.reshape(shape),
         beta=beta.reshape(shape), source=truth_dir(VARIANTS[src][0]), factor=FAC)
print(f"truth {tdir}: {shape}, components {comps}, max relative deviation of the quadratic spectrum {res:.2e}")

# %%
# run config
docs = list(yaml.safe_load_all(open(f"{RUNS}/{src}/config.yml")))
odir = os.path.relpath(f"{RUNS}/{dst}", STEERING)
for doc in docs:
    for sec, v in doc.items():
        if not isinstance(v, dict):
            continue
        if sec == "opt":
            v["odir"] = odir
        if isinstance(v.get("grid"), dict):
            v["grid"]["factor"] = 1
        if "fname" in v:
            v["fname"] = v["fname"].replace(f"_synth_{src}_", f"_synth_{dst}_")
        for k in ("psf_kernel_fn", "n_inv_kernel_fn"):
            if k in v:
                v[k] = v[k].replace("_synth1d_r10_", f"_synth1dlr{tag}_r10_").replace("_1536.pkl", "_512.pkl")
os.makedirs(f"{STEERING}/{odir}", exist_ok=True)
with open(f"{STEERING}/{odir}/config.yml", "w") as f:
    yaml.safe_dump_all(docs, f, sort_keys=False, default_flow_style=None, width=120)
print("saved:", f"{STEERING}/{odir}/config.yml")

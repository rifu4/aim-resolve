# flux at the boundaries of the object boxes (o0, o1) against truth: rings inside / outside each box edge
# usage: edge_check.py PKL RUN TRUTH SEC [PNG]   SF (freq [1.0]): apparent truth at 1062 MHz; MF: intrinsic truth at model freq 1
import os, sys, pickle
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from scipy.ndimage import distance_transform_edt
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic"); os.chdir(ST)
from synth_common import TruthSky, truth_dir
import aim_resolve as aim
from aim_resolve.fast_resolve.beam import build_primary_beam
pkl, run, truth, sec = sys.argv[1:5]
png = sys.argv[5] if len(sys.argv) > 5 else None
p = f"runs/synth_1deg/r10_sf5_mf15/{run}"
cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{p}/config.yml"), aim.get_builders)
sky = cfg.instantiate_sec(sec)
samples, _, mj = pickle.load(open(pkl, "rb"))
freq = np.asarray(sky.freq, dtype="float64").ravel()
k = min(1, freq.size - 1); sf = freq[k] < 1e8; nu = 1062e6 if sf else freq[k]
ts = TruthSky(truth)
u = np.log(nu / ts.nu_ref)
if ts.comps:
    tc = {c: i0 * np.exp(a * u + b * u**2) for c, (i0, a, b) in zip(("bg", "o0", "o1"), ts.comps)}
else:  # one spectrum for all components
    t = np.load(f"{truth_dir(truth)}/truth.npz")
    tc = {c: t[f"i0_{c}"] * np.exp(ts.alpha * u + ts.beta * u**2) for c in ("bg", "o0", "o1")}
pb = np.asarray(build_primary_beam(sky.grid, [nu], channel_average=False), dtype="float64").squeeze() if sf else 1.0
tc = {c: v * pb for c, v in tc.items()}
T = sum(tc.values())
comps = {c.prefix.split(".")[0]: np.mean([np.asarray(c(x, map=True), dtype="float64").reshape((-1,) + T.shape)[k] for x in samples], 0) for c in sky.models}
rec = sum(comps.values())
print(f"EDGE {run} {sec} mj {mj} nu {nu / 1e6:.1f} ({'apparent' if sf else 'intrinsic'}) total rec/truth {rec.sum() / T.sum():.3f}")
for o in ("o0", "o1"):
    B = comps[o] != 0
    din, dout = distance_transform_edt(B), distance_transform_edt(~B)
    rings = {"in 0-3px": B & (din <= 3), "in 3-10px": B & (din > 3) & (din <= 10), "in 10-25px": B & (din > 10) & (din <= 25),
             "interior": B & (din > 25), "out 0-3px": ~B & (dout <= 3), "out 3-10px": ~B & (dout > 3) & (dout <= 10)}
    for nm, M in rings.items():
        print(f"EDGE {o} {nm:10s} n {M.sum():6d} truth share {T[M].sum() / T.sum():.4f} | rec/truth {rec[M].sum() / T[M].sum():.3f}"
              f" | {o} rec/truth {comps[o][M].sum() / max(tc[o][M].sum(), 1e-30):.3f} | bg rec/truth {comps['bg'][M].sum() / tc['bg'][M].sum():.3f}"
              f" | {o} share rec {comps[o][M].sum() / rec[M].sum():.3f} truth {tc[o][M].sum() / T[M].sum():.3f}")
    a, b = B & (din <= 2), ~B & (dout <= 2)  # step across the edge (sum of all components)
    print(f"EDGE {o} step in/out 2px: rec {np.median(rec[a]) / np.median(rec[b]):.3f} truth {np.median(T[a]) / np.median(T[b]):.3f}")
if png:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    for a_, img, ti, kw in [(ax[0], rec / T, "rec / truth", dict(vmin=0.7, vmax=1.3, cmap="RdBu_r")),
                            (ax[1], comps["o0"] + comps["o1"], "o0 + o1 rec", dict(norm=matplotlib.colors.LogNorm(T.max() * 1e-4, T.max()))),
                            (ax[2], comps["bg"] / tc["bg"], "bg rec / bg truth", dict(vmin=0.5, vmax=2, cmap="RdBu_r"))]:
        im = a_.imshow(img.T, origin="lower", **kw); a_.set_title(ti); plt.colorbar(im, ax=a_)
        for o in ("o0", "o1"):
            a_.contour((comps[o] != 0).T.astype(float), levels=[0.5], colors="k", linewidths=0.6)
    f.suptitle(f"{run} {sec} mj {mj}"); f.tight_layout(); f.savefig(png, dpi=110)
    print("EDGE plot", png)

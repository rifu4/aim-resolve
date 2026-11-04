# flux per region vs truth through the SF -> MF switch: end of SF (sky.0, apparent) and MF snapshots (sky.1, intrinsic), per component
import os, sys, pickle
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic"); os.chdir(ST)
from synth_common import TruthSky, truth_dir
import aim_resolve as aim
from aim_resolve.fast_resolve.beam import beam_offsets, build_primary_beam
SNAP = sys.argv[1]
m = np.load(f"{truth_dir('1deg_main_lr')}/truth.npz")
o1 = np.argmax(np.stack([m[f"i0_{c}"] for c in ("bg", "o0", "o1")]), 0) == 2
for run, truth, sec in [a.split(":") for a in sys.argv[2:]]:
    p = f"runs/synth_1deg/r10_sf5_mf15/{run}"
    cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{p}/config.yml"), aim.get_builders)
    sky = cfg.instantiate_sec(sec)
    samples, _, mj = pickle.load(open(f"{SNAP}/{run}.pkl", "rb"))
    freq = np.asarray(sky.freq, dtype="float64").ravel()
    k = min(1, freq.size - 1); nu = freq[k] if freq[k] > 1e8 else 1062e6
    T = np.asarray(TruthSky(truth)(nu), dtype="float64")
    pb = np.asarray(build_primary_beam(sky.grid, [nu], channel_average=False), dtype="float64").squeeze()
    comps = {c.prefix.split(".")[0]: np.mean([np.asarray(c(x, map=True), dtype="float64").reshape((-1,) + T.shape)[k] for x in samples], 0) for c in sky.models}
    rec = sum(comps.values())
    xx, yy = beam_offsets(sky.grid); r = np.hypot(xx, yy)
    print("SW", run, sec, "mj", mj, "nu", round(nu / 1e6, 1), "whole rec/truth comps", {n: round(float(v.sum()), 3) for n, v in {}.items()}, f"total {rec.sum() / T.sum():.3f}")
    for nm, M in {"O1 r<.25": o1 & (r < .25), "O1 .30-.35": o1 & (r >= .3) & (r < .35), "O1 >.35": o1 & (r >= .35), "other r>.25": ~o1 & (r > .25)}.items():
        print(f"SW   {nm:12s} PB {np.median(pb[M]):.3f} (flux wtd {(pb[M]*T[M]).sum()/T[M].sum():.3f}) rec/truth {rec[M].sum()/T[M].sum():.3f}  rec/apparent {rec[M].sum()/(T[M]*pb[M]).sum():.3f} | bg share {comps['bg'][M].sum()/T[M].sum():.3f}")

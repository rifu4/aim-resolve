# is the flux excess at O1's edge a mean-vs-median effect of the positive model? region sums of sample mean and geometric mean; excess vs radius, intrinsic and apparent
import os, sys, pickle
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic"); os.chdir(ST)
from synth_common import truth_dir
import aim_resolve as aim
from aim_resolve.fast_resolve.beam import beam_offsets, build_primary_beam
m = np.load(f"{truth_dir('1deg_main_lr')}/truth.npz")
o1 = np.argmax(np.stack([m[f"i0_{c}"] for c in ("bg", "o0", "o1")]), 0) == 2
for run, truth in [a.split(":") for a in sys.argv[1:]]:
    t = np.load(f"{truth_dir(truth)}/truth.npz")
    p = f"runs/synth_1deg/r10_sf5_mf15/{run}"
    cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{p}/config.yml"), aim.get_builders)
    sky = cfg.instantiate_sec("sky.1")
    samples, *_ = pickle.load(open(f"{p}/last.pkl", "rb"))
    cube = np.stack([sum(np.asarray(c(x, map=True), dtype="float64")[1] for c in sky.models) for x in samples])
    mean, gm = cube.mean(0), np.exp(np.log(np.clip(cube, 1e-300, None)).mean(0))
    slog = np.log(np.clip(cube, 1e-300, None)).std(0)
    xx, yy = beam_offsets(sky.grid); r = np.hypot(xx, yy)
    freq = np.asarray(sky.freq, dtype="float64")
    pb = np.asarray(build_primary_beam(sky.grid, freq[[1]], channel_average=False), dtype="float64").squeeze()
    print("RUN", run, len(samples), "samples")
    for nm, M in {"O1 r<0.25": o1 & (r < .25), "O1 .30-.35": o1 & (r >= .3) & (r < .35), "O1 >.35": o1 & (r >= .35), "other r>.25": ~o1 & (r > .25)}.items():
        tt = t["i0"][M].sum()
        print(f"RUN  {nm:12s} mean/truth {mean[M].sum()/tt:.3f}  geomean/truth {gm[M].sum()/tt:.3f}  median sigma_ln {np.median(slog[M]):.3f}  median PB {np.median(pb[M]):.3f}  flux-wtd excess*PB share {((mean-t['i0'])[M]*pb[M]).sum()/(t['i0'][M]*pb[M]).sum():+.3f}")

# object offsets: config value (get_offset, log of the mean over the box) against the mean log brightness of the truth in the box
import os, sys, pickle
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic"); os.chdir(ST)
from synth_common import truth_dir
import aim_resolve as aim
run, truth = sys.argv[1:3]
p = f"runs/synth_1deg/r10_sf5_mf15/{run}"
cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{p}/config.yml"), aim.get_builders)
sky = cfg.instantiate_sec("sky.0")
samples, _, _ = pickle.load(open(sys.argv[3], "rb")); x0 = next(iter(samples))
t = np.load(f"{truth_dir(truth)}/truth.npz")
for c in sky.models:
    n = c.prefix.split(".")[0]
    i = t[f"i0_{n}"]
    B = np.asarray(c(x0, map=True), dtype="float64").reshape(i.shape) != 0
    v = i[B]; pos = v[v > 0]
    print(f"{n}: box px {B.sum()} | pos frac in box {pos.size / v.size:.2f} | log(mean box) {np.log(v.mean()):.2f} | mean log (pos) {np.log(pos).mean():.2f} | median log {np.log(np.median(pos)):.2f} | std log {np.log(pos).std():.2f} | flux outside box {i[~B].sum() / i.sum():.3f}")

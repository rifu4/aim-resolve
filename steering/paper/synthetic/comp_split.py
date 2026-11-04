# flux of each model component in the O1 regions, truth vs reconstruction (reference frequency, posterior mean)
import os, sys
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic")
import sys, pickle, numpy as np
sys.path.insert(0, "paper/synthetic")
from synth_common import truth_dir
import aim_resolve as aim
from aim_resolve.fast_resolve.beam import beam_offsets
import os
os.chdir(ST)
run, truth = sys.argv[1], sys.argv[2]
t = np.load(f"{truth_dir(truth)}/truth.npz")
cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
sky = cfg.instantiate_sec("sky.1")
samples, *_ = pickle.load(open(f"{run}/last.pkl", "rb"))
xx, yy = beam_offsets(sky.grid); r = np.hypot(xx, yy)
o1 = np.argmax(np.stack([t[f"i0_{c}"] for c in ("bg", "o0", "o1")]), 0) == 2
names = {m.prefix.split(".")[0]: m for m in sky.models}
rec = {k: np.mean([np.asarray(m(x, map=True), dtype="float64")[1] for x in samples], 0) for k, m in names.items()}
for nm, M in {"O1 r<0.25": o1 & (r < 0.25), "O1 0.30-0.35": o1 & (r >= 0.3) & (r < 0.35), "O1 >0.35": o1 & (r >= 0.35)}.items():
    s = t["i0"][M].sum()
    print("SPLIT", f"{nm:13s} truth " + " ".join(f"{k} {t[f'i0_{k}'][M].sum() / s:.3f}" for k in names) + " | rec " + " ".join(f"{k} {rec[k][M].sum() / s:.3f}" for k in names) + f" | sum {sum(v[M].sum() for v in rec.values()) / s:.3f}")
print("SPLIT whole field rec/truth " + " ".join(f"{k} {rec[k].sum() / t[f'i0_{k}'].sum():.3f}" for k in names) + f" total {sum(v.sum() for v in rec.values()) / t['i0'].sum():.3f}")

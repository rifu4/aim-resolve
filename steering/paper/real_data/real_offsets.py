# stage-5 offsets of a zoom extension (run 15: zoom_ext_newsf_bin.yml) with the get_offset rule (mean log field, taper at its prior median, floor 1e-5): each stage-4
# component (posterior mean of 4_rec/last.pkl) on the grid of its stage-5 counterpart, times flux_scale
# usage: real_offsets.py EXTENSION_CONFIG (e.g. nifty/config/zoom_ext_newsf_bin_t025.yml)
import os, sys, pickle, tempfile, shutil
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ST)
import aim_resolve as aim
from aim_resolve.model.map import map_signal
from aim_resolve.modeling import get_offset
R = "runs/fast_vi_1f_1024_1z_b"
d = aim.yaml_load(sys.argv[1])
tmp = tempfile.mkdtemp(); os.makedirs(f"{tmp}/files")
for f in ("base.yml", "4_pre.yml"):
    shutil.copy(f"{R}/files/{f}", f"{tmp}/files/{f}")
d["odir"] = tmp
base, ext = aim.extension_func(**d)
cfg = aim.OptimizeKLConfig.from_file((base, ext), aim.get_builders)
sky4, sky5 = cfg.instantiate_sec("sky.4"), cfg.instantiate_sec("sky.5")
samples = pickle.load(open(f"{R}/opt/4_rec/last.pkl", "rb"))[0]
fs = float(d.get("flux_scale", 1.0))
old = {c.prefix.split(".")[0]: c for c in sky4.models}
for c5 in sky5.models:
    n = c5.prefix.split(".")[0]
    if n not in old or not (n == "bg" or n.startswith("o")):
        continue
    c4 = old[n]
    rec = fs * np.asarray(map_signal(c4.grid, c5.grid)(samples.mean(c4)), dtype="float64")
    rec = rec.reshape(rec.shape[-2:]) if rec.ndim > 2 else rec
    off = get_offset(c5, rec, np.ones(rec.shape, bool), c5.freq)
    print(f"RESULT {n}: offset {off} | stage-4 offset + ln {fs:g}: {c4.offset if hasattr(c4, 'offset') else '?'} -> config {cfg.sections[f'sky_{n}.5'].get('offset'):.2f}"
          f" | log of mean {np.log(rec.mean()):.2f} | grid {rec.shape}", flush=True)
shutil.rmtree(tmp)

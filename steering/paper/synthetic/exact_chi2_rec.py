# Exact chi^2 of a run's posterior mean (apparent sky, model bins) against the MF data, normalised with the
# make_vis noise level (pure noise -> 1; floor of the best representable sky ~1.69 for A1deglr_u50), total and per bin,
# plus the integrated O1 spectrum rec / truth per bin in radial regions.
import os, sys, pickle
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic")
os.chdir(ST)
from synth_common import TruthSky, truth_dir
import aim_resolve as aim, nifty8 as ift
from aim_resolve.resolve.observation import Observation
from aim_resolve.fast_resolve.response import build_exact_responses
from aim_resolve.fast_resolve.beam import build_primary_beam, beam_offsets

run, truth = sys.argv[1], sys.argv[2]
cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
sec = cfg.sections
sky = cfg.instantiate_sec("sky.1")
with open(f"{run}/last.pkl", "rb") as f:
    samples, _, mj = pickle.load(f)
s = np.mean([np.asarray(sky(x), dtype="float64") for x in samples], 0)
freq = np.asarray(sky.freq, dtype="float64")
fname = sec["data.1"]["fname"]
bounds = np.asarray(sec["lh.1"]["freq_bounds"], dtype="float64")
obs = Observation.load_npz(fname).to_resolve_obs().to_double_precision()
R = build_exact_responses(obs, sky.grid, freq, bounds)[0]
pb = np.asarray(build_primary_beam(sky.grid, freq, obs, freq_bounds=bounds), dtype="float64")
m = np.asarray(R(ift.makeField(R.domain, s * pb)).val)[0]
d, w = np.asarray(obs.vis.val)[0], np.asarray(obs.weight.val)[0]
nu = np.asarray(obs.freq, dtype="float64")
nl = np.load(fname.replace(f"_synth_{os.path.basename(fname).split('_synth_')[1]}", "_synth_noise_level.npz"))["noise_level"]
nl = np.ones_like(nl) if "wfix" in fname else nl  # weights already rescaled
ok = w > 0
r = np.where(ok, np.abs(d - m) ** 2 * w / (2 * nl[None, :] ** 2), 0)
kb = np.clip(np.searchsorted(bounds, nu, side="right") - 1, 0, freq.size - 1)
print(f"RESULT {os.path.basename(run)} (major iteration {mj}): exact chi2 {r[ok].mean():.3f} | per bin "
      + " ".join(f"{r[:, kb == k][ok[:, kb == k]].mean():.2f}" for k in range(freq.size)))
T = TruthSky(truth); t = np.load(f"{truth_dir(truth)}/truth.npz")
tc = np.stack([T(f) for f in freq])
xx, yy = beam_offsets(sky.grid); rr = np.hypot(xx, yy)
o1 = np.argmax(np.stack([t[f"i0_{c}"] for c in ("bg", "o0", "o1")]), 0) == 2
u = np.log(freq / T.nu_ref)
for lo, hi in [(0, 0.25), (0.3, 0.35), (0.35, 1)]:
    M = o1 & (rr >= lo) & (rr < hi)
    q = s[:, M].sum(1) / tc[:, M].sum(1)
    print(f"RESULT   O1 r [{lo}, {hi}): integrated rec/truth per bin " + " ".join(f"{v:.3f}" for v in q)
          + f" | d_beta {np.polyfit(u, np.log(q), 2)[0]:+.3f}")

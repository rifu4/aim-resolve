# per-channel exact chi^2 of the posterior mean of full-sky runs (weights rescaled), per uv annulus [klambda]
import os, sys, pickle
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic"); os.chdir(ST)
import aim_resolve as aim
R0 = "runs/synth_1deg/r10_sf5_mf15"
edges = [0, .25, .5, 1, 2, 4, 7.4]
print("RESULT annuli [klambda]", edges, flush=True)
for run in sys.argv[1:]:
    cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{R0}/{run}/config.yml"), aim.get_builders)
    lh = cfg.instantiate_sec("lh.1")
    ch, sky = lh["channel_residual"], lh["sky_model"]
    obs = ch.obs; d, w = np.asarray(obs.vis.val)[0], np.asarray(obs.weight.val)[0]
    ok = w > 0
    lam = np.hypot(obs.uvw[:, 0], obs.uvw[:, 1])[:, None] * ch.nu[None] / 299792458.0 / 1e3
    with open(f"{R0}/{run}/last.pkl", "rb") as f:
        samples, _, mj = pickle.load(f)
    s = np.mean([np.asarray(sky(x), dtype="float64") for x in samples], 0)
    m = ch.predict(s)
    r = w * np.abs(d - m) ** 2 / 2
    print(f"RESULT {run} (mj {mj}): per-channel chi2 {r[ok].mean():.3f} | annuli " + " ".join(f"{r[ok & (lam >= a) & (lam < b)].mean():.3f}" for a, b in zip(edges[:-1], edges[1:])) + " | n " + " ".join(str(int((ok & (lam >= a) & (lam < b)).sum() // 1000)) + "k" for a, b in zip(edges[:-1], edges[1:])), flush=True)

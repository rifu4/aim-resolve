# learned per-mode noise scale (posterior mean of noise_model(x), multiplies K^-1/2) vs Fourier radius and |K|
import os, sys, pickle
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ST)
import aim_resolve as aim
for run in sys.argv[1:]:
    cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
    lh = cfg.instantiate_sec("lh.1")
    resp = lh["sky_response"]
    with open(f"{run}/last.pkl", "rb") as f:
        samples, _, mj = pickle.load(f)
    nm2 = np.mean([np.asarray(resp.noise_model(x)) ** 2 for x in samples], 0)  # weight factor of the residual per mode
    nk = np.abs(1 / np.asarray(resp.fft_kernel) ** 2)
    n = nk.shape[-1]; print("shapes nm2", nm2.shape, "nk", nk.shape, "nan", np.isnan(nm2).sum(), np.isnan(nk).sum())
    f = np.fft.fftfreq(n); rho = np.hypot(*np.meshgrid(f, f, indexing="ij")) / 0.5
    rel = nk / nk.max(axis=(-2, -1), keepdims=True)
    print(f"\n{os.path.basename(run)} (mj {mj}): weight nm^2 per mode (1 = prior mode), medians over 8 bins")
    for lo, hi in [(0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.5)]:
        s = (rho >= lo) & (rho < hi)
        if not s.any(): continue
        print(f"   rho [{lo:.2f}, {hi:.2f}) Nyquist: n {s.sum():7d}  nm^2 median {np.median(nm2[:, s]):.3f}  p10 {np.percentile(nm2[:, s], 10):.3f}")
    for lo, hi in [(0, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.1)]:
        s = (rel >= lo) & (rel < hi)
        if not s.any(): continue
        print(f"   K/max [{lo:g}, {hi:g}): frac {s.mean():.3f}  nm^2 median {np.median(nm2[s]):.3f}  p10 {np.percentile(nm2[s], 10):.3f}")

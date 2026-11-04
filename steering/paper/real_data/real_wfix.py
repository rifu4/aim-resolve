# real data with weights rescaled per channel (w / s_c^2), written next to the originals, which stay
import numpy as np
for b in ("986-1137mhz", "961-1503mhz"):
    fn = f"/scratch/users/rfuchs/data/eso_{b}.npz"
    d = dict(np.load(fn, allow_pickle=True))
    nl = np.load(f"/scratch/users/rfuchs/data/synth/eso_{b}_synth_noise_level.npz")
    assert np.allclose(d["freq"], nl["freq"]), b
    d["weight"] = (d["weight"] / nl["noise_level"][None, None, :] ** 2).astype(d["weight"].dtype)
    np.savez(fn.replace(".npz", "_wfix.npz"), **d)
    print("written", b, d["weight"].shape, flush=True)

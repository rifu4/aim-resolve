# SF data from one MF bin: the channels of model bin 1 (lh.6 freq_bounds of mf_ext_8f.yml, [1007.35, 1053.28) MHz) of the
# corrected-weight MF data; row keys and aux tables are kept, the channel keys (vis, weight, freq) are cut
import numpy as np
D = "/scratch/users/rfuchs/data"
LO, HI = 1007354462.890625, 1053276308.59375
d = np.load(f"{D}/eso_961-1503mhz_wfix.npz")
f = d["freq"]; lo, hi = np.searchsorted(f, [LO, HI])
out = {k: d[k] for k in d.files}
out["vis"], out["weight"], out["freq"] = d["vis"][:, :, lo:hi], d["weight"][:, :, lo:hi], f[lo:hi]
w = out["weight"][0]; rows = (w > 0).any(1)
print(f"channels {hi - lo} ({f[lo] / 1e6:.2f}-{f[hi - 1] / 1e6:.2f} MHz) | rows with data {rows.sum()} of {rows.size} | flagged {np.mean(w == 0):.3f}"
      f" | effective freq {np.sum(w.sum(0) * f[lo:hi]) / w.sum() / 1e6:.2f} MHz")
np.savez(f"{D}/eso_1007-1053mhz_wfix.npz", **out)
print("written", f"{D}/eso_1007-1053mhz_wfix.npz")

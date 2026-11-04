# %%
# ---------------------------------------------------------------------------
# Row subset of the real data for the faster synthetic runs: a random FRACTION of
# the rows (baselines x times, fixed seed), all channels, flags and weights kept;
# the weights are scaled by 1 / FRACTION, so the information (S/N) per frequency
# bin stays about the same (the synthetic noise is drawn as sigma / sqrt(w)).
# Checked against the full data: binning error per uv scale (as check_nfreq_binning.py).
#
# usage: python make_rows_subset.py [fraction, default 0.1]
# Output: <data>_rows<pct>pct_w.npz next to the real data files (SF and MF)
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import DATA_MF, DATA_SF

FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
SEED = 0
C_LIGHT = 299792458.0


def subset_name(fn, fraction=FRACTION):
    return fn.replace(".npz", f"_rows{round(100 * fraction)}pct_w.npz")


def make_subset(src):
    d = np.load(src, allow_pickle=True)
    n = d["antpos0"].shape[0]
    idx = np.sort(np.random.default_rng(SEED).choice(n, size=int(FRACTION * n), replace=False))
    out = {}
    for k in d.files:
        a = d[k]
        if k == "vis":
            a = a[:, idx, :]
        elif k == "weight":
            a = a[:, idx, :] / FRACTION
        elif k.startswith("antpos") and a.shape[0] == n:
            a = a[idx]
        out[k] = a
    np.savez(subset_name(src), **out)
    print(f"{os.path.basename(src)}: rows {n} -> {idx.size}, weights x {1 / FRACTION:g} -> {subset_name(src)}", flush=True)
    return d, out


def uv_weights(d, n_uv=12, edges=None):
    """Weight per (uv annulus in lambda, channel), as check_channel_binning.py."""
    uvw, w, nu = np.asarray(d["antpos0"]), np.asarray(d["weight"])[0], np.asarray(d["freq"], dtype="float64")
    uvlen = np.hypot(uvw[:, 0], uvw[:, 1])
    if edges is None:
        lo = max(np.percentile(uvlen[uvlen > 0], 0.5) * nu.min() / C_LIGHT, 1.0)
        edges = np.geomspace(lo, uvlen.max() * nu.max() / C_LIGHT, n_uv + 1)
    w_uv = np.zeros((edges.size - 1, nu.size))
    for c in range(nu.size):
        i = np.searchsorted(edges, uvlen * nu[c] / C_LIGHT) - 1
        ok = (i >= 0) & (i < edges.size - 1)
        w_uv[:, c] = np.bincount(i[ok], weights=w[ok, c], minlength=edges.size - 1)
    return w_uv, w.sum(axis=0), edges


if __name__ == "__main__":
    make_subset(DATA_SF)
    full, sub = make_subset(DATA_MF)
    w_full, wc_full, edges = uv_weights(full)
    w_sub, wc_sub, _ = uv_weights(sub, edges=edges)
    np.savez(subset_name(DATA_MF).replace(".npz", "_uvweights.npz"), nu=np.asarray(full["freq"]), edges=edges,
             w_uv_full=w_full, w_uv_sub=w_sub, w_chan_full=wc_full, w_chan_sub=wc_sub)
    print("saved uv weights for the binning check")

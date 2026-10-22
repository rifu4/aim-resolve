# %%
# ---------------------------------------------------------------------------
# Phase 0 of the synthetic-data test (../synthetic_test_plan.md):
#   fixed channel bins (resolve's bins of the nominal model frequencies) and
#   the effective model frequency of each bin from the real weights / flags.
# Output: OUT/prep/prep.npz and prep.json.
# ---------------------------------------------------------------------------
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import DATA_MF, DATA_SF, OUT, REF_MF, effective_frequencies

odir = f"{OUT}/prep"
os.makedirs(odir, exist_ok=True)

with open(f"{REF_MF}/opt.yml") as f:
    ref_cfg = {k: v for d in yaml.safe_load_all(f) if d for k, v in d.items()}
nu_nominal = np.asarray(ref_cfg["sky_bg.6"]["freq"], dtype="float64")

d = np.load(DATA_MF, allow_pickle=True)
nu_data = np.asarray(d["freq"], dtype="float64")
w_chan = np.asarray(d["weight"]).sum(axis=(0, 1), dtype="float64")
print("nominal model freqs [MHz]:", np.round(nu_nominal / 1e6, 3).tolist())
nu_eff, bounds, chan_bin = effective_frequencies(nu_nominal, nu_data, w_chan)
print("bin bounds (fixed, likelihood `freq_bounds`) [MHz]:", np.round(bounds / 1e6, 3).tolist())
print("channels per bin:", np.bincount(chan_bin).tolist())
print("effective model freqs [MHz]:", np.round(nu_eff / 1e6, 3).tolist())
print("offset to nominal [MHz]:", np.round((nu_eff - nu_nominal) / 1e6, 3).tolist())

dsf = np.load(DATA_SF, allow_pickle=True)
nu_sf = np.asarray(dsf["freq"], dtype="float64")
w_sf = np.asarray(dsf["weight"]).sum(axis=(0, 1), dtype="float64")
print(f"single-frequency data: {nu_sf.size} channels, {nu_sf.min() / 1e6:.1f}-{nu_sf.max() / 1e6:.1f} MHz,"
      f" weighted mean {np.sum(w_sf * nu_sf) / np.sum(w_sf) / 1e6:.2f} MHz")

np.savez(f"{odir}/prep.npz", nu_nominal=nu_nominal, nu_eff=nu_eff, bounds=bounds, nu_data=nu_data, w_chan=w_chan,
         chan_bin=chan_bin)
with open(f"{odir}/prep.json", "w") as f:
    json.dump(dict(nu_nominal=nu_nominal.tolist(), nu_eff=nu_eff.tolist(), freq_bounds=bounds.tolist()), f, indent=2)
print("saved:", f"{odir}/prep.npz", f"{odir}/prep.json")

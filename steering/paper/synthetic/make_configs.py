# %%
# ---------------------------------------------------------------------------
# Phase 3 of the synthetic-data test (../synthetic_test_plan.md): run configs.
#
# One config per variant (RUNS/<variant>/config.yml, for nifty/start_rec.py):
#   opt.0  single-frequency reconstruction on the synthetic 986-1137 MHz data,
#          one correlated field on the 0.5 deg field, no primary beam (as the
#          single-frequency stages of the reference run)
#   opt.1  frequency transition (mode: freq) + multi-frequency reconstruction
#          on the synthetic 961-1503 MHz data, model frequencies = effective
#          frequencies with fixed bins (freq_bounds), pb_correction: true
# Priors: those of the reference run's diffuse components (bg.5 / o0.5 and
# bg.6 / o0.6, identical apart from the offset). Optimiser: base_opt.6 of
# nifty/config/mf_ext.yml for both stages, with N_IT_SF single-frequency and N_IT_MF multi-frequency
# iterations (synth_common; the single-frequency stage starts from scratch here, so it gets the same ramp-up).
# The kernels are shared by all variants (same uvw / weights / channels).
#
# --curv: the spectral model with explicit curvature (alpha, beta, delta) instead of the reference
#         run's model; the single-frequency stage uses its spatial part (exact frequency transition),
#         N_IT_MF_CURV multi-frequency iterations, linear sampling only (CURV_SAMPLING) and a separate
#         run directory (RUNS_CURV)
#
# usage: python make_configs.py [variants, e.g. A,B,C] [--curv]
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import (
    FAC, FOV_DEG, N_COARSE, N_IT_MF, N_IT_SF, OUT, REF_IDX, REF_MF, REF_SF, RUNS, SAMPLING, STEERING, SYNTH_DATA,
    VARIANTS, data_variant, noise_seed,
)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
CURV = "--curv" in sys.argv
variants = args[0].split(",") if args else ["A", "C"]
N_IT_MF_CURV = 20  # the image test (A_img_sfmf) needed ~20 MF iterations after the transition to converge
RUNS_CURV = f"{OUT}/curv_sf{N_IT_SF}_mf{N_IT_MF_CURV}"
CURV_SAMPLING = dict(n_samples=4, sample_mode="linear_resample")  # as the real multi-frequency run (mf_ext.yml)
CURVATURE = dict(mean=[0.0, 0.5], fluctuations=[1.0, 0.5])


def load_sections(fn):
    with open(fn) as f:
        return {k: v for d in yaml.safe_load_all(f) if d for k, v in d.items()}


ref_sf, ref_mf = load_sections(f"{REF_SF}/opt.yml"), load_sections(f"{REF_MF}/opt.yml")
base_opt = load_sections(f"{STEERING}/nifty/config/mf_ext.yml")["base_opt.6"]
prep = np.load(f"{OUT}/prep/prep.npz")
nu_eff, bounds = prep["nu_eff"], prep["bounds"]

grid = dict(space=[N_COARSE, N_COARSE], factor=FAC, fov=[f"{FOV_DEG}deg", f"{FOV_DEG}deg"])
noise = lambda prefix: dict(parameters=dict(mode=1.0, mean=1.1), scaling=True, varcov=False, prefix=prefix)  # noqa: E731
kernels = {
    "sf": ("runs/kernels/pk_synth05_986-1137mhz_1f_768.pkl", "runs/kernels/nk_synth05_986-1137mhz_1f_768.pkl"),
    "mf": ("runs/kernels/pk_synth05_961-1503mhz_6f_eff_768.pkl", "runs/kernels/nk_synth05_961-1503mhz_6f_eff_768.pkl"),
}

for v in variants:
    truth = np.load(f"{OUT}/truth/{VARIANTS[v][0]}/truth.npz")
    # prior mean of the log-brightness, as get_offset does it in the reference pipeline (mean log of the sky image)
    offset = round(float(np.mean(np.log(truth["i0"]))), 1)
    odir = os.path.relpath(f"{RUNS_CURV if CURV else RUNS}/{v}", STEERING)
    os.makedirs(f"{STEERING}/{odir}", exist_ok=True)
    params_mf = dict(ref_mf["sky_bg.6"]["params"], ref_freq_index=REF_IDX)
    if CURV:
        params_mf["spectral_curvature"] = CURVATURE
        params_sf = {k: params_mf[k] for k in ("zero_mode", "spatial_amplitude")}
        sampling, n_it_mf = CURV_SAMPLING, N_IT_MF_CURV
    else:
        params_sf = ref_sf["sky_bg.5"]["params"]
        sampling, n_it_mf = SAMPLING.get(v, {}), N_IT_MF

    head = dict(opt=dict(odir=odir, key=42, **(dict(kl_map="smap") if CURV else {})))
    stage_sf = {
        # stopped around reduced chi^2 ~ 1 (more iterations overfit, see synth_common)
        "opt.0": dict(base="base_opt_synth", likelihood="=lh.0", n_major_iterations=N_IT_SF),
        "lh.0": dict(sky="=sky.0", data="=data.0", noise=noise("nm.0"), split=0,
                     psf_kernel_fn=kernels["sf"][0], n_inv_kernel_fn=kernels["sf"][1], mode="fast"),
        # B differs from A only in the multi-frequency data (binned spectrum): same single-frequency data as A
        "data.0": dict(fname=f"{SYNTH_DATA}/eso_986-1137mhz_synth_{data_variant(v, 'sf')}_s{noise_seed(v)}.npz", mode="radio"),
        "sky.0": dict(prefix="sky.0", background="=sky_bg.0"),
        "sky_bg.0": dict(prefix="bg.0", grid=grid, freq=[1.0], params=params_sf, offset=offset),
        "base_opt_synth": {**base_opt, **sampling},
    }
    stage_mf = {
        "opt.1": dict(base="base_opt_synth", likelihood="=lh.1", transitions="=trans.1", n_major_iterations=n_it_mf),
        "lh.1": dict(sky="=sky.1", data="=data.1", noise=noise("nm.1"), split=0,
                     psf_kernel_fn=kernels["mf"][0], n_inv_kernel_fn=kernels["mf"][1], mode="fast",
                     pb_correction=True, freq_bounds=[float(b) for b in bounds]),
        "data.1": dict(fname=f"{SYNTH_DATA}/eso_961-1503mhz_synth_{data_variant(v, 'mf')}_s{noise_seed(v)}.npz", mode="radio"),
        "sky.1": dict(prefix="sky.1", background="=sky_bg.1"),
        "sky_bg.1": dict(prefix="bg.1", grid=grid, freq=[float(f) for f in nu_eff],
                         params=params_mf, offset=offset),
        "trans.1": dict(lh_old="=lh.0", lh_new="=lh.1", mode="freq"),
    }
    fn = f"{STEERING}/{odir}/config.yml"
    with open(fn, "w") as f:
        yaml.safe_dump_all([head, stage_sf, stage_mf], f, sort_keys=False, default_flow_style=None, width=120)
    print(f"saved: {fn} (offset {offset})")

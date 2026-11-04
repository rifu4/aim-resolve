# %%
# ---------------------------------------------------------------------------
# Run configs of the 1 deg synthetic test (truth make_truth_1deg.py): background + objects
# O0 (ESO 137-006) and O1 (ESO 137-007) on the 1 deg field, 10 % row subset of the data
# (make_rows_subset.py, make_vis.py --rows 10 --field 1deg).
#   stage 0  single-frequency, all three components, N_IT_SF iterations, started from the
#            image-space fit of the components to their truth images (prefit_comp.py
#            --truth-comps, which writes last.pkl; the config resumes from it)
#   stage 1  transition `freq` (exact), multi-frequency with the 8 frequencies / fixed bins of
#            nifty/config/mf_ext_8f.yml and the primary beam, N_IT_MF iterations
# Model: grids, offsets and windows of O0 / O1 and the priors as in the reference run
# REF_MF (4_rec_3z_1_6f_1), the spectral model with explicit curvature and the narrower
# beta prior CURVATURE. Optimiser: base_opt.6 of mf_ext.yml with 4 samples, the
# multi-frequency schedule of the image test (50 KL steps from the start, then 100-200;
# sampling CG 200 -> 500 -> 1000) and nonlinear resampling from the first iteration (SN settings of
# base_opt.6, as the multi-frequency stage of the reference run).
#
# usage: python make_config_1deg.py [variants, default A1deg]
# Output: OUT_1DEG/r10_sf<N_IT_SF>_mf<N_IT_MF>/<variant>/config.yml
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import FAC, N_COARSE_1DEG, OUT_1DEG, REF_IDX, REF_MF, STEERING, SYNTH_DATA, VARIANTS, data_variant, noise_seed, truth_dir

variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A1deg"]
N_IT_SF, N_IT_MF = 5, 15
RUNS = f"{OUT_1DEG}/r10_sf{N_IT_SF}_mf{N_IT_MF}"
CURVATURE = dict(mean=[0.0, 0.3], fluctuations=[0.3, 0.15])  # narrower than 0 +- 0.5 / 1 +- 0.5 (synthetic tests)
OBJECTS = ("o0", "o1")
KERNELS = {
    "sf": ("runs/kernels/pk_synth1d_r10_986-1137mhz_1f_1536.pkl", "runs/kernels/nk_synth1d_r10_986-1137mhz_1f_1536.pkl"),
    "mf": ("runs/kernels/pk_synth1d_r10_961-1503mhz_8f_eff_1536.pkl", "runs/kernels/nk_synth1d_r10_961-1503mhz_8f_eff_1536.pkl"),
}
SCHEDULE = dict(kl=dict(miniter="5*[50] + [100]", maxiter="5*[50] + [200]"),
                sl=dict(miniter="2*[200] + 3*[500] + [1000]", maxiter="2*[200] + 3*[500] + [1000]"))


def load_sections(fn):
    with open(fn) as f:
        return {k: v for d in yaml.safe_load_all(f) if d for k, v in d.items()}


ref = load_sections(f"{REF_MF}/opt.yml")
base_yml = load_sections(f"{STEERING}/nifty/config/base.yml")
mf8 = load_sections(f"{STEERING}/nifty/config/mf_ext_8f.yml")
nu_mf, bounds = [float(f) for f in mf8["freq"]], [float(b) for b in mf8["lh.6"]["freq_bounds"]]
base = load_sections(f"{STEERING}/nifty/config/mf_ext.yml")["base_opt.6"]
base_opt = dict(base, n_samples=4, sample_mode="nonlinear_resample",
                kl_kwargs=dict(base["kl_kwargs"], minimize_kwargs=dict(base["kl_kwargs"]["minimize_kwargs"], **SCHEDULE["kl"])),
                draw_linear_kwargs=dict(base["draw_linear_kwargs"], cg_kwargs=dict(base["draw_linear_kwargs"]["cg_kwargs"], **SCHEDULE["sl"])))
base_opt.pop("n_major_iterations", None)
grid = dict(space=[N_COARSE_1DEG, N_COARSE_1DEG], factor=FAC, fov=["1deg", "1deg"])
noise = lambda prefix: dict(parameters=dict(mode=1.0, mean=1.1), scaling=True, varcov=False, prefix=prefix)  # noqa: E731


def params(sec, mf):
    p = ref[sec]["params"]
    if mf:
        return dict(p, ref_freq_index=REF_IDX, spectral_curvature=CURVATURE)
    return {k: p[k] for k in ("zero_mode", "spatial_amplitude")}


for v in variants:
    truth = np.load(f"{truth_dir(VARIANTS[v][0])}/truth.npz")
    offset = round(float(np.mean(np.log(truth["i0"]))), 1)  # mean log brightness of the truth (as get_offset)
    odir = os.path.relpath(f"{RUNS}/{v}", STEERING)
    os.makedirs(f"{STEERING}/{odir}", exist_ok=True)
    docs = [dict(opt=dict(odir=odir, key=42, kl_map="smap", resume=True))]
    for i, (mf, stage, n_it) in enumerate([(False, "sf", N_IT_SF), (True, "mf", N_IT_MF)]):
        freq = nu_mf if mf else [1.0]
        data = (f"{SYNTH_DATA}/eso_961-1503mhz_rows10pct_w_synth_{data_variant(v, 'mf')}_s{noise_seed(v)}.npz" if mf else
                f"{SYNTH_DATA}/eso_986-1137mhz_rows10pct_w_synth_{data_variant(v, 'sf')}_s{noise_seed(v)}.npz")
        lh = dict(sky=f"=sky.{i}", data=f"=data.{i}", noise=noise(f"nm.{i}"), split=0, psf_kernel_fn=KERNELS[stage][0],
                  n_inv_kernel_fn=KERNELS[stage][1], mode="fast")
        if mf:
            lh.update(pb_correction=True, freq_bounds=bounds)
        sec = {
            f"opt.{i}": dict(base="base_opt_1deg", likelihood=f"=lh.{i}", n_major_iterations=n_it,
                             **(dict(transitions=f"=trans.{i}") if mf else {})),
            f"lh.{i}": lh,
            f"data.{i}": dict(fname=data, mode="radio"),
            f"sky.{i}": dict(prefix=f"sky.{i}", background=f"=sky_bg.{i}", **{f"{k}.{i}": f"=sky_{k}.{i}" for k in OBJECTS}),
            f"sky_bg.{i}": dict(prefix=f"bg.{i}", grid=grid, freq=freq, params=params("sky_bg.6", mf), offset=offset),
        }
        for k in OBJECTS:
            r = ref[f"sky_{k}.6"]
            sec[f"sky_{k}.{i}"] = dict(prefix=f"{k}.{i}", grid=r["grid"], freq=freq, params=params(f"sky_{k}.6", mf),
                                       offset=r["offset"], gaussian=r["gaussian"])
        if mf:
            sec[f"trans.{i}"] = dict(lh_old=f"=lh.{i - 1}", lh_new=f"=lh.{i}", mode="freq")
        else:
            sec["base_opt_1deg"] = base_opt
        docs.append(sec)
    fn = f"{STEERING}/{odir}/config.yml"
    with open(fn, "w") as f:
        yaml.safe_dump_all(docs, f, sort_keys=False, default_flow_style=None, width=120)
    print(f"saved: {fn} (background offset {offset})")

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
# --comp: (implies --curv) components of the reference run inside the field (make_truth_comp.py, variants
#         Acomp / Ccomp), built up as in the reference pipeline: stage 0 background only (N_IT_BG_COMP
#         single-frequency iterations), transition `addt` (each component fitted on its own to the stage-0
#         reconstruction, separated with the reference run's masks 4_msk.npz cut to the field; base_trans),
#         stage 1 all components (N_IT_SF_COMP single-frequency iterations), transition `freq`, stage 2
#         multi-frequency with the 8 frequencies / fixed bins of nifty/config/mf_ext_8f.yml and the schedule
#         of the image test A_img_sf15_mfsched. Objects o0 / o2, the tiles
#         and point sources fully inside, with the reference run's geometry, priors, offsets and windows;
#         run directory RUNS_COMP
#
# --comp --imgfit: instead of the background stage and `addt` inside the run, the components start from a fit
#         in image space to the truth image (prefit_comp.py, addt procedure with the reference masks); stage 0
#         single-frequency with all components (N_IT_SF_IMGFIT iterations), transition `freq`, stage 1
#         multi-frequency (N_IT_MF_IMGFIT iterations), both with the multi-frequency schedule; run directory
#         RUNS_IMGFIT (prefit_comp.py writes the start position as last.pkl, the configs resume from it)
#
# --8f: (implies --curv) the single-component runs with the 8 model frequencies / fixed bins of
#         nifty/config/mf_ext_8f.yml, N_IT_SF single-frequency and N_IT_MF_8F multi-frequency iterations with
#         the multi-frequency schedule of the image test; run directory RUNS_8F
#
# --rows10: the 10 % row subset of the data (make_rows_subset.py, make_vis.py --rows 10; weights x 10, same
#         S/N per bin, same binning error per uv scale), its own kernels and run directories (prefix r10_)
#
# usage: python make_configs.py [variants, e.g. A,B,C] [--curv] [--8f] [--comp [--imgfit]] [--rows10]
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import (
    FAC, FOV_DEG, N_COARSE, N_IT_MF, N_IT_SF, OUT, REF_IDX, REF_MF, REF_SF, RUNS, SAMPLING, STEERING, SYNTH_DATA,
    VARIANTS, OUT_COMP, data_variant, noise_seed, truth_dir,
)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
COMP = "--comp" in sys.argv
EIGHT = "--8f" in sys.argv and not COMP
ROWS10 = "--rows10" in sys.argv
DTAG, KTAG, RTAG = ("_rows10pct_w", "synth05r10", "r10_") if ROWS10 else ("", "synth05", "")
IMGFIT = "--imgfit" in sys.argv
CURV = "--curv" in sys.argv or COMP or EIGHT
variants = args[0].split(",") if args else ["A", "C"]
N_IT_MF_CURV = 20  # the image test (A_img_sfmf) needed ~20 MF iterations after the transition to converge
RUNS_CURV = f"{OUT}/curv_sf{N_IT_SF}_mf{N_IT_MF_CURV}"
N_IT_MF_8F = 15  # with the new multi-frequency schedule (chi^2 ~ 1 after ~12 iterations in the component runs)
RUNS_8F = f"{OUT}/curv8f_sf{N_IT_SF}_mf{N_IT_MF_8F}"
CURV_SAMPLING = dict(n_samples=4, sample_mode="linear_resample")  # as the real multi-frequency run (mf_ext.yml)
CURVATURE = dict(mean=[0.0, 0.5], fluctuations=[1.0, 0.5])
POINT_BETA = dict(mean=0.0, std=0.5)
N_IT_BG_COMP = 10  # background-only single-frequency stage before the components are added
N_IT_SF_COMP = 15  # single-frequency stage with all components (image test A_img_sf15_mfsched)
REF_MASK = f"{REF_MF}/../../files/4_msk.npz"
N_IT_SF_IMGFIT, N_IT_MF_IMGFIT = 5, 15  # after the image-space fit of the components (chi^2 ~ 1 after ~5 SF iterations)
RUNS_IMGFIT = f"{OUT_COMP}/imgfit_sf{N_IT_SF_IMGFIT}_mf{N_IT_MF_IMGFIT}"  # component masks of the reference run (last stage before the zoom)
RUNS_COMP = f"{OUT_COMP}/sf{N_IT_SF_COMP}_mf{N_IT_MF_CURV}"
# multi-frequency schedule of the image test A_img_sf15_mfsched: 50 KL steps from the first iteration, 100-200
# from the sixth; sampling CG ramped 200 -> 500 -> 1000 (the single-frequency stage keeps the base ramp)
MF_SCHEDULE = dict(kl=dict(miniter="5*[50] + [100]", maxiter="5*[50] + [200]"),
                   sl=dict(miniter="2*[200] + 3*[500] + [1000]", maxiter="2*[200] + 3*[500] + [1000]"))


def load_sections(fn):
    with open(fn) as f:
        return {k: v for d in yaml.safe_load_all(f) if d for k, v in d.items()}


ref_sf, ref_mf = load_sections(f"{REF_SF}/opt.yml"), load_sections(f"{REF_MF}/opt.yml")
base_opt = load_sections(f"{STEERING}/nifty/config/mf_ext.yml")["base_opt.6"]
prep = np.load(f"{OUT}/prep/prep.npz")
nu_eff, bounds = prep["nu_eff"], prep["bounds"]
if COMP or EIGHT:  # the 8 model frequencies and fixed bins of the real multi-frequency run (nifty/config/mf_ext_8f.yml)
    mf8 = load_sections(f"{STEERING}/nifty/config/mf_ext_8f.yml")
    nu_eff, bounds = np.asarray(mf8["freq"], dtype="float64"), np.asarray(mf8["lh.6"]["freq_bounds"], dtype="float64")

grid = dict(space=[N_COARSE, N_COARSE], factor=FAC, fov=[f"{FOV_DEG}deg", f"{FOV_DEG}deg"])
noise = lambda prefix: dict(parameters=dict(mode=1.0, mean=1.1), scaling=True, varcov=False, prefix=prefix)  # noqa: E731
kernels = {
    "sf": (f"runs/kernels/pk_{KTAG}_986-1137mhz_1f_768.pkl", f"runs/kernels/nk_{KTAG}_986-1137mhz_1f_768.pkl"),
    "mf": ("runs/kernels/pk_synth05_961-1503mhz_6f_eff_768.pkl", "runs/kernels/nk_synth05_961-1503mhz_6f_eff_768.pkl"),
}
if COMP or EIGHT:
    kernels["mf"] = (f"runs/kernels/pk_{KTAG}_961-1503mhz_8f_eff_768.pkl", f"runs/kernels/nk_{KTAG}_961-1503mhz_8f_eff_768.pkl")


def add_components(stage, i, truth_name, mf):
    """Component sections of the reference run (sky_o0.6, sky_o2.6, sky_t0.6, sky_p0.6) restricted to the
    components inside the field (components.yml of make_truth_comp.py) for stage `i` (single- or
    multi-frequency)."""
    geo = yaml.safe_load(open(f"{truth_dir(truth_name)}/components.yml"))
    sub_grid = dict(space=[N_COARSE, N_COARSE])  # placement grid of tiles / points (the reference uses its bg grid)
    freq = [float(f) for f in nu_eff] if mf else [1.0]
    sky = stage[f"sky.{i}"]
    for k in geo["objects"]:
        ref = ref_mf[f"sky_{k}.6"]
        p = dict(ref["params"], ref_freq_index=REF_IDX, spectral_curvature=CURVATURE) if mf else \
            {kk: ref["params"][kk] for kk in ("zero_mode", "spatial_amplitude")}
        stage[f"sky_{k}.{i}"] = dict(prefix=f"{k}.{i}", grid=dict(ref["grid"], **geo["objects"][k]), freq=freq,
                                     params=p, offset=ref["offset"], gaussian=ref["gaussian"])
        sky[f"{k}.{i}"] = f"=sky_{k}.{i}"
    ref = ref_mf["sky_t0.6"]
    idx = [next(n for n, c in enumerate(ref["tile_grid"]["center"]) if c == tc) for tc in geo["tile_centers"]]
    p = dict(ref["params"], ref_freq_index=REF_IDX, spectral_curvature=CURVATURE) if mf else \
        {kk: ref["params"][kk] for kk in ("zero_mode", "spatial_amplitude")}
    stage[f"sky_t0.{i}"] = dict(prefix=f"t0.{i}", tile_grid=dict(ref["tile_grid"], center=geo["tile_centers"],
                                n_copies=len(idx)), grid=sub_grid, freq=freq, params=p,
                                offset=[ref["offset"][n] for n in idx], gaussian=ref["gaussian"])
    sky[f"t0.{i}"] = f"=sky_t0.{i}"
    ref = ref_mf["sky_p0.6"]
    coords = [list(map(float, c)) for c in geo["point_coordinates"]]
    idx = [int(np.argmin(np.abs(np.array(ref["point_grid"]["coordinates"]) - c).max(axis=1))) for c in coords]
    p = dict(ref["params"], ref_freq_index=REF_IDX, beta=POINT_BETA) if mf else ref_sf["sky_p0.5"]["params"]
    stage[f"sky_p0.{i}"] = dict(prefix=f"p0.{i}", point_grid=dict(ref["point_grid"], coordinates=coords,
                                n_copies=len(coords)), grid=sub_grid, freq=freq, params=p,
                                offset=[ref["offset"][n] for n in idx])
    sky[f"p0.{i}"] = f"=sky_p0.{i}"


def comp_masks(truth_name, i, fn):
    """Masks for the `addt` transition into stage `i`: the reference run's component masks (REF_MASK, coarse
    1024^2 grid of the 2 deg field, one mask per point source / tile) cut to the field, refined to the fine
    grid and restricted to the components inside; one point mask per sub-pixel (the model's point grid);
    the background is the rest, as in masks_from_model."""
    geo = yaml.safe_load(open(f"{truth_dir(truth_name)}/components.yml"))
    ref = np.load(REF_MASK)
    n_ref = ref["sum"].shape[0]
    lo = (n_ref - N_COARSE) // 2
    fine = lambda m: np.kron(np.asarray(m, dtype="float64")[lo:lo + N_COARSE, lo:lo + N_COARSE], np.ones((FAC, FAC)))  # noqa: E731
    it_ref = next(k for k in ref.files if k.startswith("bg.")).split(".")[1]
    out = {}
    for k in geo["objects"]:
        out[f"{k}.{i}"] = fine(ref[f"{k}.{it_ref}"])
    centers = ref_mf["sky_t0.6"]["tile_grid"]["center"]
    tiles = ref[f"t0.{it_ref}"]
    out[f"t0.{i}"] = np.stack([fine(tiles[centers.index(c)]) for c in geo["tile_centers"]])
    pts = ref[f"p0.{it_ref}"]
    names = [p["name"] for p in geo["points"]]
    out[f"p0.{i}"] = np.stack([fine(pts[int(n.split("_")[1])]) for n in names for _ in range(FAC * FAC)])
    out["sum"] = np.sum([v.sum(axis=0) if v.ndim == 3 else v for v in out.values()], axis=0)
    out[f"bg.{i}"] = np.floor(1 - out["sum"]).clip(0, 1)
    # each point mask has to contain its source (checks the source order of the reference masks)
    for n, p in enumerate(geo["points"]):
        assert out[f"p0.{i}"][n * FAC * FAC][p["i"], p["j"]] > 0, p["name"]
    np.savez(fn, **out)
    return fn


def renumber(stage, old, new):
    """Copy of a stage's sections with the stage index `old` replaced by `new` in section names, prefixes and
    references (strings such as 'sky_bg.1', '=lh.1', 'nm.1'); numbers are left untouched. The previous stage
    (old - 1) in references is shifted as well."""
    import re

    pat = re.compile(rf"^(=?[A-Za-z_][A-Za-z0-9_]*)\.({old}|{old - 1})$")

    def sub(x):
        if isinstance(x, dict):
            return {sub(k): sub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [sub(v) for v in x]
        if isinstance(x, str):
            m = pat.match(x)
            if m:
                return f"{m.group(1)}.{new if int(m.group(2)) == old else new - 1}"
        return x

    return sub(stage)


def comp_stage_sf(stage_bg, v, odir):
    """Stage 1: single-frequency reconstruction with all components, entered with the `addt` transition from
    the background-only stage 0 (components fitted on their own, separated by the reference masks)."""
    st = renumber({k: val for k, val in stage_bg.items() if k != "base_opt_synth"}, 0, 1)
    st["opt.1"] = dict(base="base_opt_synth", likelihood="=lh.1", transitions="=trans.1", n_major_iterations=N_IT_SF_COMP)
    mask_fn = comp_masks(VARIANTS[v][0], 1, f"{STEERING}/{odir}/masks_addt.npz")
    # the component fits use base_trans of nifty/config/base.yml, written out as in the reference run's trans.5
    # (a '=section' reference would be instantiated as an object)
    opt_dct = load_sections(f"{STEERING}/nifty/config/base.yml")["base_trans"]
    st["trans.1"] = dict(lh_old="=lh.0", lh_new="=lh.1", mode="addt", opt_dct=opt_dct, offsets=False,
                         mask=os.path.relpath(mask_fn, STEERING), odir=f"{odir}/trans")
    add_components(st, 1, VARIANTS[v][0], mf=False)
    return st

for v in variants:
    truth = np.load(f"{truth_dir(VARIANTS[v][0])}/truth.npz")
    # prior mean of the log-brightness, as get_offset does it in the reference pipeline (mean log of the sky image)
    offset = round(float(np.mean(np.log(truth["i0"]))), 1)
    runs = (RUNS_IMGFIT if IMGFIT else RUNS_COMP) if COMP else RUNS_8F if EIGHT else RUNS_CURV if CURV else RUNS
    runs = f"{os.path.dirname(runs)}/{RTAG}{os.path.basename(runs)}"
    odir = os.path.relpath(f"{runs}/{v}", STEERING)
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
        "data.0": dict(fname=f"{SYNTH_DATA}/eso_986-1137mhz{DTAG}_synth_{data_variant(v, 'sf')}_s{noise_seed(v)}.npz", mode="radio"),
        "sky.0": dict(prefix="sky.0", background="=sky_bg.0"),
        "sky_bg.0": dict(prefix="bg.0", grid=grid, freq=[1.0], params=params_sf, offset=offset),
        "base_opt_synth": {**base_opt, **sampling},
    }
    if COMP:
        stage_sf["opt.0"]["n_major_iterations"] = N_IT_BG_COMP
    stage_mf = {
        "opt.1": dict(base="base_opt_synth", likelihood="=lh.1", transitions="=trans.1", n_major_iterations=n_it_mf),
        "lh.1": dict(sky="=sky.1", data="=data.1", noise=noise("nm.1"), split=0,
                     psf_kernel_fn=kernels["mf"][0], n_inv_kernel_fn=kernels["mf"][1], mode="fast",
                     pb_correction=True, freq_bounds=[float(b) for b in bounds]),
        "data.1": dict(fname=f"{SYNTH_DATA}/eso_961-1503mhz{DTAG}_synth_{data_variant(v, 'mf')}_s{noise_seed(v)}.npz", mode="radio"),
        "sky.1": dict(prefix="sky.1", background="=sky_bg.1"),
        "sky_bg.1": dict(prefix="bg.1", grid=grid, freq=[float(f) for f in nu_eff],
                         params=params_mf, offset=offset),
        "trans.1": dict(lh_old="=lh.0", lh_new="=lh.1", mode="freq"),
    }
    docs = [head, stage_sf, stage_mf]
    if EIGHT:
        stage_mf["opt.1"]["n_major_iterations"] = N_IT_MF_8F
        base = stage_sf["base_opt_synth"]
        stage_mf["base_opt_synth_mf"] = dict(base, kl_kwargs=dict(base["kl_kwargs"], minimize_kwargs=dict(
            base["kl_kwargs"]["minimize_kwargs"], **MF_SCHEDULE["kl"])), draw_linear_kwargs=dict(
            base["draw_linear_kwargs"], cg_kwargs=dict(base["draw_linear_kwargs"]["cg_kwargs"], **MF_SCHEDULE["sl"])))
        stage_mf["opt.1"]["base"] = "base_opt_synth_mf"
    if COMP and IMGFIT:
        add_components(stage_sf, 0, VARIANTS[v][0], mf=False)
        add_components(stage_mf, 1, VARIANTS[v][0], mf=True)
        comp_masks(VARIANTS[v][0], 0, f"{STEERING}/{odir}/masks_prefit.npz")
        stage_sf["opt.0"]["n_major_iterations"] = N_IT_SF_IMGFIT
        stage_mf["opt.1"]["n_major_iterations"] = N_IT_MF_IMGFIT
        base = stage_sf["base_opt_synth"]
        # the start is close to the solution: the stronger multi-frequency schedule for both stages
        stage_sf["base_opt_synth"] = dict(base, kl_kwargs=dict(base["kl_kwargs"], minimize_kwargs=dict(
            base["kl_kwargs"]["minimize_kwargs"], **MF_SCHEDULE["kl"])), draw_linear_kwargs=dict(
            base["draw_linear_kwargs"], cg_kwargs=dict(base["draw_linear_kwargs"]["cg_kwargs"], **MF_SCHEDULE["sl"])))
        head["opt"]["resume"] = True  # start position written by prefit_comp.py
    elif COMP:
        docs = [head, stage_sf, comp_stage_sf(stage_sf, v, odir), renumber(stage_mf, 1, 2)]
        add_components(docs[3], 2, VARIANTS[v][0], mf=True)
        base = stage_sf["base_opt_synth"]
        docs[3]["base_opt_synth_mf"] = dict(base, kl_kwargs=dict(base["kl_kwargs"], minimize_kwargs=dict(
            base["kl_kwargs"]["minimize_kwargs"], **MF_SCHEDULE["kl"])), draw_linear_kwargs=dict(
            base["draw_linear_kwargs"], cg_kwargs=dict(base["draw_linear_kwargs"]["cg_kwargs"], **MF_SCHEDULE["sl"])))
        docs[3]["opt.2"]["base"] = "base_opt_synth_mf"
    fn = f"{STEERING}/{odir}/config.yml"
    with open(fn, "w") as f:
        yaml.safe_dump_all(docs, f, sort_keys=False, default_flow_style=None, width=120)
    print(f"saved: {fn} (offset {offset})")

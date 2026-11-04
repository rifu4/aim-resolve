# %%
# ---------------------------------------------------------------------------
# Start position of the component runs (make_configs.py --comp --imgfit): the
# components of stage 0 are fitted in image space to the truth image, following
# transition_addt (and steering/scripts/pre_fit.py): the background on the
# masked image, then each point / object / tile component on its masked region
# of the remainder, then the whole sky jointly; masks: the reference run's
# masks cut to the field (masks_prefit.npz), fit settings: base_trans of
# nifty/config/base.yml with linear sampling only, offsets: those of the config (reference run).
#
# Truth image: the truth I0 at the weighted mean frequency of the single-
# frequency data (stage 0 is single-frequency). It does not depend on beta, so
# one fit serves all variants: the start position is written to every run
# directory as last.pkl (resume format), with an empty minisanity.txt.
#
# --truth-comps: instead of the masked fits to the total image, each component is fitted to its own truth
#        component (BG, O0, O2, tiles, points of make_truth_comp.py) at the same frequency, so the start has the
#        truth's decomposition (the masked background fit is free inside the large O0 mask and took over part
#        of the galaxy). No masks are needed.
#
# --get-offsets (with --truth-comps): the offset of every component from its own truth image with
#        modeling.get_offset (log of the mean brightness over the component's grid, no mask), as pre_fit.py
#        does it, set before the fit and written into the run configs (all stages of the component)
#
# usage: python prefit_comp.py [--n-it N] [--sample-mode MODE] [--truth-comps [--get-offsets]] RUN_DIR [RUN_DIR ...]   (first run's config is used; N iterations per
#        component fit, default 30: with base_trans' 20 the extended emission of O0 was not yet fitted and the
#        background took it over in the first single-frequency iteration)
# ---------------------------------------------------------------------------
import os
import pickle
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import DATA_SF, STEERING, VARIANTS, TruthSky, truth_dir

import aim_resolve as aim
from aim_resolve.fast_resolve.fast_kl import OptimizeVIState
from aim_resolve.mask import masks_to_boxes
from aim_resolve.modeling import get_offset
from aim_resolve.model.map import map_signal
from aim_resolve.fast_resolve.beam import build_primary_beam
from aim_resolve.optimize.samples import MySamples, domain_tree, random_init
from aim_resolve.transition import optimize_and_plot
from nifty.re import Model, Vector

jax.config.update("jax_enable_x64", True)

args = sys.argv[1:]
n_it = 30
SAMPLE_MODE = "linear_resample"
if "--sample-mode" in args:
    i = args.index("--sample-mode")
    SAMPLE_MODE = args[i + 1]
    del args[i:i + 2]
if "--n-it" in args:
    i = args.index("--n-it")
    n_it = int(args[i + 1])
    del args[i:i + 2]
APPARENT = "--apparent" in args  # fit the truth times the primary beam, the sky a stage without a beam fits
if APPARENT:
    args.remove("--apparent")
TRUTH_COMPS = "--truth-comps" in args
GET_OFFSETS = "--get-offsets" in args
args = [a for a in args if a not in ("--truth-comps", "--get-offsets")]
runs = [os.path.abspath(r) for r in args]
os.chdir(STEERING)
cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{runs[0]}/config.yml"), aim.get_builders)
lh = cfg.instantiate_sec("lh.0")
sky = lh["sky_model"]
# base_trans with linear sampling only (as the runs; the fit only provides the start position)
opt_dct = dict(aim.yaml_load(f"{STEERING}/nifty/config/base.yml")["base_trans"], sample_mode=SAMPLE_MODE,
               n_total_iterations=n_it)
print("component fits:", n_it, "iterations each, sample mode:", SAMPLE_MODE)
odir = f"{runs[0]}/prefit"
plot_dct = dict(norm="log", label=None, grid=None, odir=odir)
noise = dict(max_std=1e-5, parameters=dict())  # as transition_addt's default

# truth image at the weighted mean frequency of the single-frequency data [Jy/sr]
d = np.load(DATA_SF, allow_pickle=True)
nu_sf = float(np.sum(d["freq"] * d["weight"].sum(axis=(0, 1))) / d["weight"].sum())
variant = os.path.basename(runs[0])
rec = np.asarray(TruthSky(VARIANTS[variant][0])(nu_sf), dtype="float64")
pb_sf = np.asarray(build_primary_beam(sky.grid, [nu_sf], channel_average=False), dtype="float64").squeeze() if APPARENT else 1.0
rec = rec * pb_sf
print(f"truth image ({VARIANTS[variant][0]}) at {nu_sf / 1e6:.2f} MHz: {rec.shape}, max {rec.max():.3e} Jy/sr")

keys = list(jax.random.split(jax.random.PRNGKey(cfg.sections["opt"]["key"]), 2 + len(sky.models)))
ptree = {}


def truth_components():
    """Truth image of every component at nu_sf [Jy/sr], keyed by the component prefix."""
    t = np.load(f"{truth_dir(VARIANTS[variant][0])}/truth.npz")
    u = np.log(nu_sf / float(t["nu_ref"]))
    a, b = (t["alpha_diffuse"], t["beta_diffuse"]) if "alpha_diffuse" in t.files else (t["alpha"], t["beta"])
    spec_d = np.exp(a * u + b * u**2)
    keys = dict(bg="i0_bg", o0="i0_o0" if "i0_o0" in t.files else "i0_obj", o1="i0_o1", o2="i0_o2", t0="i0_tiles")
    img = {k: t[key] * spec_d for k, key in keys.items() if key in t.files}
    img |= {k: t[f"i0_{k}"] * np.exp(t[f"alpha_{k}"] * u + t[f"beta_{k}"] * u**2) for k in keys if f"alpha_{k}" in t.files}
    if "i0_pts" in t.files:
        img["p0"] = t["i0_pts"] * np.exp(t["alpha_pts"] * u + t["beta_pts"] * u**2)
    return {m.prefix: np.asarray(img[m.prefix.split(".")[0]], dtype="float64") * pb_sf for m in sky.models}


offsets = {}
if TRUTH_COMPS:
    imgs = truth_components()
    for comp in sky.models:
        data = imgs[comp.prefix] if comp.grid == sky.grid else map_signal(sky.grid, comp.grid)(imgs[comp.prefix])
        print(f"{comp.prefix}: truth component flux {imgs[comp.prefix].sum() / rec.sum():.4f} of the total")
        if GET_OFFSETS:
            d = np.asarray(data)
            offsets[comp.prefix] = get_offset(comp, d, np.ones(d.shape, dtype=bool), comp.freq)
            comp.set_offset(offsets[comp.prefix])
        pos_c = optimize_and_plot(key=keys.pop(), sky=comp, data=data, noise=noise.copy(), opt_dct=opt_dct,
                                  plot_dct=plot_dct | dict(name=f"prefit_{comp.prefix}.png"))
        ptree |= pos_c.tree
    for comp in sky.models:  # per-component check of the start
        f, t_ = np.asarray(comp(Vector(ptree))), imgs[comp.prefix]
        f = f if f.shape == t_.shape else np.asarray(map_signal(comp.grid, sky.grid)(f))  # objects: own grid
        print(f"  {comp.prefix}: fit / truth flux {f.sum() / max(t_.sum(), 1e-30):.4f}")
mask_dct = {} if TRUTH_COMPS else dict(np.load(f"{runs[0]}/masks_prefit.npz"))
mask_box = {} if TRUTH_COMPS else masks_to_boxes(sky, mask_dct)

if not TRUTH_COMPS:
    # background on the image with the component regions masked
    sky_bg = sky.background.copy()
    msk_bg = mask_box[sky_bg.prefix]
    sky_bg.mask = msk_bg
    pos_bg = optimize_and_plot(key=keys.pop(), sky=sky_bg, data=np.where(msk_bg, rec, 0.0), noise=noise.copy(),
                               opt_dct=opt_dct, plot_dct=plot_dct | dict(name=f"prefit_{sky_bg.prefix}.png"))
    ptree |= pos_bg.tree
    sky_bg.mask = None
    rec_sub = (rec - np.asarray(sky_bg(pos_bg))).clip(0, None)

    # each component on its masked region of the remainder
    for comp in sky.points + sky.objects + sky.tiles:
        sub = map_signal(sky.grid, comp.grid)(rec_sub)
        msk = mask_box[comp.prefix]
        if isinstance(comp, aim.PointModel | aim.TileModel):
            msk = msk.sum(axis=0).clip(0, 1)
        pos_c = optimize_and_plot(key=keys.pop(), sky=comp, data=np.where(msk, sub, 0.0), noise=noise.copy(),
                                  opt_dct=opt_dct, plot_dct=plot_dct | dict(name=f"prefit_{comp.prefix}.png"))
        ptree |= pos_c.tree

# whole sky jointly, starting from the component fits
pos_sky = optimize_and_plot(key=keys.pop(), sky=sky, data=rec, noise=noise.copy(), pos=Vector(ptree), opt_dct=None,
                            plot_dct=plot_dct | dict(name=f"prefit_{sky.prefix}.png"))
pos = random_init(keys.pop(), [v for v in lh.values() if isinstance(v, Model)], pos_sky.tree, factor=0.01)
fit = np.asarray(sky(pos))
print(f"prefit: total flux fit / truth {fit.sum() / rec.sum():.4f}, median |ln(fit / truth)| in the bright half "
      f"{np.median(np.abs(np.log(fit[rec > np.median(rec)] / rec[rec > np.median(rec)]))):.4f}")
print("parameters:", sorted({k.split(" ")[0] for k in domain_tree(pos)}))

if offsets:  # the run's priors get the same offsets (all stages of each component)
    import yaml

    for r in runs:
        docs = list(yaml.safe_load_all(open(f"{r}/config.yml")))
        for d in docs:
            for sec in d or {}:
                base = sec.split(".")[0]
                for pf, off in offsets.items():
                    if base == f"sky_{pf.split('.')[0]}" and "offset" in d[sec]:
                        d[sec]["offset"] = off
        with open(f"{r}/config.yml", "w") as f:
            yaml.safe_dump_all(docs, f, sort_keys=False, default_flow_style=None, width=120)
        print("offsets written to", f"{r}/config.yml:", {k.split(".")[0]: v for k, v in offsets.items()})

state = OptimizeVIState(nit=0, key=jax.random.PRNGKey(0), config={})
for r in runs:
    with open(f"{r}/last.pkl", "wb") as f:
        pickle.dump((MySamples(pos=pos, samples=None, keys=None), state, 0), f)
    open(f"{r}/minisanity.txt", "w").close()
    print("start position written:", f"{r}/last.pkl")

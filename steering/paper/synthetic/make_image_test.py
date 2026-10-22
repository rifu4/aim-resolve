# %%
# ---------------------------------------------------------------------------
# Image-domain test of the spectral model with curvature (alpha, beta, delta):
# the truth cube of the synthetic test ("main": ESO 137-006 + background with
# random alpha / beta) at the 6 effective model frequencies on the 0.5 deg
# field, plus independent Gaussian noise per pixel and frequency; no response
# and no primary beam. One correlated field with spectral_curvature
# (0 +- 0.5), 4 samples with linear sampling only, multi-frequency from the
# start (no single-frequency stage, no transition).
#
# With --sfmf: first a single-frequency stage (N_IT_SF iterations) on the truth
# image at nu_ref with the same noise level, then the frequency transition
# (mode: freq) and the multi-frequency stage as above (as the radio pipeline).
# The single-frequency stage uses the single-frequency ubik model (zero_mode,
# spatial_amplitude: the parametrisation of the multi-frequency model), so the
# transition carries the sky over exactly; --old-sf uses the old single-frequency
# prior model (i0: ... of bg.5) instead.
#
# usage: python make_image_test.py [--sfmf] [noise_jy_per_sr]
# Output: OUT/image_curv/data/truth_main_6f.pkl (ImageData, clean cube; the
# loader adds the noise), truth_main_1f.pkl (--sfmf) and
# OUT/image_curv/{A_img,A_img_sfmf}/config.yml
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import FAC, FOV_DEG, N_COARSE, OUT, REF_IDX, REF_MF, STEERING, TruthSky, field_grid

from aim_resolve.img_data.data import ImageData

SFMF = "--sfmf" in sys.argv
OLD_SF = "--old-sf" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
NOISE = float(_args[0]) if _args else 1e5  # Jy/sr per pixel and frequency
N_IT = 15
N_IT_SF = 10
AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2
root = f"{OUT}/image_curv"
os.makedirs(f"{root}/data", exist_ok=True)

prep = np.load(f"{OUT}/prep/prep.npz")
nu_eff = prep["nu_eff"]
truth = TruthSky("main")
cube = np.stack([truth(f) for f in nu_eff])  # (6, N, N), Jy/sr
max_std = NOISE / cube.max()  # image_data / image_likelihood: noise std = max_std * max(data)
print(f"cube max {cube.max():.3e} Jy/sr -> max_std {max_std:.3e} (noise {NOISE:.1e} Jy/sr); "
      f"S/N at 0.01 mJy/arcsec^2: {0.01 / CONV_FACTOR / NOISE:.1f}")
ImageData(cube, field_grid(), prefix="truth_main_6f").save("truth_main_6f", f"{root}/data")
img_sf = truth(nu_eff[REF_IDX])  # single-frequency image at nu_ref
max_std_sf = NOISE / img_sf.max()
if SFMF:
    ImageData(img_sf, field_grid(), prefix="truth_main_1f").save("truth_main_1f", f"{root}/data")


def load_sections(fn):
    with open(fn) as f:
        return {k: v for d in yaml.safe_load_all(f) if d for k, v in d.items()}


ref_mf = load_sections(f"{REF_MF}/opt.yml")
ref_sf = load_sections(f"{REF_MF}/../4_rec_3z_1/opt.yml")
base_opt = load_sections(f"{STEERING}/nifty/config/mf_ext.yml")["base_opt.6"]
base_opt = {**base_opt, "n_major_iterations": N_IT, "n_samples": 4, "sample_mode": "linear_resample"}
offset = round(float(np.mean(np.log(truth.i0))), 1)
params = dict(ref_mf["sky_bg.6"]["params"], ref_freq_index=REF_IDX,
              spectral_curvature=dict(mean=[0.0, 0.5], fluctuations=[1.0, 0.5]))
odir = os.path.relpath(f"{root}/{('A_img_sfmf_oldsf' if OLD_SF else 'A_img_sfmf') if SFMF else 'A_img'}", STEERING)
ddir = os.path.relpath(f"{root}/data", STEERING)
grid = dict(space=[N_COARSE, N_COARSE], factor=FAC, fov=[f"{FOV_DEG}deg"] * 2)


def mf_stage(i, transition=None):
    """Multi-frequency stage `i` (sections opt.i, lh.i, data.i, sky.i, sky_bg.i)."""
    opt = dict(base="base_opt_img", likelihood=f"=lh.{i}")
    if transition:
        opt.update(transitions=f"=trans.{i}", n_total_iterations=N_IT)
    else:
        opt.update(n_total_iterations=N_IT)
    sec = {
        f"opt.{i}": opt,
        f"lh.{i}": dict(sky=f"=sky.{i}", data=f"=data.{i}", noise=dict(max_std=float(max_std)), mode="image"),
        f"data.{i}": dict(fname="truth_main_6f.pkl", odir=ddir, mode="image", max_std=float(max_std), key=42),
        f"sky.{i}": dict(prefix=f"sky.{i}", background=f"=sky_bg.{i}"),
        f"sky_bg.{i}": dict(prefix=f"bg.{i}", grid=grid, freq=[float(f) for f in nu_eff], params=params, offset=offset),
        "base_opt_img": base_opt,
    }
    if transition:
        sec[f"trans.{i}"] = dict(lh_old=f"=lh.{i - 1}", lh_new=f"=lh.{i}", mode="freq")
    return sec


head = dict(opt=dict(odir=odir, key=42))
if SFMF:
    # single-frequency stage on the image at nu_ref: the spatial part of the multi-frequency
    # model (new) or the old single-frequency prior model of bg.5 (--old-sf)
    params_sf = (ref_sf["sky_bg.5"]["params"] if OLD_SF
                 else {k: params[k] for k in ("zero_mode", "spatial_amplitude")})
    stage_sf = {
        "opt.0": dict(base="base_opt_img", likelihood="=lh.0", n_total_iterations=N_IT_SF),
        "lh.0": dict(sky="=sky.0", data="=data.0", noise=dict(max_std=float(max_std_sf)), mode="image"),
        "data.0": dict(fname="truth_main_1f.pkl", odir=ddir, mode="image", max_std=float(max_std_sf), key=43),
        "sky.0": dict(prefix="sky.0", background="=sky_bg.0"),
        "sky_bg.0": dict(prefix="bg.0", grid=grid, freq=[1.0], params=params_sf, offset=offset),
        "base_opt_img": base_opt,
    }
    docs = [head, stage_sf, mf_stage(1, transition=True)]
else:
    docs = [head, mf_stage(0)]
os.makedirs(f"{STEERING}/{odir}", exist_ok=True)
fn = f"{STEERING}/{odir}/config.yml"
with open(fn, "w") as f:
    yaml.safe_dump_all(docs, f, sort_keys=False, default_flow_style=None, width=120)
print("saved:", f"{root}/data", fn)

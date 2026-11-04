"""Shared settings and helpers for the synthetic-data test (see ../synthetic_test_plan.md).

Version 1: a single extended component (ESO 137-006, object o0 of the reference
run) in a 0.5 deg x 0.5 deg field, reconstructed with a single correlated field.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
STEERING = "/scratch/users/rfuchs/packages/aim-resolve/steering"
REF_DIR = f"{STEERING}/runs/fast_vi_1f_1024_1z_b"
REF_MF = f"{REF_DIR}/opt/4_rec_3z_1_6f_1"  # multi-frequency reference run
REF_SF = f"{REF_DIR}/opt/4_rec_3z_1"  # single-frequency run it was extended from
DATA_MF = "/scratch/users/rfuchs/data/eso_961-1503mhz.npz"
DATA_SF = "/scratch/users/rfuchs/data/eso_986-1137mhz.npz"
SYNTH_DATA = "/scratch/users/rfuchs/data/synth"
OUT = f"{STEERING}/runs/synth_05deg"
OUT_COMP = f"{STEERING}/runs/synth_05deg_comp"  # version 2 (component structure): truths *_comp and their runs
OUT_1DEG = f"{STEERING}/runs/synth_1deg"  # version 3: 1 deg field, O0 + O1 + background (make_truth_1deg.py)
N_COARSE_1DEG = 512  # coarse pixels of the 1 deg field
# run set: 10 single-frequency + 15 multi-frequency iterations (the first set, with 15 + 30
# iterations, is in OUT/<variant>; it overfits: reduced chi^2 ~ 1 after ~6 SF / ~13 MF iterations)
N_IT_SF, N_IT_MF = 10, 15
RUN_SET = f"sf{N_IT_SF}_mf{N_IT_MF}"
RUNS = f"{OUT}/{RUN_SET}"
PREFIX = "eso137"
OBJECT = "O0"  # ESO 137-006 in the reference FITS

# ---------------------------------------------------------------------------
# Field: 0.5 deg x 0.5 deg around the phase centre, pixel size of the reference run.
# ---------------------------------------------------------------------------
FOV_DEG = 0.5
REF_N = 3072  # reference sky grid (fine pixels, 2 deg)
FAC = 3  # fine / coarse pixels
N_COARSE = 256  # coarse pixels of the field (7.03125 arcsec)
N = N_COARSE * FAC  # fine pixels of the field (2.34375 arcsec)
CROP = slice((REF_N - N) // 2, (REF_N + N) // 2)  # fine-pixel crop of the reference grid

# ---------------------------------------------------------------------------
# Truth spectra and seeds.
# ---------------------------------------------------------------------------
REF_IDX = 1
ALPHA_MEAN, ALPHA_STD = -1.0, 0.5
BETA_MEAN, BETA_STD = -0.2, 0.3
SMOOTH_ARCSEC = 45.0  # correlation length of the alpha / beta fields
SEED_TRUTH = 1
SEED_NOISE = 2

VARIANTS = {
    # name: (truth variant, spectrum per data channel)
    "A": ("main", "continuous"),  # each channel at its exact frequency, beam per channel
    "C": ("beta0", "continuous"),  # as A with beta = 0
    "B": ("main", "binned"),  # model-consistent: S and beam at the bin's model frequency
    "Ap": ("main", "continuous"),  # as A with another noise realisation (systematic vs noise)
    # version 2: truth and model with the component structure of the reference run (make_truth_comp.py)
    "Acomp": ("main_comp", "continuous"),
    "Ccomp": ("beta0_comp", "continuous"),
    "Bcomp": ("main_comp", "binned"),
    # version 3: 1 deg field (truths 1deg_main / 1deg_beta0 = OUT_1DEG/truth/main, beta0)
    "A1deg": ("1deg_main", "continuous"),
    "B1deg": ("1deg_main", "binned"),
    "C1deg": ("1deg_beta0", "continuous"),
    # low-resolution versions (make_lowres_1deg.py): factor 1 grids, uv-cut data, 3 x 3 averaged truth
    "A1deglr": ("1deg_main_lr", "continuous"),
    "A1deglr_u50": ("1deg_main_lr", "continuous"),  # uv cut at 0.5 x Nyquist instead of 0.95
    "A1deglr_u50wfix": ("1deg_main_lr", "continuous"),  # A1deglr_u50 with the weights rescaled per channel (w / s_c^2)
    "C1deglr_u50": ("1deg_beta0_lr", "continuous"),  # as A1deglr_u50 with the beta = 0 truth
    "S1deg": ("1deg_main_sep", "continuous"),  # only needed as the source name of S1deglr_u50
    "S1deglr_u50": ("1deg_main_sep_lr", "continuous"),  # as A1deglr_u50 with a spectrum per component (make_truth_sep.py)
    "BO1wfix": ("1deg_main_bo1", "continuous"),  # only needed as the source name of BO1wfixlr_u50
    "BO1wfixlr_u50": ("1deg_main_bo1_lr", "continuous"),  # background + O1 (no O0), weights rescaled per channel
    "S1wfix": ("1deg_main_sep", "continuous"),  # only needed as the source name of S1wfixlr_u50
    "S1wfixlr_u50": ("1deg_main_sep_lr", "continuous"),  # spectrum per component, weights rescaled per channel
    "T1wfix": ("1deg_main_bcl", "continuous"),  # only needed as the source name of T1wfixlr_u50
    "T1wfixlr_u50": ("1deg_main_bcl_lr", "continuous"),  # beta with its own correlation length, weights rescaled per channel
    "BO0wfix": ("1deg_main_bo0", "continuous"),  # only needed as the source name of BO0wfixlr_u50
    "BO0wfixlr_u50": ("1deg_main_bo0_lr", "continuous"),  # background + O0 (no O1), weights rescaled per channel
    "O0deg": ("1deg_main_o0", "continuous"),  # only needed as the source name of O0deglr_u50
    "O0deglr_u50": ("1deg_main_o0_lr", "continuous"),  # as A1deglr_u50 with O0 alone (no background, no O1)
    "O1deg": ("1deg_main_o1", "continuous"),  # only needed as the source name of O1deglr_u50
    "O1deglr_u50": ("1deg_main_o1_lr", "continuous"),  # as A1deglr_u50 with O1 alone (no background, no O0)
    "S1wfix2": ("1deg_main_sep", "continuous"),  # only needed as the source name of S1wfix2lr_u50
    "S1wfix2lr_u50": ("1deg_main_sep_lr", "continuous"),  # S1wfixlr_u50 with another noise realisation
    "A1nfwfix": ("1deg_main", "continuous"),  # only needed as the source name of A1nfwfixlr_u50
    "A1nfwfixlr_u50": ("1deg_main_lr", "continuous"),  # A1deglr_u50wfix without noise in the visibilities
}
NOISE_SEEDS = {"Ap": 3, "S1wfix2": 3, "S1wfix2lr_u50": 3}  # noise seed per variant, default SEED_NOISE
NOISELESS = {"A1nfwfix", "A1nfwfixlr_u50"}  # variants without noise (the weights stay)

# inference variants: same data as another variant, different sampling settings
DATA_OF = {"B4L": "B", "A1deglr_u50_kpsf": "A1deglr_u50", "A1deglr_u50_kpsf2n": "A1deglr_u50", "A1deglr_u50_kpsf2n_nonoise": "A1deglr_u50", "C1deglr_u50_kpsf2n": "C1deglr_u50", "O1deglr_u50_kpsf2n": "O1deglr_u50", "S1deglr_u50_kpsf2n": "S1deglr_u50", "O1deglr_u50_std": "O1deglr_u50", "O0deglr_u50_std": "O0deglr_u50", "O0deglr_u50_chres": "O0deglr_u50", "O0deglr_u50_sfonly": "O0deglr_u50", "O0deglr_u50_chres_nmreset": "O0deglr_u50", "A1wfix_fix7": "A1deglr_u50wfix", "A1wfix_fixbin": "A1deglr_u50wfix", "A1wfix_kfix": "A1deglr_u50wfix", "S1wfix_kfix": "S1wfixlr_u50", "S1wfix_fix7": "S1wfixlr_u50", "S1wfix2_fix7": "S1wfix2lr_u50", "S1wfix_rp": "S1wfixlr_u50", "S1wfix_bgoff": "S1wfixlr_u50", "S1wfix_rpfl": "S1wfixlr_u50", "S1wfix_rpfl_oldoff": "S1wfixlr_u50", "S1wfix_rpfl_t02": "S1wfixlr_u50", "S1wfix_rpfl5_t02": "S1wfixlr_u50", "A1wfix_sf_base": "A1deglr_u50wfix", "A1wfix_sf_app": "A1deglr_u50wfix", "A1wfix_sf_pb": "A1deglr_u50wfix", "T1wfix_kfix": "T1wfixlr_u50", "BO0wfix_kfix": "BO0wfixlr_u50"}  # variant -> variant whose synthetic data it uses
VARIANTS["B4L"] = VARIANTS["B"]
VARIANTS["A1deglr_u50_kpsf"] = VARIANTS["A1deglr_u50"]  # psf_from_n_inv (noise kernel as PSF)
VARIANTS["A1deglr_u50_kpsf2n"] = VARIANTS["A1deglr_u50"]  # psf_from_n_inv: padded (noise kernel as PSF on the 2N grid)
VARIANTS["A1deglr_u50_kpsf2n_nonoise"] = VARIANTS["A1deglr_u50"]  # as kpsf2n, without the noise model
VARIANTS["C1deglr_u50_kpsf2n"] = VARIANTS["C1deglr_u50"]  # psf_from_n_inv: padded, beta = 0 truth
VARIANTS["O1deglr_u50_kpsf2n"] = VARIANTS["O1deglr_u50"]  # psf_from_n_inv: padded, O1 as the only component
VARIANTS["S1deglr_u50_kpsf2n"] = VARIANTS["S1deglr_u50"]  # psf_from_n_inv: padded, a spectrum per component
VARIANTS["O0deglr_u50_std"] = VARIANTS["O0deglr_u50"]  # O0 as the only component, standard padded PSF + noise kernel
VARIANTS["O0deglr_u50_chres"] = VARIANTS["O0deglr_u50"]  # as O0deglr_u50_std with the residual per channel (channel_residual)
VARIANTS["O0deglr_u50_sfonly"] = VARIANTS["O0deglr_u50"]  # SF stage only, to read the noise weights at its end
VARIANTS["A1wfix_fix7"] = VARIANTS["A1wfix_fixbin"] = VARIANTS["A1deglr_u50wfix"]  # weights corrected, standard kernels, per-channel residual, noise reset at MF; SF on 7 channels / one MF bin
VARIANTS["A1wfix_kfix"] = VARIANTS["A1deglr_u50wfix"]  # as A1wfix_fix7 with psf_from_n_inv: padded
VARIANTS["S1wfix_kfix"] = VARIANTS["S1wfixlr_u50"]  # the A1wfix_kfix setup on the other truths
VARIANTS["S1wfix_fix7"] = VARIANTS["S1wfixlr_u50"]  # the A1wfix_fix7 setup on the truth with a spectrum per component
VARIANTS["S1wfix2_fix7"] = VARIANTS["S1wfix2lr_u50"]  # S1wfix_fix7 on the second noise seed
VARIANTS["S1wfix_rp"] = VARIANTS["S1wfixlr_u50"]  # S1wfix_fix7 with the real object taper ([0.3, 0.1], lognormal) and curvature prior
VARIANTS["S1wfix_bgoff"] = VARIANTS["S1wfixlr_u50"]  # S1wfix_fix7 with the background offset at the mean log brightness, not the log of the mean
VARIANTS["S1wfix_rpfl"] = VARIANTS["S1wfixlr_u50"]  # S1wfix_rp with spatial fluctuations [1, 0.5] (or [3, 1]) instead of [5, 1]
VARIANTS["S1wfix_rpfl_oldoff"] = VARIANTS["S1wfixlr_u50"]  # S1wfix_rpfl with the old offsets (log of the mean): the fluctuations alone
VARIANTS["S1wfix_rpfl_t02"] = VARIANTS["S1wfixlr_u50"]  # S1wfix_rpfl with the object tapers fixed at cov 0.2
VARIANTS["S1wfix_rpfl5_t02"] = VARIANTS["S1wfixlr_u50"]  # S1wfix_rpfl_t02 with spatial fluctuations [5, 1]
VARIANTS["A1wfix_sf_base"] = VARIANTS["A1wfix_sf_app"] = VARIANTS["A1wfix_sf_pb"] = VARIANTS["A1deglr_u50wfix"]  # A1wfix_fix7 up to the end of SF: as it is / prefit to the apparent truth / beam in SF
VARIANTS["T1wfix_kfix"] = VARIANTS["T1wfixlr_u50"]
VARIANTS["BO0wfix_kfix"] = VARIANTS["BO0wfixlr_u50"]
VARIANTS["O0deglr_u50_chres_nmreset"] = VARIANTS["O0deglr_u50"]  # per-channel residual, noise model reset at the start of MF
VARIANTS["O1deglr_u50_std"] = VARIANTS["O1deglr_u50"]  # as O1deglr_u50_kpsf2n with the standard padded PSF + noise kernel
SAMPLING = {"B4L": dict(n_samples=4, sample_mode="linear_resample")}  # overrides of base_opt (both stages)


def noise_seed(variant):
    return NOISE_SEEDS.get(DATA_OF.get(variant, variant), SEED_NOISE)


def data_variant(variant, stage):
    """Variant whose synthetic data file a variant uses ('sf' or 'mf' stage); the binned
    variant B (and variants on its data) use the single-frequency data of A."""
    v = DATA_OF.get(variant, variant)
    return {"B": "A", "Bcomp": "Acomp", "B1deg": "A1deg"}.get(v, v) if stage == "sf" else v


def field_grid(n_coarse=N_COARSE, factor=FAC):
    """SignalGrid of the field (same pixel size as the reference run): 0.5 deg (default) or, with
    n_coarse=N_COARSE_1DEG, the 1 deg field; factor=1: the low-resolution grids of make_lowres_1deg.py."""
    from aim_resolve.model.grid import SignalGrid

    fov = FOV_DEG * n_coarse / N_COARSE
    grid = SignalGrid.build(space=(n_coarse, n_coarse), factor=factor, fov=float(np.deg2rad(fov)))
    assert grid.shape == (n_coarse * factor, n_coarse * factor), grid.shape
    return grid


def truth_dir(variant):
    """Directory of a truth variant: version-2 truths (*_comp) live in OUT_COMP."""
    if variant.startswith("1deg_"):
        return f"{OUT_1DEG}/truth/{variant[5:]}"
    return f"{OUT_COMP if variant.endswith('_comp') else OUT}/truth/{variant}"


class TruthSky:
    """Intrinsic truth sky S(x, nu) = I0 exp(alpha u + beta u^2) [Jy/sr] (make_truth.py).

    Truths with components (make_truth_comp.py, variants *_comp) add point sources with
    their own spectra: S = I0_diffuse exp(alpha u + beta u^2) + I0_pts exp(alpha_p u + beta_p u^2).
    `i0` is the total at nu_ref; `alpha` / `beta` are those of the dominant component per pixel.
    """

    def __init__(self, variant):
        t = np.load(f"{truth_dir(variant)}/truth.npz")
        self.nu_ref = float(t["nu_ref"])
        self.i0, self.alpha, self.beta = t["i0"], t["alpha"], t["beta"]
        self.has_points = "i0_pts" in t.files
        self.comps = [(t[f"i0_{c}"], t[f"alpha_{c}"], t[f"beta_{c}"]) for c in ("bg", "o0", "o1") if f"alpha_{c}" in t.files]
        if self.has_points:
            self.i0_diffuse, self.alpha_diffuse, self.beta_diffuse = t["i0_diffuse"], t["alpha_diffuse"], t["beta_diffuse"]
            self.i0_pts, self.alpha_pts, self.beta_pts = t["i0_pts"], t["alpha_pts"], t["beta_pts"]

    def __call__(self, nu):
        u = np.log(float(nu) / self.nu_ref)
        if self.comps:  # a spectrum per component (make_truth_sep.py)
            return sum(i0 * np.exp(a * u + b * u**2) for i0, a, b in self.comps)
        if self.has_points:
            return (self.i0_diffuse * np.exp(self.alpha_diffuse * u + self.beta_diffuse * u**2)
                    + self.i0_pts * np.exp(self.alpha_pts * u + self.beta_pts * u**2))
        return self.i0 * np.exp(self.alpha * u + self.beta * u**2)


def bin_assignment(nu_data, bounds):
    """Model-bin index per data channel for the given bin bounds, as resolve's
    response does it (Observation.restrict_by_freq = searchsorted, [lo, hi))."""
    chan_bin = np.full(nu_data.size, -1)
    for k in range(len(bounds) - 1):
        start, stop = np.searchsorted(nu_data, [bounds[k], bounds[k + 1]])
        chan_bin[start:stop] = k
    if np.any(chan_bin < 0):
        raise ValueError("data channels outside the bin bounds")
    return chan_bin


def effective_frequencies(nu_nominal, nu_data, w_chan):
    """Fixed bins + effective model frequencies.

    The bins are resolve's bins of the nominal model frequencies (midpoints,
    outer bounds mirrored); they are kept fixed and passed to the likelihood as
    `freq_bounds`. The model frequencies become the effective frequency of each
    bin, sum(w nu) / sum(w) with the real weights (flags = 0).
    """
    from resolve.irg_space import IRGSpace

    bounds = IRGSpace(np.asarray(nu_nominal, dtype="float64")).binbounds()
    chan_bin = bin_assignment(nu_data, bounds)
    nu_eff = np.array([np.sum(w_chan[chan_bin == k] * nu_data[chan_bin == k]) / np.sum(w_chan[chan_bin == k])
                       for k in range(len(nu_nominal))])
    return nu_eff, bounds, chan_bin

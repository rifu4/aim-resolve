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
}
NOISE_SEEDS = {"Ap": 3}  # noise seed per variant, default SEED_NOISE

# inference variants: same data as another variant, different sampling settings
DATA_OF = {"B4L": "B"}  # variant -> variant whose synthetic data it uses
VARIANTS["B4L"] = VARIANTS["B"]
SAMPLING = {"B4L": dict(n_samples=4, sample_mode="linear_resample")}  # overrides of base_opt (both stages)


def noise_seed(variant):
    return NOISE_SEEDS.get(DATA_OF.get(variant, variant), SEED_NOISE)


def data_variant(variant, stage):
    """Variant whose synthetic data file a variant uses ('sf' or 'mf' stage); the binned
    variant B (and variants on its data) use the single-frequency data of A."""
    v = DATA_OF.get(variant, variant)
    return "A" if (stage == "sf" and v == "B") else v


def field_grid():
    """SignalGrid of the field (same pixel size as the reference run)."""
    from aim_resolve.model.grid import SignalGrid

    grid = SignalGrid.build(space=(N_COARSE, N_COARSE), factor=FAC, fov=float(np.deg2rad(FOV_DEG)))
    assert grid.shape == (N, N), grid.shape
    return grid


class TruthSky:
    """Intrinsic truth sky S(x, nu) = I0 exp(alpha u + beta u^2) [Jy/sr] (make_truth.py)."""

    def __init__(self, variant):
        t = np.load(f"{OUT}/truth/{variant}/truth.npz")
        self.nu_ref = float(t["nu_ref"])
        self.i0, self.alpha, self.beta = t["i0"], t["alpha"], t["beta"]

    def __call__(self, nu):
        u = np.log(float(nu) / self.nu_ref)
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

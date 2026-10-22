# Synthetic-data test: spectral index and curvature recovery

Goal: show that the multi-frequency reconstruction recovers the spectral index α and the
curvature β (log-parabola, `curvature_from_cube`) of a known sky. Also quantify how much of the β error comes
from the binning of the data channels into the 6 model frequencies (referee comment on the
channel → model-frequency mapping).

Reference run: `runs/fast_vi_1f_1024_1z_b/opt/4_rec_3z_1_6f_1`
(single-frequency run `4_rec_3z_1` → frequency transition → multi-frequency extension, `pb_correction: true`,
`n_samples: 2`).

**Version 1 (this plan):** one extended source (ESO 137-006) plus the diffuse background, in a 0.5° × 0.5° field,
reconstructed with a single correlated field. The multi-component version (all components, 1° field) is a possible
later extension.

Scripts: `steering/paper/synthetic/` · outputs: `steering/runs/synth_05deg/` · data: `/scratch/users/rfuchs/data/synth/`

---

## Decisions

| item | choice |
|---|---|
| field | 0.5° × 0.5° around the phase centre: sky grid 768² at 2.34″, model grid 256² (`factor: 3`) |
| truth I₀ | object `O0` (ESO 137-006, 82.9 Jy) + diffuse background `BG` (3.9 Jy) of the reference run, posterior mean at ν_ref (intrinsic sky). The background removes empty (zero) regions |
| truth α | Gaussian random field, mean −1, std 0.5, smoothed to ≈ 45″ |
| truth β | Gaussian random field, mean −0.2, std 0.3, smoothed to ≈ 45″, independent of α |
| | α, β apply to the total sky, so S = I₀ exp(α u + β u²) is an exact log-parabola everywhere |
| model | **one** correlated field for the total sky. Priors of the reference run's diffuse components (`bg.5`/`o0.5`, `bg.6`/`o0.6`); offset = mean ln I₀ (11.1) |
| model frequencies | **effective frequencies** ν_eff = Σwν/Σw of fixed bins (the nominal bins: 961.4 / 1022.9 / 1084.1 / 1222.0 / 1364.2 / 1433.8 / 1503.5 MHz) → 992.84, 1053.58, 1115.09, 1331.19, 1398.35, 1465.67 MHz |
| pipeline | single-frequency reconstruction → frequency transition (`mode: freq`) → multi-frequency reconstruction, in one config (`nifty/start_rec.py`) |
| optimiser | `base_opt.6` of `mf_ext.yml` for both stages: 15 single-frequency + 30 multi-frequency major iterations, `n_samples: 2` |
| GPUs | one GPU per run, at most two at the same time (A and C in parallel) |

### Variants (same truth I₀ and α, same noise realisation)

| variant | truth spectrum per data channel | tests | default |
|---|---|---|---|
| **A** | ln S = ln I₀ + α u_c + β u_c², each channel at its exact frequency, beam PB(ν_c) per channel | inference + binning, i.e. the realistic case | **on** |
| **C** | like A, β ≡ 0 | the referee's question directly: is β ≈ 0 recovered? | **on**, in parallel with A |
| **B** | each channel gets its bin's value at ν_eff, beam as in the model (bin average) | inference alone; A − B is the binning error | optional |

### Primary beam (JimBeam `MKAT-AA-L-JIM-2020`)

| stage | real run | synthetic test |
|---|---|---|
| truth | – | intrinsic sky |
| data | – | S(x, ν_c) · PB(x, ν_c) per channel |
| single-frequency stage | no beam (apparent sky) | same |
| frequency transition | copies the parameters unchanged | same |
| multi-frequency stage | `pb_correction: true`, bin-averaged beam | same, bins = fixed `freq_bounds` |

---

## Code change: `freq_bounds` (done)

resolve derives the channel bins from the model frequencies (midpoints, outer bounds mirrored). With ν_eff as model
frequencies the outer bounds move inward and edge channels fall outside; iterating to a consistent set collapses (52
channels dropped). New opt-in likelihood option **`freq_bounds`** (bin bounds in Hz) decouples the two:
- `fast_resolve/response.py`: `frequency_domain(freq, freq_bounds)`, an `IRGSpace` with explicit `binbounds`, used by
  `build_exact_responses`;
- `fast_resolve/beam.py`: `build_primary_beam(..., freq_bounds)` uses the same bins for the channel-averaged beam;
- `likelihood.py`: `fast_likelihood(..., freq_bounds=None)` passes it to both. Default behaviour unchanged.
- Tests: `tests/fast_resolve/test_response.py` (bounds, validation, resolve's channel assignment),
  `tests/fast_resolve/test_beam.py` (beam average with explicit bounds). All `tests/fast_resolve` pass.

---

## Phases

### 0. Preparation (`prep.py`) ✓
Fixed bins and ν_eff from the real weights and flags of `eso_961-1503mhz.npz` (all 468 channels used).

### 1. Truth (`make_truth.py`) ✓
`truth/{main,beta0}/`: `truth.npz`, `eso137_truth_<ν>MHz.fits` at ν_eff, `eso137_truth_spectral_index.fits`,
`eso137_truth_curvature.fits`, `truth_overview.png`.
- *Check:* O0 falls off smoothly to ≈ 0 at its box edge (edge max 2e-6 of the peak, no step); with the
  background there are no zeros (minimum 2.5e3 Jy/sr).

### 2. Synthetic visibilities (`make_vis.py`)
For both real data files (`eso_961-1503mhz.npz`, 468 channels; `eso_986-1137mhz.npz`, 7 channels): per channel
S(x, ν_c) · PB(x, ν_c), degridded with the inference's resolve operator (w-gridding, ε = 1e-9, double precision);
noise Re, Im ~ N(0, (s_c/√w)²), with s_c the robust noise level of the real data per channel (longest 10 % of
baselines, 0.53–1.10). Flags stay flags.

**Validation (`--check`, done):**
- **V1, conventions:** per-channel prediction vs the inference's multi-frequency operator with `freq_bounds` and beam →
  relative difference **3e-11** ✓
- **V2, realism:** dirty images per bin, synthetic vs real: correlation 0.95–0.99, same peak pixel, peak ratio
  **1.00–1.02** (multi-frequency data) ✓
- **V3, noise:** synthetic noise level equals the real one per channel ✓
- **Finding:** the real single-frequency file `eso_986-1137mhz.npz` is ≈ **2× fainter** than the multi-frequency file
  at the same position (synthetic/real peak 1.95 there, 1.00 for the multi-frequency data). The synthetic data use one
  consistent flux scale for both. To be checked separately: how the two files were made (Stokes I convention?).

### 3. Kernels and configs (`make_configs.py`)
- `runs/synth_05deg/{A,C}/config.yml` (see decisions); checked: models instantiate, all single-frequency parameters
  have a counterpart in the multi-frequency model, schedule = 15 single-frequency + transition + 30 multi-frequency
  iterations.
- Kernels for the 768² grid (shared by all variants): `runs/kernels/{pk,nk}_synth05_986-1137mhz_1f_768.pkl`,
  `runs/kernels/{pk,nk}_synth05_961-1503mhz_6f_eff_768.pkl`. Built once on the CPU before the runs start
  (avoids two runs writing the same file).

### 4. Inference
A and C in parallel, one GPU each: `python nifty/start_rec.py --config runs/synth_05deg/<v>/config.yml --cuda_device <gpu>`.
- *Check:* reduced χ² ≈ 1 (`minisanity.txt`), residuals noise-like, after each stage.

### 5. Post-processing
`save_fits.py` / `plot_from_fits.py` adapted to the run (one component, grid read from the config): FITS of the
posterior samples and mean at ν_eff; α, β per sample with `curvature_from_cube` at u = ln(ν_eff/ν_ref). The truth α, β
go through the same fit on the truth cube at ν_eff.

### 6. Validation (`compare_synthetic.py`)
**Mask:** pixels with enough signal (flux mask as in `cs_alpha`/`cs_curvature`), in S/N bins at ν_ref, and the
regions of `regions.yml` that lie in the field.

| | metric | pass criterion |
|---|---|---|
| a. intensity | (I_rec − I_true)/I_true at ν_ref; integrated flux | a few % at high S/N |
| b. α, β accuracy | Δα, Δβ maps; recovered vs true scatter; median bias and MAD per S/N bin | high S/N: \|median Δα\| < 0.05, \|median Δβ\| < 0.05 |
| c. calibration | z = (rec − true)/σ_post for α, β; histogram; 1σ/2σ coverage, on pixels one beam apart, per S/N bin (2 samples → σ is noisy) | std(z) 0.8–1.2 |
| d. β = 0 (C) | recovered β in the mask | median β consistent with 0 within σ_β; no trend with α |
| e. binning (A − B, if B is run) | Δβ_A − Δβ_B per S/N bin vs the prediction of `check_channel_binning.py` | at the predicted level |
| f. systematics | Δβ vs true α; Δβ vs S/N; Δα vs Δβ | no significant trend at high S/N |

**For the paper:** if |Δβ| (A, and C's median) ≪ measured |β| (0.2–0.5) and < σ_β, the binning with effective
frequencies is adequate; otherwise its size decides whether the per-channel interpolation is needed. Figure: truth vs
recovered α and β (A), Δβ maps (A, C), z histograms; table of bias and scatter per S/N bin.

## Notes
- The single-frequency stage starts from scratch here (the reference single-frequency stage continued from earlier
  stages), so it uses the same ramp-up schedule as the multi-frequency stage, with 15 instead of 30 iterations (it only
  initialises I₀; α/β come from the multi-frequency stage, which keeps the 30 iterations of the real run).
- The synthetic sky has no emission outside the 0.5° field, unlike the real sky; the test is about α/β of
  ESO 137-006 and does not cover confusion from outside the field.

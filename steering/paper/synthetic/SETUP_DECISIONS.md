# Setup decisions for the real SF and MF runs

Status: 2026-11-04. Based on the synthetic tests on the 1 deg field (O0 = ESO 137-006, O1 = ESO 137-007, background).

Almost all tests are at low resolution (`*lr_u50`: uv cut at 0.5 Nyquist) and share one noise seed. A second noise seed on the S1 truth (P+K) moved the O1 edge metric by 0.05 in pixel beta, 0.01 in integrated beta and 0.6 % in flux, so differences above about 0.05 / 0.02 / 1 % are real.

Metric used below, unless noted: O1 at r > 0.35 deg, as beta error per pixel / integrated beta offset / region flux relative to truth.


## Fixed

### Data and weights

| choice | value | why |
|---|---|---|
| Visibility weights | Corrected per channel, w -> w / s_c^2 (s_c = real noise over 1/sqrt(w)) | s_c runs from 1.06 at 985 MHz to 0.61 at 1477 MHz, so the raw weights misrepresent the noise across the band |
| SF data (real) | One MF bin, 1007.35-1053.28 MHz (55 channels, effective 1030.1 MHz, `eso_1007-1053mhz_wfix.npz`) | The 7-channel SF file is on another flux scale than the MF data: SF / MF = 0.536 to 0.492 across 987-1062 MHz (about nu^-1.2), the two files are one day apart. The SF stage must start MF on the MF scale. The zoom transition from the old stage 4 (7-channel data) needs `flux_scale: 2.0` |
| SF data (synthetic) | 7 channels, 986.7-1137.2 MHz | The synthetic data have no scale mismatch. One MF bin was worse in MF there (-0.30 / -0.34 / 1.07-1.08 against -0.20 / -0.25 / 1.04-1.06, A1 truth, older setup), within one-realisation scatter. Not yet repeated with the current setup |
| MF data | 468 channels, 8 model bins | Existing setup, no test gave a reason to change it |

### Likelihood and noise

| choice | value | why |
|---|---|---|
| Noise model | On (inverse-gamma per Fourier mode) | Running without one failed on real data. No-noise synthetic runs are diagnostics only |
| Noise at the SF -> MF switch and at new data | Reset to the prior (`reset_noise: true`) | SF ends with low noise weights on short baselines (0.036). Reset keeps them high to the end of MF (0.80 shortest, 0.94 elsewhere) |
| Residual per major iteration | MF: per channel, sky and beam per data channel (`channel_residual: true`) | Exact chi^2 1.68 with the sky per bin, 1.00 per channel. The truth through this prediction gives 1.000 on every baseline range, so in-bin error is gone |
| Primary beam in MF | Yes (`pb_correction: true`) | The beam acts on the apparent sky; needed for spectra at the field edge |
| Primary beam in SF | Synthetic: no. Real: yes, averaged over the band | `A1wfix_sf_pb` ends SF far worse: O1 edge 1.33 of the apparent sky (no beam: 1.04), outer field 1.34 of truth. The real SF runs (15-17) keep the beam for a cleaner transition to MF, with one MF bin the band average is close to the beam at 1030 MHz |

### Model and inference

| choice | value | why |
|---|---|---|
| Components | Background, O0, O1, each with its own alpha and beta | Physical, and the per-component truth is the best-behaved test case |
| Prior pixel size | The pixel size of the refined grid (`distances / factor`) for the fields and the taper | Bug fixed on 2026-11-03: the priors used the coarse pixel size with the refined shape, so on zoomed grids (factor 3) the taper acted 3 times narrower than configured and the offsets came out wrong (o0 19.6) |
| Offsets | `get_offset`: mean over the mask of log(max(I, 1e-5 max)) - log(taper at its prior median), background taper 1 | The offset is the mean of the log field; the log of the mean lay about var(log) / 2 too high. Largest single gain: `S1wfix_rpfl` against `S1wfix_rpfl_oldoff` (same priors) +0.03 / -0.07 against -0.21 / -0.23 |
| alpha and beta power spectrum | Shared | A truth with beta at its own correlation length (150") gave no real gain (-0.117 / -0.17, against -0.115 / -0.21) |
| MF major iterations | 15 | Continuing from 15 to 20 MF iterations changed nothing relevant: -0.200 / -0.25 to -0.202 / -0.24, flux 1.04-1.06 in both. Exact chi^2 1.709 to 1.704 |

### Diagnostics

| choice | value | why |
|---|---|---|
| Fit quality measure | Per-channel exact chi^2, split by baseline length | Logged chi^2 is not comparable between runs, and binned exact chi^2 misleads: S1 reads 1.94 binned but 1.022 per channel |
| Box edges | Rings inside / outside each object box and the brightness step across the edge (`edge_check.py`) | A taper that is too wide leaves a step at the box edge; the region metrics do not show it |
| Check on real results | Self-template fit (`self_template.py`) | Needs no truth. Raise `MIN_SHARE` first: a faint ring gave a noise flag |


## Leaning, waiting for a decision

### Taper and fluctuations (S1 truth, P+K, new offsets unless noted)

| run | taper | fluctuations | O1 > 0.35 deg: pixel / integrated beta / flux | bg share at O1 > 0.35 (truth 0.123) | outer field | per-channel chi^2 (shortest annulus) | O1 / O0 step (truth 1.033 / 0.988) |
|---|---|---|---|---|---|---|---|
| `S1wfix_fix7` | synthetic [0.1, 0.01] | synthetic | -0.14 / -0.15 / 1.032 | 0.284 | 1.073 | - | 1.016 / 0.997 |
| `S1wfix_rp` (old offsets) | lognormal [0.3, 0.1] | [5, 1] | -0.07 / -0.15 / 1.033 | 0.215 | 1.077 | 1.022 (1.064) | 1.159 / 1.002 |
| `S1wfix_rpfl_oldoff` (old offsets) | lognormal [0.3, 0.1] | [1, 0.5] | -0.21 / -0.23 / 1.053 | 0.216 | 1.101 | 1.024 (1.086) | 1.084 / 0.991 |
| `S1wfix_rpfl` | lognormal [0.3, 0.1] | [1, 0.5] | +0.03 / -0.07 / 1.056 | 0.215 | 1.157 | 1.020 (1.080) | 1.159 / 0.997 |
| `S1wfix_rpfl_t02` | fixed 0.2 | [1, 0.5] | -0.05 / -0.12 / 1.064 | 0.203 | 1.160 | 1.014 (1.059) | 1.076 / 1.062 |
| `S1wfix_rpfl5_t02` | fixed 0.2 | [5, 1] | +0.01 / -0.07 / 1.057 | 0.220 | 1.134 | 1.011 (1.037) | 1.071 / 1.056 |

Taper (cov in box units, the box spans +-0.5; the taper is 0.044 / 0.14 / 0.25 at the box edge for 0.2 / 0.25 / 0.3):

| choice | leaning | evidence |
|---|---|---|
| Object taper | Fixed 0.2 | The wide lognormal taper [0.3, 0.1] leaves a step of 1.16 at O1's box edge in MF (truth 1.03). On real data (run 17, lognormal [0.25, 0.05]) the tapers ran 5-11 sigma from the prior median (O0 cov_y -5.6 sigma, O1 cov_x +11.4 sigma); run 15 collapsed O0's cov_y to 0.06 and the background took O0's outer parts. Fixed 0.2 halves the O1 step (1.07-1.08) but adds one at O0 (1.06): O0 puts 1-3.5 % of the flux at its edge where the truth has 0.2 %, and the background is about 30 % high there inside and outside the box. Integrated beta is 0.05 worse than `rpfl` with [1, 0.5], equal with [5, 1] |
| Spatial fluctuations | Open between [5, 1] and [1, 0.5] | The prefit puts the latents 5-9 sigma below the [5, 1] prior (synthetic and real; the truth components want amplitudes of 0.3-1), [1, 0.5] fits them (bg -1.1, o0 -0.06, o1 -0.26 sigma). At the end of SF [1, 0.5] was better (outer field 1.25 against 1.40, background in the O1 box 1.24 against 1.41). After MF [5, 1] is better: integrated beta -0.07 against -0.12, outer field 1.13 against 1.16, chi^2 at the shortest baselines 1.037 against 1.059, O1 beta MAD 0.12 against 0.16. Box-edge steps equal. An intermediate [2, 1] would put the fitted amplitudes about 2.7 sigma below the median |

### Kernels

| choice | leaning | decided by | current evidence |
|---|---|---|---|
| MF kernels | P+K (all runs above) | Real-prior test | S1 truth: P+K -0.138 / -0.154 / 1.03-1.04 (seed 3: -0.091 / -0.142 / 1.035-1.045), K on 2N +0.016 / -0.058 / 1.06-1.08. A1 truth: P+K -0.20 / -0.25 / 1.04-1.06, K on 2N -0.115 / -0.21 / 1.09-1.10. Per-channel chi^2 identical (1.022 both, S1). K on 2N gives the better beta, P+K the smaller flux excess; both differences exceed seed scatter |
| SF kernels | Standard P (2N) + K | - | SF has no beta; real wfix kernels exist for P+K, also for the one-bin SF data |


## Open

### Causes of the remaining excess

The background holds too much flux under the object boxes: 1.4-1.6 times its truth in the box interiors in every run with the real priors, about 1.3 times around O0's box edge, and twice its share at O1's edge (0.20-0.22 against 0.12). The field outside O1 at r > 0.25 deg is 13-16 % high with the new offsets. The data do not support this: the truth fits the shortest baselines at 0.999, the reconstructions at 1.04-1.09.

| question | test | status |
|---|---|---|
| Is the imbalance at the start of the synthetic runs the cause? | `A1wfix_sf_base` against `A1wfix_sf_app` (prefit to the apparent truth) | No. End of SF identical: O1 edge 1.038 / 1.038 of apparent, outer field 1.300 / 1.306 |
| Is it noise rectified by the positive model? | `A1wfix_fix7_nf` (noise-free data) | No. Noise-free gives -0.171 / -0.23 / 1.04-1.06, bg share 0.246, as the noisy run |
| Is the background offset too high? | `S1wfix_bgoff` (lower background offset) | No. The zero mode compensated the lower offset |
| Does the object taper hold the excess? | Real taper priors (`S1wfix_rp`, `rpfl`), fixed taper (`rpfl_t02`, `rpfl5_t02`) | Partly: a wider taper lowers the background share at O1's edge (0.28 to 0.20-0.22), but leaves box-edge steps. The background excess inside the boxes remains in all runs |
| Does a tighter background prior push the flux into the boxes? | Background fluctuations [0.5, 0.25] (the synthetic background wants about 0.55), objects [1, 0.5] | Proposed, not run. On real data the background wanted a larger amplitude (about 1.0) than the objects (0.37-0.58), so a tight background prior may not carry over |

The outer field sits about 30 % above the apparent sky at the end of SF (at the intrinsic level) regardless of the start, and the excess is there without noise. It is deterministic: the model or prior, not noise or the start.

### Validation still missing

| item | why it matters |
|---|---|
| beta = 0 truth with the final setup | Direct test of whether the model invents curvature at the edge; a likely referee question. The data exist |
| Full-res synthetic test with the S1 truth and the final setup | Main paper result. The low-res tests now use the real priors (runs above), but not the full resolution |
| Synthetic SF on one MF bin with the final setup | The real SF uses one MF bin, the synthetic tests 7 channels |
| Per-channel residual at full res | Never run at full res; 468 channels could dominate each major iteration and raise the 200 GB memory use. Needs a one-iteration timing test |

### Known limitations, no fix yet

| item | state |
|---|---|
| O1 edge beta bias | Reduced from -0.59 to about -0.2 (shared truth) and to about 0 / -0.07 per pixel / integrated with the new offsets (per-component truth), not removed. Goes in the paper as a systematic at r > 0.35 deg |
| beta uncertainties too small | 45-66 % of pixels above the flux cut within 1 sigma (68 % expected). Measure the underestimate from the synthetic runs and quote it |
| Longest baselines | Per-channel chi^2 1.02-1.05 at 4-7.4 klambda, near the uv cut. Not investigated |
| Faint background spectrum | alpha high by +0.04 to +0.16, with too few pixels within 1 sigma. Only about 110-140 pixels above the flux cut |


## Ruled out as causes

| candidate | evidence |
|---|---|
| Data generation, beam, truth, bookkeeping | Checked directly |
| PSF position dependence, O1 box, fine structure | Checked directly |
| Convergence | No change from 15 to 20 MF iterations |
| In-bin model error | Truth through the per-channel prediction: 1.000 on all baseline ranges |
| P/K mismatch as the main cause | Halves the bias on the original setup, adds little once the other fixes are on. Excess appears with both kernels |
| Mean against median of the positive model | Accounts for about 1 % |
| Beam amplification (1/PB) of the excess | Same fractional excess in the apparent sky; beam about 0.78 there |
| Falling noise weights | Stay high with the reset, bias remains |
| alpha / beta correlation length | No real gain |
| Background offset | Compensated by the zero mode (`S1wfix_bgoff`) |


## Real runs

| run | setup | result |
|---|---|---|
| 15 (`4_rec_3z_15`) | One MF bin, `flux_scale: 2.0`, lognormal taper [0.3, 0.1], fluctuations [1, 0.5] | Stopped at iteration 53: O0 cov_y collapsed to -5.1 sigma (about 0.06), the background took O0's outer parts |
| 16 (`4_rec_3z_16`) | As 15, object taper [0.3, 0.05] | Stopped: the O2 transition fit failed (one knot plus rings) |
| 17 (`4_rec_3z_17`) | Taper [0.25, 0.05] for objects and tiles, offsets from `real_offsets.py` | Tapers 5-11 sigma from the prior median at iteration 70 |

Before the real MF, `mf_ext_8f.yml` needs: the spatial fluctuations of the SF stage in `params_mf`, no `flux_scale`, `channel_residual`, `reset_noise` and the `_wfix` data and kernels.


## Order for the real runs

1. Done: corrected-weight data and P+K kernels for SF and MF (`*_wfix.npz`, `runs/kernels/*_wfix.pkl`), one-bin SF data and kernels.
2. Done: prior pixel-size fix, `get_offset` on the mean log field.
3. Decide the spatial fluctuations ([5, 1] or [1, 0.5]); the taper is fixed at 0.2 unless the O0 step needs more.
4. Real SF with the fixed taper 0.2 (from `zoom_ext_newsf_bin_t025.yml`).
5. Timing test of the per-channel residual at full res.
6. Real MF rerun, with the full-res synthetic test on the same setup, then the self-template fit.


## Scripts

Synthetic tests (`paper/synthetic/`): truths `make_truth_*.py`, data `make_vis.py`, `make_lowres_1deg.py`, `make_rows_subset.py`, configs `make_configs.py`, `make_config_1deg.py` (variants in `synth_common.py`), prefit `prefit_comp.py`. Analysis of a run: `analyze_run.sh` (calls `compare_synthetic.py`, `compare_component.py`, `check_beta_radius.py`, `exact_chi2_rec.py`, `comp_split.py`, `nm_modes.py`), plus `edge_check.py` (box edges), `switch_check.py` (SF -> MF), `mean_bias.py`, `chi2ch_full.py` (per-channel chi^2 by annulus), `obj_offsets.py`, `plot_truth.py` (truth maps as the callback plots).

Real data (`paper/real_data/`): `real_wfix.py` (corrected weights), `real_bin_data.py` (one-bin SF data), `real_kernels.py`, `real_bin_kernels.py` (kernels), `sf_mf_scale.py` (SF / MF flux scale), `real_offsets.py` (offsets of a zoom extension).

Runs: `steering/runs/synth_1deg/r10_sf5_mf15/`. Logs: `steering/runs/synth_1deg/logs/`.

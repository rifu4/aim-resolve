# %%
# ---------------------------------------------------------------------------
# Check of the data-channel -> model-frequency binning (referee question):
# does the piecewise-constant spectral model within each frequency bin bias the
# recovered spectral index alpha and curvature beta?
#
# How the mapping works (resolve.response._InterferometryResponse): the sky
# frequencies are an IRGSpace whose bin bounds are the midpoints between
# neighbouring model frequencies (outer bounds mirrored). Every data channel in
# [bound_k, bound_k+1) gets the sky image of model frequency k, i.e. the model is
# spectrally CONSTANT within a bin (the uv coordinates are scaled with the true
# channel frequency, so only the spectrum is binned).
#
# For a source with true spectrum S(nu), fitting a bin-constant model to the
# channels of bin k (weighted least squares) gives the weighted bin mean
#     S_k = sum_i w_i S(nu_i) / sum_i w_i ,
# which is then interpreted as the flux at the nominal model frequency nu_k. The
# script quantifies this:
#   1) per bin: channel range, flags, effective frequency sum(w nu) / sum(w),
#      offset to nu_k, and the in-bin spread of a steep spectrum (per-channel
#      model mismatch at the bin edges);
#   2) alpha / beta bias: S_k(alpha_true, beta_true) evaluated with the real
#      per-channel weights, then fitted with the log-parabola of
#      plot_from_fits.curvature_from_cube (ln S = a + alpha u + beta u^2,
#      u = ln(nu / nu_ref)) -> delta alpha, delta beta;
#   3) the same per uv-scale: for extended emission the weight a channel puts
#      on a given angular scale depends on the uv coverage at |u| = |b| nu / c,
#      so the effective frequency of a bin can differ between scales.
# For an unresolved source (2) is exact; (3) is the scale-by-scale version.
# ---------------------------------------------------------------------------
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from resolve.irg_space import IRGSpace

import aim_resolve as aim

# %%
# ---------------------------------------------------------------------------
# Input / output.
# ---------------------------------------------------------------------------
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"
mf_rec = "4_rec_3z_1_6f_1"
lh_sec = "lh.6"
REF_IDX = 1  # reference-frequency index of the spectral model (ref_freq = freq[1])

odir = f"{dir}/opt/{mf_rec}/binning_check"
os.makedirs(odir, exist_ok=True)

ALPHAS = np.round(np.arange(-3.0, 0.51, 0.25), 2)  # true spectral indices to test
BETAS = [0.0, -0.3, -0.5]  # true curvatures to test
ALPHA_STEEP = -2.5  # steep reference spectrum for the per-bin / per-scale tables
N_UV = 12  # number of uv-length annuli (log-spaced) for the scale dependence
C_LIGHT = 299792458.0

# %%
# ---------------------------------------------------------------------------
# Model frequencies and data file of the run (from opt.yml).
# ---------------------------------------------------------------------------
opt_yml = f"{dir}/opt/{mf_rec}/opt.yml"
print("load:", opt_yml)
optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)
lh_cfg = optim_cfg.sections[lh_sec]
data_sec = lh_cfg["data"].lstrip("=")
it = lh_sec.split(".")[-1]
# model frequencies: the `freq` of the sky components of this iteration (all equal)
nu_model = next(
    np.asarray(dct["freq"], dtype="float64") for sec, dct in optim_cfg.sections.items()
    if sec.startswith("sky_") and sec.endswith(f".{it}") and isinstance(dct, dict) and "freq" in dct
)
data_fn = optim_cfg.sections[data_sec]["fname"]
print("model freqs [MHz]:", np.round(nu_model / 1e6, 1).tolist(), "| ref:", round(nu_model[REF_IDX] / 1e6, 1))
print("data:", data_fn)

# %%
# ---------------------------------------------------------------------------
# Data: channel frequencies, weights (0 = flagged) and uv lengths.
# ---------------------------------------------------------------------------
d = np.load(data_fn, allow_pickle=True)
nu_data = np.asarray(d["freq"], dtype="float64")  # (n_chan,)
w = np.asarray(d["weight"])  # (n_pol, n_row, n_chan)
w = w.sum(axis=0) if w.shape[0] > 1 else w[0]  # (n_row, n_chan)
uvw = np.asarray(d["antpos0"], dtype="float64")  # (n_row, 3) uvw [m]
uvlen_m = np.hypot(uvw[:, 0], uvw[:, 1])
del uvw
print("data: n_row =", w.shape[0], "| n_chan =", nu_data.size,
      f"| {nu_data.min() / 1e6:.1f}-{nu_data.max() / 1e6:.1f} MHz")

w_chan = w.sum(axis=0, dtype="float64")  # total weight per channel
flag_frac = (w == 0).mean(axis=0)

# %%
# ---------------------------------------------------------------------------
# Channel -> model-frequency assignment, exactly as resolve does it
# (IRGSpace.binbounds + Observation.restrict_by_freq = searchsorted).
# ---------------------------------------------------------------------------
bounds = IRGSpace(nu_model).binbounds()
chan_bin = np.full(nu_data.size, -1)
for k in range(nu_model.size):
    start, stop = np.searchsorted(nu_data, [bounds[k], bounds[k + 1]])
    chan_bin[start:stop] = k
assert np.all(chan_bin >= 0), "data channels outside the sky frequency domain"

u_model = np.log(nu_model / nu_model[REF_IDX])

# %%
# ---------------------------------------------------------------------------
# 1) Per-bin table: effective frequency and in-bin spread.
# ---------------------------------------------------------------------------
print("\n=== 1) channel -> model-frequency bins ===")
hdr = (f"{'bin':>3} {'nu_k':>7} {'bounds':>15} {'data range':>15} {'n_ch':>4} {'n_ok':>4} {'flag%':>5} "
       f"{'nu_eff':>8} {'d_nu':>6} {'h_ln':>6} {'edge err a=' + str(ALPHA_STEEP):>18} {'bias':>7}")
print(hdr)
nu_eff = np.zeros(nu_model.size)
rows = []
for k in range(nu_model.size):
    m = chan_bin == k
    nu_k, wk = nu_data[m], w_chan[m]
    nu_eff[k] = np.sum(wk * nu_k) / np.sum(wk)
    ok = wk > 0
    h = 0.5 * np.log(nu_k[ok].max() / nu_k[ok].min())
    s = (nu_k[ok] / nu_model[k]) ** ALPHA_STEEP
    s_mean = np.sum(wk * (nu_k / nu_model[k]) ** ALPHA_STEEP) / np.sum(wk)
    rows.append(dict(k=k, nu=nu_model[k], nu_eff=nu_eff[k], n=m.sum(), n_ok=ok.sum(), h=h))
    print(f"{k:>3} {nu_model[k] / 1e6:7.1f} {bounds[k] / 1e6:7.1f}-{bounds[k + 1] / 1e6:<7.1f} "
          f"{nu_k.min() / 1e6:7.1f}-{nu_k.max() / 1e6:<7.1f} {m.sum():4d} {ok.sum():4d} "
          f"{100 * flag_frac[m].mean():5.1f} {nu_eff[k] / 1e6:8.2f} {(nu_eff[k] - nu_model[k]) / 1e6:+6.2f} "
          f"{h:6.4f} {100 * (s.max() - 1):+8.1f}% / {100 * (s.min() - 1):+6.1f}% {100 * (s_mean - 1):+6.2f}%")
print("nu_eff = sum(w nu) / sum(w) with the real weights (flags = 0); d_nu = nu_eff - nu_k;"
      "\nh_ln = half bin width in ln(nu); edge err = S(channel) / S(nu_k) - 1 at the bin edges"
      f"\n(per-channel model mismatch); bias = weighted bin mean / S(nu_k) - 1, all for alpha = {ALPHA_STEEP}.")


# %%
# ---------------------------------------------------------------------------
# 2) alpha / beta bias for an unresolved source (all baselines).
# ---------------------------------------------------------------------------
def binned_spectrum(alpha, beta, weights):
    """Weighted bin means of S = (nu / nu_ref)^alpha * exp(beta u^2) per model bin."""
    u = np.log(nu_data / nu_model[REF_IDX])
    s = np.exp(alpha * u + beta * u**2)
    return np.array([np.sum(weights[chan_bin == k] * s[chan_bin == k]) / np.sum(weights[chan_bin == k])
                     for k in range(nu_model.size)])


def fit_alpha_beta(s_k):
    """Log-parabola fit as in plot_from_fits.curvature_from_cube."""
    beta, alpha, _ = np.polyfit(u_model, np.log(s_k), 2)
    return alpha, beta


print("\n=== 2) recovered (alpha, beta) of a pure log-parabola, unresolved source ===")
print(f"{'alpha':>6} " + " ".join(f"{'b=' + str(b) + ': da':>11} {'db':>8}" for b in BETAS))
bias = np.zeros((len(BETAS), ALPHAS.size, 2))
for j, a in enumerate(ALPHAS):
    line = f"{a:6.2f} "
    for i, b in enumerate(BETAS):
        a_rec, b_rec = fit_alpha_beta(binned_spectrum(a, b, w_chan))
        bias[i, j] = a_rec - a, b_rec - b
        line += f"{a_rec - a:+11.4f} {b_rec - b:+8.4f} "
    print(line)

# Same with the nominal model frequencies replaced by nu_eff (no binning error
# left except second order): shows how much of the bias is the frequency offset.
a_rec, b_rec = fit_alpha_beta(binned_spectrum(ALPHA_STEEP, 0.0, w_chan))
print(f"\nalpha = {ALPHA_STEEP}, beta = 0: d_alpha = {a_rec - ALPHA_STEEP:+.4f}, d_beta = {b_rec:+.4f}")
u_eff = np.log(nu_eff / nu_model[REF_IDX])
b_eff, a_eff, _ = np.polyfit(u_eff, np.log(binned_spectrum(ALPHA_STEEP, 0.0, w_chan)), 2)
print(f"  fitted at nu_eff instead of nu_k: d_alpha = {a_eff - ALPHA_STEEP:+.4f}, d_beta = {b_eff:+.4f}")

# %%
# ---------------------------------------------------------------------------
# 3) Scale dependence: weights per channel in uv-length annuli (in lambda).
# ---------------------------------------------------------------------------
print("\n=== 3) per uv-scale (annuli in |u| = |b| nu / c) ===")
uvlen_lam_max = uvlen_m.max() * nu_data.max() / C_LIGHT
uvlen_lam_min = max(np.percentile(uvlen_m[uvlen_m > 0], 0.5) * nu_data.min() / C_LIGHT, 1.0)
uv_edges = np.geomspace(uvlen_lam_min, uvlen_lam_max, N_UV + 1)
w_uv = np.zeros((N_UV, nu_data.size))  # weight per (annulus, channel)
for c in range(nu_data.size):
    idx = np.searchsorted(uv_edges, uvlen_m * nu_data[c] / C_LIGHT) - 1
    ok = (idx >= 0) & (idx < N_UV)
    w_uv[:, c] = np.bincount(idx[ok], weights=w[ok, c], minlength=N_UV)
del w

uv_mid = np.sqrt(uv_edges[:-1] * uv_edges[1:])
scale_arcsec = 206264.806 / uv_mid
print(f"{'|u| [klambda]':>16} {'scale':>8} {'w frac':>7} " + " ".join(f"{'dnu' + str(k):>7}" for k in range(nu_model.size))
      + f" {'da':>8} {'db':>8}   (alpha = {ALPHA_STEEP}, beta = 0; dnu in MHz)")
scale_bias = np.full((N_UV, 2), np.nan)
nu_eff_uv = np.full((N_UV, nu_model.size), np.nan)
for n in range(N_UV):
    wn = w_uv[n]
    if np.any([np.sum(wn[chan_bin == k]) == 0 for k in range(nu_model.size)]):
        print(f"{uv_edges[n] / 1e3:7.2f}-{uv_edges[n + 1] / 1e3:<8.2f} {scale_arcsec[n]:7.1f}\"  (some bin without data)")
        continue
    nu_eff_uv[n] = [np.sum(wn[chan_bin == k] * nu_data[chan_bin == k]) / np.sum(wn[chan_bin == k]) for k in range(nu_model.size)]
    a_rec, b_rec = fit_alpha_beta(binned_spectrum(ALPHA_STEEP, 0.0, wn))
    scale_bias[n] = a_rec - ALPHA_STEEP, b_rec
    print(f"{uv_edges[n] / 1e3:7.2f}-{uv_edges[n + 1] / 1e3:<8.2f} {scale_arcsec[n]:7.1f}\" {wn.sum() / w_uv.sum():7.3f} "
          + " ".join(f"{(nu_eff_uv[n, k] - nu_model[k]) / 1e6:+7.2f}" for k in range(nu_model.size))
          + f" {a_rec - ALPHA_STEEP:+8.4f} {b_rec:+8.4f}")

# %%
# ---------------------------------------------------------------------------
# Figures and saved numbers.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for k in range(nu_model.size):
    m = chan_bin == k
    ax.plot(nu_data[m] / 1e6, w_chan[m] / w_chan.max(), ".", ms=3, label=f"bin {k}: {nu_model[k] / 1e6:.0f} MHz")
    ax.axvline(nu_model[k] / 1e6, color=f"C{k}", lw=1)
    ax.axvline(nu_eff[k] / 1e6, color=f"C{k}", lw=1, ls="--")
for bb in bounds:
    ax.axvline(bb / 1e6, color="gray", lw=0.5, ls=":")
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("relative channel weight")
ax.set_title("channels per bin (solid: nu_k, dashed: nu_eff)")
ax.legend(fontsize=7)

ax = axes[1]
for i, b in enumerate(BETAS):
    ax.plot(ALPHAS, bias[i, :, 1], "o-", ms=3, label=rf"$\Delta\beta$, $\beta$={b}")
    ax.plot(ALPHAS, bias[i, :, 0], "s--", ms=3, label=rf"$\Delta\alpha$, $\beta$={b}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(r"true $\alpha$")
ax.set_ylabel("recovered - true")
ax.set_title("binning bias, unresolved source")
ax.legend(fontsize=7)

ax = axes[2]
ax.semilogx(scale_arcsec, scale_bias[:, 1], "o-", label=r"$\Delta\beta$")
ax.semilogx(scale_arcsec, scale_bias[:, 0], "s--", label=r"$\Delta\alpha$")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("angular scale 1/|u| [arcsec]")
ax.set_ylabel("recovered - true")
ax.set_title(rf"per uv-scale, $\alpha$={ALPHA_STEEP}, $\beta$=0")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(odir, "binning_check.png"), dpi=200)
print("\nsaved:", os.path.join(odir, "binning_check.png"))

np.savez(
    os.path.join(odir, "binning_check.npz"),
    nu_model=nu_model, nu_eff=nu_eff, bounds=bounds, nu_data=nu_data, w_chan=w_chan, chan_bin=chan_bin,
    alphas=ALPHAS, betas=np.array(BETAS), bias=bias,
    uv_edges=uv_edges, w_uv=w_uv, nu_eff_uv=nu_eff_uv, scale_bias=scale_bias, alpha_steep=ALPHA_STEEP,
)
print("saved:", os.path.join(odir, "binning_check.npz"))

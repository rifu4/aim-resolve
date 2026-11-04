# %%
# ---------------------------------------------------------------------------
# Wiener filter example (as the NIFTy.re notebook 3_wiener_filter), in the style
# of Fig. 1 of Ensslin (2019), "Information theory for fields": a 1D Gaussian
# random field s with known power spectrum, measured with noise everywhere except
# in two small masked regions, d = R s + n. The Wiener filter gives the posterior
# mean m = D j and the posterior covariance D = (S^-1 + R^T N^-1 R)^-1; the plot
# shows m with the 1 sigma band sqrt(diag D), which widens in the gaps.
#
# Posterior mean: nifty.re.wiener_filter_posterior (CG in the
# standardised coordinates xi, s = A xi). The 1 sigma band is the exact diag(D),
# from the dense posterior covariance (N = 512 pixels).
#
# The signal realisation is that of the N_SIGNAL = 256 pixel version: its harmonic modes are drawn on
# that grid and placed at the same physical wavenumbers of the N pixel grid (the higher modes are
# drawn freshly, ~1e-4 of the field), so the signal and its correlation length on [0, 1) are kept.
#
# Fonts: Linux Libertine as in the thesis (math in Libertine italic), legend at 9 pt; figure at the thesis
# text width (380 pt = 5.26 in), so include it with width=\textwidth. The OTF files are taken from TeX Live
# or from fonts/ next to this script (copies of the TeX Live files).
#
# usage: python wiener_filter_example.py
# Output: steering/runs/wiener_filter/wiener_filter.{png,pdf}
# ---------------------------------------------------------------------------
import glob
import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

import nifty.re as jft

jax.config.update("jax_enable_x64", True)

ODIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "wiener_filter")
N = 512  # pixels of the periodic 1D domain [0, 1) = data points outside the gaps
N_SIGNAL = 256  # grid of the signal realisation
SNR = 2.5  # signal std / noise std of the realisation
GAPS = [(0.30, 0.40), (0.65, 0.70)]  # masked regions (no data)
SEED = 42

# %%
# signal prior: s = A xi, xi ~ N(0, 1), A = HT sqrt(P(k)) (harmonic transform of the amplitude spectrum)
x = np.arange(N) / N
k = np.abs(np.fft.fftfreq(N, d=1 / N))
power = 1 / (1 + (k / 4) ** 2) ** 2  # smooth field, correlation length ~ 1/(2 pi 4)
k_sig = np.abs(np.fft.fftfreq(N_SIGNAL, d=1 / N_SIGNAL))
norm = (1 / (1 + (k_sig[1:] / 4) ** 2) ** 2).sum()  # normalisation of the N_SIGNAL grid (same amplitudes)
amp = jnp.asarray(np.sqrt(power / norm) * 0.26)  # field std ~0.3: the data stay within [-1, 1]


class Signal(jft.Model):
    def __init__(self):
        super().__init__(domain=jax.ShapeDtypeStruct((N,), jnp.float64))

    def __call__(self, xi):
        return jft.correlated_field.hartley(amp * xi)


signal = Signal()
mask = np.ones(N, dtype=bool)
for a, b in GAPS:
    mask &= ~((x >= a) & (x < b))
response = lambda s: s[mask]  # noqa: E731
signal_response = jft.Model(lambda xi: response(signal(xi)), domain=signal.domain)

key = jax.random.PRNGKey(SEED)
key, k_s, k_n, k_wf = jax.random.split(key, 4)
xi_sig = jft.random_like(k_s, jax.ShapeDtypeStruct((N_SIGNAL,), jnp.float64))
idx = np.round(np.fft.fftfreq(N_SIGNAL, d=1 / N_SIGNAL)).astype(int) % N  # same wavenumbers on the N grid
xi_true = jft.random_like(jax.random.fold_in(k_s, 1), signal.domain).at[idx].set(xi_sig)
s_true = np.asarray(signal(xi_true))
NOISE_STD = s_true.std() / SNR
data = response(s_true) + NOISE_STD * jax.random.normal(k_n, (mask.sum(),))

# %%
# Wiener filter: posterior mean with NIFTy
lh = jft.Gaussian(data, noise_cov_inv=lambda r: r / NOISE_STD**2).amend(signal_response)
samples, _ = jft.wiener_filter_posterior(
    lh, key=k_wf, n_samples=0,
    draw_linear_kwargs=dict(cg_name="WF", cg_kwargs=dict(absdelta=1e-10, maxiter=1000)))
s_mean = np.asarray(signal(samples.pos))

# exact posterior covariance in signal space: D = A (1 + A^T R^T N^-1 R A)^-1 A^T
A = np.asarray(jax.vmap(signal, in_axes=1, out_axes=1)(jnp.eye(N)))
RA = A[mask]
D = A @ np.linalg.solve(np.eye(N) + RA.T @ RA / NOISE_STD**2, A.T)
s_std = np.sqrt(np.diag(D))
m_exact = D[:, mask] @ np.asarray(data) / NOISE_STD**2
print(f"max |m_nifty - m_exact| = {np.abs(s_mean - m_exact).max():.2e}")
print(f"posterior std: {np.median(s_std[mask]):.3f} with data, {s_std[~mask].max():.3f} max in the gaps "
      f"(prior std {np.sqrt(np.diag(A @ A.T)).mean():.3f}, noise std {NOISE_STD})")
z = (s_true - s_mean) / s_std
print(f"{mask.sum()} data points, noise std {NOISE_STD:.3f}, max |d| {np.abs(data).max():.3f}")
print(f"|s - m| < 1 sigma for {np.mean(np.abs(z) < 1):.2f} of the pixels")

# %%
FONT_DIRS = sorted(glob.glob("/usr/local/texlive/*/texmf-dist/fonts/opentype/public/libertine"))[::-1] + [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")]
font_dir = next(d for d in FONT_DIRS if os.path.isfile(f"{d}/LinLibertine_R.otf"))
for f in ("LinLibertine_R.otf", "LinLibertine_RI.otf", "LinLibertine_RB.otf"):
    fm.fontManager.addfont(f"{font_dir}/{f}")
plt.rcParams.update({
    "font.family": "Linux Libertine O", "mathtext.fontset": "custom", "mathtext.rm": "Linux Libertine O",
    "mathtext.it": "Linux Libertine O:italic", "mathtext.bf": "Linux Libertine O:bold",
    "font.size": 9, "legend.fontsize": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(5.26, 5.26 * 9 / 16))  # thesis text width (380 pt), 16:9: fonts at true size
for a, b in GAPS:
    ax.axvspan(a, b, color="0.92", lw=0, zorder=0)
ax.fill_between(x, s_mean - s_std, s_mean + s_std, color="#2a78c8", alpha=0.18, lw=0, label=r"$1\sigma$")
ax.plot(x[mask], data, "o", ms=1.8, color="0.65", mew=0, zorder=1, label=r"$d$")
ax.plot(x, s_true, color="k", lw=1.0, zorder=3, label=r"$s$")
ax.plot(x, s_mean, color="#2a78c8", lw=1.3, zorder=4, label=r"$m$")
ax.set_xlim(0, 1)
lo = min(np.min(data), np.min(s_mean - s_std), s_true.min())
hi = max(np.max(data), np.max(s_mean + s_std), s_true.max())
ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))  # tight around everything plotted
ax.set_xticks([])
ax.set_yticks([])
h, lab = ax.get_legend_handles_labels()
order = [2, 1, 3, 0]  # signal, data, mean, band
leg = ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper right", fancybox=False, edgecolor="k",
                framealpha=1, markerscale=1.6, handlelength=1.5, labelspacing=0.3)
leg.get_frame().set_linewidth(0.8)
fig.tight_layout()
os.makedirs(ODIR, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{ODIR}/wiener_filter.{ext}", dpi=200, bbox_inches="tight")
print("saved:", os.path.abspath(f"{ODIR}/wiener_filter.png"))

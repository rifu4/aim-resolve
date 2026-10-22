# %%
# ---------------------------------------------------------------------------
# Phases 5 + 6 of the synthetic-data test (../synthetic_test_plan.md):
# posterior samples of a synthetic run -> alpha / beta, compared with the truth.
#
# Definitions (as in plot_from_fits.py):
#   beta        quadratic coefficient of the log-parabola fit (curvature_from_cube)
#               ln S = a + alpha u + beta u^2, u = ln(nu / nu_ref), per sample
#   alpha_model the model's spectral index (paper's alpha); in the model the
#               deviations have no linear part, so it is the slope of the line
#               through the reference point, sum((ln S - ln S_ref) u) / sum(u^2)
#               -> compared with the same slope of the truth cube (alpha_lin)
#   beta_model  the model's spectral curvature (models with spectral_curvature /
#               beta), the u^2 coefficient -> compared with the true beta
# Truth and reconstruction use the same fits at the same frequencies (nu_eff).
# Uncertainty: std over the posterior samples (ddof = 1, as MySamples.std).
#
# Evaluated over the whole field and in the flux mask (reconstructed flux at
# nu_ref > FLUX_MIN mJy/arcsec^2, the paper's threshold for ESO 137-006), the
# latter also in flux bins. z-scores on pixels one beam apart.
#
# usage: python compare_synthetic.py A [C ...]                     (runs in RUNS/<variant>, sky.1)
#        python compare_synthetic.py --run DIR [--sky sky.0] [--truth main] [--label NAME]
#        --no-maps: skip maps_mask.png (e.g. for the image tests, whose maps come from plot_image_test.py)
# Output: RUNS/<variant>/compare/: posterior.npz, FITS of the means, plots (flux mask only), summary.json
# ---------------------------------------------------------------------------
import json
import os
import pickle
import sys

import astropy.io.fits as pyfits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import OUT, PREFIX, REF_IDX, RUNS, STEERING, VARIANTS, TruthSky

import aim_resolve as aim

AS2RAD = np.pi / 180 / 3600
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2
FLUX_MIN = 1e-2  # mJy/arcsec^2, plot_from_fits.py COMPONENTS["O0"]["flux_min"]
BEAM_PIX = 3  # decimation for z-scores (~ one beam of ~7" at 2.34" pixels)
N_FLUX_BINS = 4

if "--run" in sys.argv:
    def _arg(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    run_dir = os.path.abspath(_arg("--run", None))
    jobs = [dict(run=run_dir, sky=_arg("--sky", "sky.1"), truth=_arg("--truth", "main"),
                 label=_arg("--label", os.path.basename(run_dir)))]
else:
    jobs = [dict(run=f"{RUNS}/{v}", sky="sky.1", truth=VARIANTS[v][0], label=v) for v in (sys.argv[1:] or ["A", "C"])]


def curvature(cube, u):
    """Per-pixel log-parabola fit ln S = a + alpha u + beta u^2 -> beta (as curvature_from_cube)."""
    ln = np.log(np.clip(cube, 1e-12, None)).reshape(cube.shape[0], -1)
    return np.polyfit(u, ln, 2)[0].reshape(cube.shape[1:])


def anchored_slope(cube, u):
    """Slope of the line through the reference point (u = 0), as the model's alpha."""
    ln = np.log(np.clip(cube, 1e-12, None))
    return np.tensordot(u, ln - ln[REF_IDX], axes=1) / np.sum(u**2)


def stats(d, z):
    """Bias / scatter of the error d and calibration of z = d / sigma."""
    d, z = d[np.isfinite(d)], z[np.isfinite(z)]
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    return dict(n=int(d.size), median=float(np.median(d)), mean=float(np.mean(d)), mad_std=float(mad),
                rms=float(np.sqrt(np.mean(d**2))), z_std=float(np.std(z)), z_median=float(np.median(z)),
                cover_1sig=float(np.mean(np.abs(z) < 1)), cover_2sig=float(np.mean(np.abs(z) < 2)))


for job in jobs:
    run, truth_name, v = job["run"], job["truth"], job["label"]
    odir = f"{run}/compare"
    os.makedirs(odir, exist_ok=True)
    print(f"\n=== {v} (truth {truth_name}, {job['sky']}) ===")

    # %%
    # -----------------------------------------------------------------------
    # Posterior samples -> cubes, alpha, beta.
    # -----------------------------------------------------------------------
    cfg = aim.OptimizeKLConfig.from_file((f"{STEERING}/nifty/config/base.yml", f"{run}/config.yml"), aim.get_builders)
    sky = cfg.instantiate_sec(job["sky"])
    has_curv = sky.background.has_spectral_curvature
    freq = np.asarray(sky.freq, dtype="float64")
    u = np.log(freq / freq[REF_IDX])
    with open(f"{run}/last.pkl", "rb") as f:
        samples, *_ = pickle.load(f)
    print("samples:", len(samples), "| model freqs [MHz]:", np.round(freq / 1e6, 2).tolist())

    cubes, a_model, b_lp, b_model = [], [], [], []
    for x in samples:
        c = np.asarray(sky(x), dtype="float64")  # intrinsic sky (nfreq, N, N) [Jy/sr]
        cubes.append(c)
        a_model.append(np.asarray(sky.background.spectral_index(x), dtype="float64"))
        b_lp.append(curvature(c, u))
        if has_curv:
            b_model.append(np.asarray(sky.background.spectral_curvature(x), dtype="float64"))
    cubes, a_model, b_lp = (np.stack(t) for t in (cubes, a_model, b_lp))
    rec = dict(
        flux=cubes[:, REF_IDX].mean(0) * CONV_FACTOR,
        flux_std=cubes[:, REF_IDX].std(0, ddof=1) * CONV_FACTOR,
        alpha_model=a_model.mean(0), alpha_model_std=a_model.std(0, ddof=1),
        beta=b_lp.mean(0), beta_std=b_lp.std(0, ddof=1),
    )
    if has_curv:
        b_model = np.stack(b_model)
        rec.update(beta_model=b_model.mean(0), beta_model_std=b_model.std(0, ddof=1))
    print("model with spectral curvature:", has_curv)

    t = TruthSky(truth_name)
    t_cube = np.stack([t(f) for f in freq])
    t_beta = curvature(t_cube, u)
    truth = dict(flux=t_cube[REF_IDX] * CONV_FACTOR, alpha=t.alpha, beta=t_beta, alpha_lin=anchored_slope(t_cube, u))
    print(f"truth fit check: max |beta_fit - beta| {np.max(np.abs(t_beta - t.beta)):.1e} (exact log-parabola)")

    np.savez(f"{odir}/posterior.npz", freq=freq, **{f"rec_{k}": val for k, val in rec.items()},
             **{f"truth_{k}": val for k, val in truth.items()}, alpha_model_k=a_model, beta_k=b_lp,
             **(dict(beta_model_k=b_model) if has_curv else {}))
    maps_out = [("flux_mean", rec["flux"]), ("alpha_model_mean", rec["alpha_model"]),
                ("beta_mean", rec["beta"]), ("beta_std", rec["beta_std"])]
    if has_curv:
        maps_out += [("beta_model_mean", rec["beta_model"]), ("beta_model_std", rec["beta_model_std"])]
    for name, arr in maps_out:
        pyfits.PrimaryHDU(np.asarray(arr, dtype="float32").T).writeto(f"{odir}/{PREFIX}_{name}.fits", overwrite=True)

    # %%
    # -----------------------------------------------------------------------
    # Errors, masks and statistics.
    # -----------------------------------------------------------------------
    err = {
        "alpha_model": (rec["alpha_model"] - truth["alpha_lin"], rec["alpha_model_std"]),
        "beta": (rec["beta"] - truth["beta"], rec["beta_std"]),
    }
    if has_curv:
        err["beta_model"] = (rec["beta_model"] - truth["beta"], rec["beta_model_std"])
    regions = {"whole field": np.ones_like(rec["flux"], bool), f"flux > {FLUX_MIN:g}": rec["flux"] > FLUX_MIN}
    m = regions[f"flux > {FLUX_MIN:g}"]
    edges = np.quantile(rec["flux"][m], np.linspace(0, 1, N_FLUX_BINS + 1))
    for i in range(N_FLUX_BINS):
        regions[f"flux bin {i + 1} [{edges[i]:.2g}, {edges[i + 1]:.2g}]"] = m & (rec["flux"] >= edges[i]) & (
            rec["flux"] <= edges[i + 1])
    dec = np.zeros_like(m)
    dec[::BEAM_PIX, ::BEAM_PIX] = True

    summary = dict(variant=v, run=run, truth=truth_name, n_samples=len(samples), flux_min=FLUX_MIN,
                   spectral_curvature=bool(has_curv), regions={})
    print(f"{'region':34s} {'qty':12s} {'n':>7s} {'median':>8s} {'MAD':>7s} {'rms':>7s} {'z_std':>6s} {'<1s':>5s} {'<2s':>5s}")
    for rname, rm in regions.items():
        summary["regions"][rname] = {}
        for q, (d, s) in err.items():
            st = stats(d[rm], (d / s)[rm & dec])
            summary["regions"][rname][q] = st
            print(f"{rname:34s} {q:12s} {st['n']:7d} {st['median']:+8.3f} {st['mad_std']:7.3f} {st['rms']:7.3f} "
                  f"{st['z_std']:6.2f} {st['cover_1sig']:5.2f} {st['cover_2sig']:5.2f}")
        rb = rec["beta"][rm]
        summary["regions"][rname]["beta_rec"] = dict(median=float(np.median(rb)), mean=float(np.mean(rb)))
        summary["regions"][rname]["beta_std_median"] = float(np.median(rec["beta_std"][rm]))
    # binning-bias signature: Delta beta vs true alpha in the flux mask
    slope = np.polyfit(truth["alpha"][m], err["beta"][0][m], 1)[0]
    summary["dbeta_vs_true_alpha_slope_in_mask"] = float(slope)
    print(f"d(Delta beta)/d(alpha_true) in the mask: {slope:+.4f}")
    with open(f"{odir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # %%
    # -----------------------------------------------------------------------
    # Plots, all restricted to the flux mask (the whole-field numbers are in
    # summary.json): maps, scatter, z histograms, error vs flux.
    # -----------------------------------------------------------------------
    mask_label = f"flux > {FLUX_MIN:g} mJy/arcsec$^2$"
    # (quantity, truth key, label, map limits, scatter limits)
    # with a curvature term beta is plotted from the model only (the log-parabola fit stays in summary.json)
    quantities = [("alpha_model", "alpha_lin", "alpha (model)", (-2.5, 0.5), (-3, 1))]
    if has_curv:
        quantities.append(("beta_model", "beta", "beta (model)", (-1, 1), (-1.5, 1.5)))
    else:
        quantities.append(("beta", "beta", "beta (log-parabola fit)", (-1, 1), (-1.5, 1.5)))
    for tag, rm in ([] if "--no-maps" in sys.argv else [("mask", m)]):
        fig, axes = plt.subplots(len(quantities), 4, figsize=(20, 4.9 * len(quantities)))
        rows = [(lab, truth[tq], rec[q], rec[f"{q}_std"], "RdYlBu_r" if q.startswith("alpha") else "coolwarm", lim, 0.5)
                for q, tq, lab, lim, _ in quantities]
        for r, (name, tr, rc, sd, cmap, lim, dlim) in enumerate(rows):
            for c, (arr, title, kw) in enumerate([
                (tr, f"truth {name}", dict(cmap=cmap, vmin=lim[0], vmax=lim[1])),
                (rc, f"reconstruction {name}", dict(cmap=cmap, vmin=lim[0], vmax=lim[1])),
                (rc - tr, f"rec - truth", dict(cmap="PuOr", vmin=-dlim, vmax=dlim)),
                (sd, f"posterior std", dict(cmap="viridis", vmin=0, vmax=dlim)),
            ]):
                ax = axes[r, c]
                im = ax.imshow(np.where(rm, arr, np.nan).T, origin="lower", **kw)
                ax.contour(rec["flux"].T, levels=[FLUX_MIN, 1e-1, 1], colors="k", linewidths=0.4)
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"variant {v}: {mask_label}")
        fig.tight_layout()
        fig.savefig(f"{odir}/maps_{tag}.png", dpi=110)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(quantities), figsize=(5.5 * len(quantities), 5))
    for c, (q, tq, _, _, lim) in enumerate(quantities):
        ax = axes[c]
        ax.hexbin(truth[tq][m], rec[q][m], gridsize=80, bins="log", extent=(*lim, *lim), cmap="Blues")
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"truth {tq}")
        ax.set_ylabel(f"reconstruction {q}")
    fig.suptitle(f"variant {v}: {mask_label}")
    fig.tight_layout()
    fig.savefig(f"{odir}/scatter.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(quantities), figsize=(5.5 * len(quantities), 8.5))
    zz = np.linspace(-5, 5, 200)
    for c, (q, *_) in enumerate(quantities):
        d, s = err[q]
        ax = axes[0, c]
        z = (d / s)[m & dec]
        ax.hist(np.clip(z, -5, 5), 80, range=(-5, 5), density=True, histtype="step",
                label=f"(std {np.std(z):.2f}, |z|<1: {np.mean(np.abs(z) < 1):.2f})")
        ax.plot(zz, np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8, label="N(0,1)")
        ax.set_title(f"z = (rec - truth) / sigma, {q}")
        ax.legend(fontsize=8)
        ax = axes[1, c]
        lf = np.log10(np.clip(rec["flux"], 1e-6, None))
        bins = np.linspace(np.log10(FLUX_MIN), np.log10(rec["flux"].max()), 16)
        idx = np.where(m, np.digitize(lf, bins), 0)
        med = [np.median(d[idx == i]) if np.sum(idx == i) > 50 else np.nan for i in range(1, bins.size)]
        p16 = [np.percentile(d[idx == i], 16) if np.sum(idx == i) > 50 else np.nan for i in range(1, bins.size)]
        p84 = [np.percentile(d[idx == i], 84) if np.sum(idx == i) > 50 else np.nan for i in range(1, bins.size)]
        sm = [np.median(s[idx == i]) if np.sum(idx == i) > 50 else np.nan for i in range(1, bins.size)]
        bc = 10 ** (0.5 * (bins[1:] + bins[:-1]))  # bin centres in mJy/arcsec^2 (log axis, as the sky plots)
        ax.fill_between(bc, p16, p84, alpha=0.3, label="16-84 % of rec - truth")
        ax.plot(bc, med, "o-", ms=3, label="median rec - truth")
        ax.plot(bc, sm, "k:", label="median posterior std")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("reconstructed flux at nu_ref [mJy/arcsec$^2$]")
        ax.set_ylabel(f"error in {q}")
        ax.legend(fontsize=7)
    fig.suptitle(f"variant {v}: {mask_label}")
    fig.tight_layout()
    fig.savefig(f"{odir}/errors.png", dpi=110)
    plt.close(fig)
    print("saved:", odir)

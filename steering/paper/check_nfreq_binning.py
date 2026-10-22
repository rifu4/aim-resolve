# %%
# ---------------------------------------------------------------------------
# Binning error of the spectral curvature beta for different numbers / placements of
# the model-frequency bins (referee question on the channel binning).
#
# The data channels of each bin are averaged with their weights (as the gridding does);
# the model frequency of a bin is its effective frequency sum(w nu) / sum(w). For a pure
# power law (beta_true = 0) the log-parabola fit to the binned spectrum gives the binning
# error d beta. It is computed for the channel weights of compact emission (w_chan) and
# for the weights of each uv-scale ring (w_uv, extended emission), using the real
# weights and flags (binning_check.npz of check_channel_binning.py).
#
# Binnings: equal-width bins per band (6, 8, 12, 16 frequencies), the current 6f bins,
# 8f with the edges placed by hand at the weight discontinuities, and 8f with the edges
# optimised (coordinate descent on the uv-weighted mean |d beta|, averaged over alpha).
#
# usage: python check_nfreq_binning.py
# ---------------------------------------------------------------------------
import numpy as np

NPZ = ("/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b/opt/"
       "4_rec_3z_1_6f_1/binning_check/binning_check.npz")
ALPHAS = (-0.7, -1.5, -2.5)
MIN_CH = 25  # minimum number of channels per bin (optimised binning)

z = np.load(NPZ)
nu, wc, wuv, uv_edges = z["nu_data"], z["w_chan"], z["w_uv"], z["uv_edges"]
scale = 206265 / np.sqrt(uv_edges[:-1] * uv_edges[1:])  # arcsec
frac = wuv.sum(1) / wuv.sum()
W = np.vstack([wc, wuv])  # row 0: compact, rows 1...: uv rings
GAP = int(np.searchsorted(nu, 1200e6))  # first channel of the high band
N = nu.size


def edges_from_mhz(bounds_mhz):
    """Channel-index edges [0, ..., N] from bin bounds in MHz (channels with bound_k <= nu < bound_k+1)."""
    e = np.searchsorted(nu, np.asarray(bounds_mhz, dtype=float) * 1e6)
    e[0], e[-1] = 0, N
    return e


def equal(n_per_band):
    lo = np.linspace(961.43, 1145.3, n_per_band + 1)
    hi = np.linspace(1294.8, 1503.46, n_per_band + 1)
    return edges_from_mhz(np.concatenate([lo, hi[1:]]))  # the bound between the bands lies in the gap


def dbeta(e, alpha):
    """d beta per weighting (row 0 compact, then uv rings); NaN where a bin has no weight."""
    s = (nu / 1e9) ** alpha
    num = np.add.reduceat(W * s, e[:-1], axis=1)
    den = np.add.reduceat(W, e[:-1], axis=1)
    nu_eff = np.add.reduceat(wc * nu, e[:-1]) / np.add.reduceat(wc, e[:-1])
    u = np.log(nu_eff / 1e9)
    A = np.stack([u**2, u, np.ones_like(u)], axis=1)
    ok = np.all(den > 0, axis=1)
    out = np.full(W.shape[0], np.nan)
    out[ok] = np.linalg.lstsq(A, np.log(num[ok] / den[ok]).T, rcond=None)[0][0]
    return out


def metrics(e, alpha):
    d = dbeta(e, alpha)
    per, ok = d[1:], np.isfinite(d[1:])
    return dict(compact=d[0], wmean=np.sum(np.abs(per[ok]) * frac[ok]) / frac[ok].sum(),
                large=np.nanmax(np.abs(per[(scale > 60) & (scale < 260)])),
                small=np.nanmax(np.abs(per[(scale > 10) & (scale < 41)])))


def cost(e):
    return np.mean([metrics(e, a)["wmean"] for a in ALPHAS])


def optimise(n_lo, n_hi, n_restart=20, seed=0):
    """Coordinate descent over the interior edges of each band (the band bound stays at GAP)."""
    rng = np.random.default_rng(seed)
    best_e, best_c = None, np.inf
    for r in range(n_restart):
        if r == 0:
            e = np.concatenate([np.linspace(0, GAP, n_lo + 1), np.linspace(GAP, N, n_hi + 1)[1:]]).round().astype(int)
        else:
            lo = np.sort(rng.choice(np.arange(MIN_CH, GAP - MIN_CH), n_lo - 1, replace=False))
            hi = np.sort(rng.choice(np.arange(GAP + MIN_CH, N - MIN_CH), n_hi - 1, replace=False))
            e = np.concatenate([[0], lo, [GAP], hi, [N]])
        if np.any(np.diff(e) < MIN_CH):
            continue
        c = cost(e)
        for _ in range(20):
            improved = False
            for i in range(1, e.size - 1):
                if e[i] == GAP:
                    continue
                for j in range(e[i - 1] + MIN_CH, e[i + 1] - MIN_CH + 1):
                    trial = e.copy()
                    trial[i] = j
                    ct = cost(trial)
                    if ct < c - 1e-9:
                        e, c, improved = trial, ct, True
            if not improved:
                break
        if c < best_c:
            best_e, best_c = e, c
    return best_e


def bounds_mhz(e):
    """Bin bounds in MHz: midpoints between the neighbouring channels (outer: band edges)."""
    b = [961.43] + [0.5 * (nu[i - 1] + nu[i]) / 1e6 for i in e[1:-1]] + [1503.46]
    return np.array(b)


# %%
binnings = {
    "6 equal": equal(3),
    "6 current": edges_from_mhz([961.43, 1022.87, 1084.11, 1222.04, 1364.23, 1433.82, 1503.46]),
    "8 equal (mf_ext_8f)": equal(4),
    # by hand at the weight steps: low band 1052.3 (dip ch 108), 1086 (start of the RFI-affected part);
    # high band 1302.3 is too close to the band edge, so 1361 (middle), 1419.3 (before the HI dip),
    # 1462.0 (weight step 1.07 -> 0.56)
    "8 at weight steps": edges_from_mhz([961.43, 1008.0, 1052.35, 1085.8, 1200.0, 1361.0, 1419.3, 1461.95, 1503.46]),
}
for nl, nh in [(4, 4), (3, 5), (5, 3)]:
    binnings[f"8 optimised ({nl}+{nh})"] = optimise(nl, nh)
binnings["12 equal"] = equal(6)
binnings["16 equal"] = equal(8)

print(f"binning error of beta for a power law (beta_true = 0); mean over alpha = {ALPHAS} and alpha = -2.5 only\n")
hdr = (f"{'binning':24s} {'n_ch per bin':>26s} | {'compact':>8s} {'uv-wmean':>8s} {'60-250\"':>8s} {'10-40\"':>8s}"
       f" | {'a=-2.5: uv-wmean':>16s} {'60-250\"':>8s} {'10-40\"':>8s}")
print(hdr)
print("-" * len(hdr))
for name, e in binnings.items():
    m = [metrics(e, a) for a in ALPHAS]
    avg = {k: np.mean([abs(x[k]) for x in m]) for k in m[0]}
    m25 = metrics(e, -2.5)
    nch = ",".join(str(x) for x in np.diff(e))
    print(f"{name:24s} {nch:>26s} | {avg['compact']:8.4f} {avg['wmean']:8.3f} {avg['large']:8.3f} {avg['small']:8.3f}"
          f" | {m25['wmean']:16.3f} {m25['large']:8.3f} {m25['small']:8.3f}")
print("\nbin bounds [MHz] of the 8-frequency binnings:")
for name, e in binnings.items():
    if name.startswith("8"):
        print(f"  {name:24s} {np.array2string(bounds_mhz(e), precision=1, max_line_width=200)}")

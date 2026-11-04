# SF (7 channels) against MF (468 channels) visibility scale: weighted means in uv cells (lambda) on short baselines
import numpy as np
D = "/scratch/users/rfuchs/data"; C = 299792458.
def load(fn):
    d = np.load(fn); uvw = d["antpos0"]; m = np.hypot(uvw[:, 0], uvw[:, 1]) < 200.
    return d["freq"], uvw[m], d["vis"][0][m], d["weight"][0][m], d["antpos3"]
fa, ua, va, wa, ta = load(f"{D}/eso_986-1137mhz_wfix.npz")
fb, ub, vb, wb, tb = load(f"{D}/eso_961-1503mhz_wfix.npz")
print(f"short rows SF {len(ua)} MF {len(ub)} | time SF {ta.min():.0f}-{ta.max():.0f} MF {tb.min():.0f}-{tb.max():.0f} | offset {tb.min() - ta.min():.0f} s", flush=True)
CELL, UMAX, NV = 15., 600., 100
def grid(uvw, vis, w, f):
    u, v = uvw[:, 0] * f / C, uvw[:, 1] * f / C
    s = np.where(u < 0, -1, 1); u, v = u * s, v * s; vis = np.where(s < 0, np.conj(vis), vis)
    ok = (w > 0) & (np.hypot(u, v) < UMAX)
    idx = np.floor(u[ok] / CELL).astype(int) * NV + np.floor((v[ok] + UMAX) / CELL).astype(int)
    n = int(UMAX / CELL + 1) * NV
    wv = w[ok] * vis[ok]
    return np.bincount(idx, wv.real, n) + 1j * np.bincount(idx, wv.imag, n), np.bincount(idx, w[ok], n)
half = np.median(np.diff(fa)) / 2
for c, f in enumerate(fa):
    sva, swa = grid(ua, va[:, c], wa[:, c], f)
    sel = np.where(np.abs(fb - f) <= half)[0]; svb = swb = 0
    for j in sel:
        x, y = grid(ub, vb[:, j], wb[:, j], fb[j]); svb = svb + x; swb = swb + y
    m = (swa > 0) & (swb > 0)
    A, B = sva[m] / swa[m], svb[m] / swb[m]; W = 1 / (1 / swa[m] + 1 / swb[m])
    r = np.sum(W * A * np.conj(B)) / np.sum(W * np.abs(B) ** 2)
    wsa = np.median(wa[:, c][wa[:, c] > 0]); wsb = np.median(wb[:, sel][wb[:, sel] > 0])
    print(f"SF ch {f / 1e6:7.1f}: MF chans {sel.size:2d} | cells {m.sum():5d} | SF/MF complex {r.real:+.3f}{r.imag:+.3f}j"
          f" | median |SF|/|MF| {np.median(np.abs(A)) / np.median(np.abs(B)):.3f} | weight per vis SF/MF {wsa / wsb:.1f}", flush=True)

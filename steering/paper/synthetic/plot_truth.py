# truth plots in the layout of the callback plots (plot_kl_callback): {odir}/plots/true_sky.0.png (SF: apparent
# brightness at 1062 MHz, as edge_check), true_sky.1.png / _alpha / _beta (MF: intrinsic at the reference frequency)
# usage: plot_truth.py RUN TRUTH
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ST}/paper/synthetic"); os.chdir(ST)
from synth_common import TruthSky
import aim_resolve as aim
from aim_resolve.plot.panels import _alpha_cmap, _each, _finite
from aim_resolve.fast_resolve.beam import build_primary_beam

run, truth = sys.argv[1:3]
p = f"runs/synth_1deg/r10_sf5_mf15/{run}"
odir = f"{p}/plots"
cfg = aim.OptimizeKLConfig.from_file(("nifty/config/base.yml", f"{p}/config_full.yml"), aim.get_builders)
theme, cfg_range, pnl = aim.split_style(cfg.sections.get("base_style"))
dynamic_range = cfg_range if cfg.sections.get("base_style", {}).get("dynamic_range") is not None else 1e5
spectral_range = 1e3
ts = TruthSky(truth)
names = ("bg", "o0", "o1")


def boxes(sky):
    """Pixel slices of each component box on the sky grid (support of the mapped component)."""
    out = {}
    for md in sky.models[1:]:
        img = np.asarray(md(md.init(jax.random.PRNGKey(0)), map=True), dtype="float64")
        img = img.reshape((-1,) + img.shape[-2:])[0]
        ix, iy = np.nonzero(img)
        out[md.prefix.split(".")[0]] = (slice(ix.min(), ix.max() + 1), slice(iy.min(), iy.max() + 1))
    return out


def comps_at(nu):
    u = np.log(nu / ts.nu_ref)
    return {c: i0 * np.exp(a * u + b * u**2) for c, (i0, a, b) in zip(names, ts.comps)}


for sec in ("sky.0", "sky.1"):
    sky = cfg.instantiate_sec(sec)
    freq = np.asarray(sky.freq, dtype="float64").ravel()
    sf = freq.size == 1
    nu = 1062e6 if sf else ts.nu_ref
    tc = comps_at(nu)
    if sf:  # apparent brightness, as the SF model
        pb = np.asarray(build_primary_beam(sky.grid, [nu], channel_average=False), dtype="float64").squeeze()
        tc = {c: v * pb for c, v in tc.items()}
    bx = boxes(sky)
    tot = sum(tc.values())
    labels = [f"true {sky.prefix}", f"true {sky.models[0].prefix}"] + [f"true {md.prefix}" for md in sky.models[1:]]
    order = ["bg"] + [md.prefix.split(".")[0] for md in sky.models[1:]]
    refs = [tot, tc["bg"]] + [tc[c][bx[c]] for c in order[1:]]
    vmin, vmax = aim.dynamic_limits(refs[0], dynamic_range)
    step = f"truth {truth} · {sky.prefix}"
    with aim.plot_style(**theme):
        aim.plot_panels([dict(arrays=refs, labels=labels, cmap="inferno", vmin=vmin, vmax=vmax)], name=f"true_{sky.prefix}.png",
                        odir=odir, step=f"{step} · {nu / 1e6:.1f} MHz{' apparent' if sf else ' · reference frequency'}", **pnl)
    print("wrote", f"{odir}/true_{sky.prefix}.png")
    if sf:
        continue

    # spectral maps as in plot_kl_callback: components above max / spectral_range, the sky composed of the
    # brightest component above the cut, the background as it is; contours of the brightness
    cut = vmax / spectral_range
    overlays = [_each(lambda r: dict(contours=[r > vmax / d for d in (1e3, 1e2, 10)], lw=0.4), r) for r in refs]
    cidx = {c: k for k, c in enumerate(names)}
    for attr, kind in ((1, "alpha"), (2, "beta")):
        vals = {c: ts.comps[cidx[c]][attr] for c in names}
        cc = order[1:]
        bright = np.stack([np.where(tc[c] > cut, tc[c], -np.inf) for c in cc])
        best = np.argmax(bright, axis=0)
        stack = np.stack([vals[c] for c in cc])
        sky_map = np.where(np.isfinite(bright.max(0)), np.take_along_axis(stack, best[None], 0)[0], np.nan)
        maps = [sky_map, vals["bg"]] + [np.where(tc[c][bx[c]] > cut, vals[c][bx[c]], np.nan) for c in cc]
        finite = _finite([m for i, m in enumerate(maps) if i != 1])
        if kind == "alpha":
            lo, hi = (float(v) for v in np.percentile(finite, [0.1, 99.9]))
            group = dict(cmap=_alpha_cmap(lo, hi), vmin=lo, vmax=hi, cbar_label="spectral index")
        else:
            lim = float(np.percentile(np.abs(finite), 99)) or 1.0
            group = dict(cmap="coolwarm", vmin=-lim, vmax=lim, cbar_label="spectral curvature")
        group.update(arrays=maps, labels=labels, norm="linear", overlays=overlays, bad_color="none")
        with aim.plot_style(**theme):
            aim.plot_panels([group], name=f"true_{sky.prefix}_{kind}.png", odir=odir,
                            step=f"{step} · {'spectral index' if kind == 'alpha' else 'spectral curvature'}", **pnl)
        print("wrote", f"{odir}/true_{sky.prefix}_{kind}.png")

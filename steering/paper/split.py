# %%
# ---------------------------------------------------------------------------
# Split-convolution study for the fast-resolve response.
#
# The fast-resolve response convolves a sky image of shape (n, n) with a PSF
# kernel of shape (2n, 2n). The single convolution pads the sky to (2n, 2n)
# and FFTs at that shape. The two-step (split) convolution replaces this by a
# small high-resolution transform (sky padded to (n + k, n + k)) plus a coarse
# low-resolution transform (everything downsampled by a factor z), see
# `PSFConvolve` / `PSFSplitConvolve` / `build_split_kernel` in
# `aim_resolve/fast_resolve/convolve.py`.
#
# This script visualises the trade-off of that scheme (pure numpy/matplotlib).
# The 2-D image plots come first, the 1-D curves after:
#   * Plots 1-2: the 2-D PSF kernel (full and central-half zoom);
#   * Plots 3-4: the 2-D downsampling error outside the central high-resolution
#     crop of size k (full and zoom) -- the low-res branch represents the kernel
#     by a block-averaged (downsampled-then-upsampled) copy;
#   * Plot 5: the zoomed kernel and its error side by side;
#   * Plot 6: the analytic memory footprint of the split vs single convolution;
#   * Plot 7: the max downsampling error outside the crop vs k;
#   * Plot 8: the memory and the max error side by side.
# ---------------------------------------------------------------------------
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
# Full-sky image resolution per dimension (sky shape is (n, n), kernel (2n, 2n)).
N = 3072

# Down-sampling (zoom) factors for the low-resolution branch.
Z_FACTORS = (2, 3, 4)

# Discrete central-kernel sizes k: 128 to 1536 in steps of 128.
K_SIZES = tuple(range(128, 1536, 128))

# Continuous range of k for the curves.
K_GRID = np.linspace(K_SIZES[0], K_SIZES[-1], 512)

# Central-kernel sizes swept for the error-outside curves (Plots 7/8).
K_SPLIT = K_SIZES

# Matplotlib colors per zoom factor (matching the original plot).
COLORS = {2: "tab:blue", 3: "lightgray", 4: "crimson"}

# Positive half of RdBu_r (white -> red), so the error maps share the exact red
# ramp of the diverging kernel map's positive side.
REDS_HALF = LinearSegmentedColormap.from_list(
    "RdBu_r_pos", plt.get_cmap("RdBu_r")(np.linspace(0.5, 1.0, 256))
)

ODIR = os.path.join(os.path.dirname(__file__), "plots")

# ---------------------------------------------------------------------------
# Response (PSF) kernel used for the empirical accuracy / speed comparison.
# Point at your own kernel here; it must be a pickled ndarray of shape
# (2n, 2n) or (n_freq, 2n, 2n). The one shipped with the repo has n = 512.
# ---------------------------------------------------------------------------
KERNEL_FN = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/kernels/pk_eso_992mhz-1468mhz_6f_2deg_3072.pkl"

# Kernel maps: which frequency to slice through and the central kernel size
# whose center/outer split boundary is marked.
SLICE_FREQ = 1  # second frequency (0-indexed)
CENTER_SIZE = 768

# Whether to produce the downsampling-error plots (Plots 3-5, 7-8).
ERROR_MAP = True

# Colour limit for the error map (Plot 3). Clipping well below the central-peak
# error focuses the colour scale on the smaller outer errors.
ERROR_VMAX = 5e-3


def relative_memory(n, k, z):
    """Two-step convolution memory relative to the single full convolution.

    Parameters
    ----------
    n : int
        Sky image resolution per dimension (kernel is (2n, 2n)).
    k : array_like
        Central (high-resolution) kernel crop size.
    z : int
        Down-sampling  factor of the low-resolution branch.

    Returns
    -------
    np.ndarray
        Ratio of padded FFT areas (split / full).
    """
    k = np.asarray(k, dtype=float)
    mem_full = (2 * n) ** 2
    mem_high = (n + k) ** 2
    mem_low = (2 * n / z) ** 2
    return (mem_high + mem_low) / mem_full


def load_kernel(fn):
    """Load a PSF kernel and return it as an (n_freq, 2n, 2n) float array."""
    with open(fn, "rb") as f:
        kernel = pickle.load(f)
    kernel = np.asarray(kernel)
    if kernel.ndim == 2:
        kernel = kernel[None]
    if kernel.ndim != 3 or kernel.shape[-1] != kernel.shape[-2]:
        raise ValueError(f"expected a square (n_freq, 2n, 2n) kernel, got {kernel.shape}")
    return kernel


# %%
# ---------------------------------------------------------------------------
# Load the PSF kernel and set up the shared geometry used by the plots
# below: frequency-slice index, pixel-offset coordinates, central-crop size.
# ---------------------------------------------------------------------------
kernel = load_kernel(KERNEL_FN)
n_freq, k2, _ = kernel.shape
N_KER = k2 // 2
print(f"\nloaded kernel {os.path.basename(KERNEL_FN)}: "
      f"shape {kernel.shape} (n = {N_KER})")

freq = min(SLICE_FREQ, n_freq - 1)
center = k2 // 2
coords = np.arange(k2) - center  # pixel offset from the kernel centre
half = CENTER_SIZE // 2
os.makedirs(ODIR, exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# Plot 1: the full 2-D PSF kernel at the second frequency on a symlog
# colour scale. A diverging map with symmetric limits shows the positive core
# and negative sidelobes; the crimson box marks the central (high-res) crop of
# CENTER_SIZE pixels.
# ---------------------------------------------------------------------------
img = np.asarray(kernel[freq])
vmax = float(np.abs(img).max())
extent = [coords[0], coords[-1], coords[0], coords[-1]]
norms = {
    "log": SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax),
}
for scale, norm in norms.items():
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=300)
    im = ax.imshow(img, origin="lower", extent=extent, cmap="RdBu_r", norm=norm)
    ax.set_aspect("equal")  # keep the image itself quadratic (colorbar excluded)
    ax.add_patch(Rectangle(
        (-half, -half), CENTER_SIZE, CENTER_SIZE,
        fill=False, edgecolor="black", ls="--", lw=1.2,
        label=f"central part ($k = {CENTER_SIZE}$)",
    ))
    # keep just the black frame: no axis labels or ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    # colorbar axis pinned to the bottom, matching the image width
    cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      label="PSF kernel value")
    if scale == "log":
        # thin the crowded symlog ticks: drop the outermost +/- pair (e.g.
        # +/-10^4) and the +/-10^-2 and +/-10^-4 decades
        drop_exp = {-2, -4}
        ticks = np.asarray(cb.get_ticks(), dtype=float)
        if ticks.size:
            tmax = np.abs(ticks).max()
            keep = [
                t for t in ticks
                if abs(t) < tmax
                and not (t != 0 and round(np.log10(abs(t))) in drop_exp)
            ]
            cb.set_ticks(keep)

    fig.tight_layout()
    out = os.path.join(ODIR, f"kernel_2d_{scale}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# %%
# ---------------------------------------------------------------------------
# Plot 2: the same 2-D kernel but zoomed to the central half of the kernel
# (n x n = N_KER x N_KER), keeping the k = CENTER_SIZE crop marker.
# ---------------------------------------------------------------------------
h4 = k2 // 4
sl_c = slice(k2 // 2 - h4, k2 // 2 + h4)  # central half -> (n, n)
img_z = img[sl_c, sl_c]
coords_z = coords[sl_c]
ext_z = [coords_z[0], coords_z[-1], coords_z[0], coords_z[-1]]

fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=300)
im = ax.imshow(img_z, origin="lower", extent=ext_z, cmap="RdBu_r",
               norm=SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax))
ax.set_aspect("equal")
ax.add_patch(Rectangle(
    (-half, -half), CENTER_SIZE, CENTER_SIZE,
    fill=False, edgecolor="black", ls="--", lw=1.2,
    label=f"central part ($k = {CENTER_SIZE}$)",
))
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="upper right", fontsize=8)
cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.05)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", label="PSF kernel value")
drop_exp = {-2, -4}
ticks = np.asarray(cb.get_ticks(), dtype=float)
if ticks.size:
    tmax = np.abs(ticks).max()
    cb.set_ticks([
        t for t in ticks
        if abs(t) < tmax and not (t != 0 and round(np.log10(abs(t))) in drop_exp)
    ])

fig.tight_layout()
out = os.path.join(ODIR, "kernel_2d_log_zoom.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("saved", out)


# %%
# ---------------------------------------------------------------------------
# Plot 3: downsampling-error map — the low-res branch represents the kernel by
# a block-averaged (downsampled-then-upsampled) copy. This shows that error at
# *every* location, (downsampled - high-res) kernel for z = ERROR_MAP_Z over
# the full 2n kernel, with the k = CENTER_SIZE box marking the region we keep
# at high resolution (where this error is avoided).
# ---------------------------------------------------------------------------
ERROR_MAP_Z = 3
if ERROR_MAP and k2 % ERROR_MAP_Z != 0:
    print(f"error-map plot skipped: z={ERROR_MAP_Z} does not divide 2n = {k2}")
elif ERROR_MAP:
    kimg = np.asarray(kernel[freq])  # full (2n, 2n) high-res kernel
    m = k2 // ERROR_MAP_Z
    kdown = kimg.reshape(m, ERROR_MAP_Z, m, ERROR_MAP_Z).mean(axis=(1, 3))
    kup = np.repeat(np.repeat(kdown, ERROR_MAP_Z, axis=0), ERROR_MAP_Z, axis=1)
    err = kup - kimg  # downsampled - high-res kernel, at every location

    vmax_e = ERROR_VMAX  # fixed clip to focus on the outer errors
    vmin_e = vmax_e * 1e-3
    err_norm = LogNorm(vmin=vmin_e, vmax=vmax_e)
    coords_k = np.arange(k2) - k2 // 2
    ext = [coords_k[0], coords_k[-1], coords_k[0], coords_k[-1]]

    aerr = np.abs(err)
    # max error outside the central (high-res) crop -- the worst error the
    # split scheme actually incurs (inside the crop it is reproduced exactly)
    inside_1d = (coords_k >= -half) & (coords_k < half)
    outside = ~(inside_1d[:, None] & inside_1d[None, :])
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=300)
    im = ax.imshow(aerr, origin="lower", extent=ext, cmap=REDS_HALF, norm=err_norm)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle(
        (-half, -half), CENTER_SIZE, CENTER_SIZE,
        fill=False, edgecolor="black", ls="--", lw=1.2,
        label=f"central part ($k = {CENTER_SIZE}$)",
    ))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      label=rf"PSF kernel error ($z = {ERROR_MAP_Z}$)")
    # keep only in-range ticks, then every second decade
    ticks = np.asarray(cb.get_ticks(), dtype=float)
    ticks = ticks[(ticks >= vmin_e) & (ticks <= vmax_e)]
    exps = [round(np.log10(t)) for t in ticks if t > 0]
    if exps:
        emax = max(exps)
        cb.set_ticks(ticks)

    fig.tight_layout()
    out = os.path.join(ODIR, "split_error_map.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# %%
# ---------------------------------------------------------------------------
# Plot 4: the same absolute error map zoomed to the central half of the kernel
# (n x n = N_KER x N_KER), keeping the k = CENTER_SIZE crop marker.
# ---------------------------------------------------------------------------
if ERROR_MAP and k2 % ERROR_MAP_Z == 0:
    h4 = k2 // 4
    sl_c = slice(k2 // 2 - h4, k2 // 2 + h4)  # central half -> (n, n)
    aerr_z = aerr[sl_c, sl_c]
    coords_z = coords_k[sl_c]
    ext_z = [coords_z[0], coords_z[-1], coords_z[0], coords_z[-1]]

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=300)
    im = ax.imshow(aerr_z, origin="lower", extent=ext_z, cmap=REDS_HALF, norm=err_norm)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle(
        (-half, -half), CENTER_SIZE, CENTER_SIZE,
        fill=False, edgecolor="black", ls="--", lw=1.2,
        label=f"central part ($k = {CENTER_SIZE}$)",
    ))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      label=rf"PSF kernel error ($z = {ERROR_MAP_Z}$)")
    ticks = np.asarray(cb.get_ticks(), dtype=float)
    ticks = ticks[(ticks >= vmin_e) & (ticks <= vmax_e)]
    exps = [round(np.log10(t)) for t in ticks if t > 0]
    if exps:
        emax = max(exps)
        cb.set_ticks(ticks)

    fig.tight_layout()
    out = os.path.join(ODIR, "split_error_map_zoom.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# %%
# ---------------------------------------------------------------------------
# Plot 5: combined two-panel 2-D summary (central-half zoom) -- (left) the
# PSF kernel value (Plot 2) and (right) the downsampling error (Plot 4),
# both with the k = CENTER_SIZE crop marker and a bottom colorbar.
# ---------------------------------------------------------------------------
if ERROR_MAP and k2 % ERROR_MAP_Z == 0:
    h4 = k2 // 4
    sl_c = slice(k2 // 2 - h4, k2 // 2 + h4)  # central half -> (n, n)
    coords_z = coords[sl_c]
    ext_z = [coords_z[0], coords_z[-1], coords_z[0], coords_z[-1]]

    # narrow figure + small wspace so the inter-column gap matches the
    # image-to-colorbar pad (~0.05 in); no tight_layout (it would reset wspace)
    fig, (ax_k, ax_e) = plt.subplots(
        1, 2, figsize=(7.0, 5.4), dpi=300, gridspec_kw={"wspace": 0.02})

    # left panel: PSF kernel value (symlog, diverging)
    im_k = ax_k.imshow(
        img[sl_c, sl_c], origin="lower", extent=ext_z, cmap="RdBu_r",
        norm=SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax),
    )
    ax_k.set_aspect("equal")
    ax_k.add_patch(Rectangle(
        (-half, -half), CENTER_SIZE, CENTER_SIZE,
        fill=False, edgecolor="black", ls="--", lw=1.2,
        label=f"central part ($k = {CENTER_SIZE}$)",
    ))
    ax_k.set_xticks([])
    ax_k.set_yticks([])
    ax_k.legend(loc="upper right", fontsize=8)
    cax_k = make_axes_locatable(ax_k).append_axes("bottom", size="5%", pad=0.05)
    cb_k = fig.colorbar(im_k, cax=cax_k, orientation="horizontal",
                        label="PSF kernel value")
    drop_exp = {-2, -4}
    ticks = np.asarray(cb_k.get_ticks(), dtype=float)
    if ticks.size:
        tmax = np.abs(ticks).max()
        cb_k.set_ticks([
            t for t in ticks
            if abs(t) < tmax and not (t != 0 and round(np.log10(abs(t))) in drop_exp)
        ])

    # right panel: downsampling error (log, positive)
    im_e = ax_e.imshow(aerr[sl_c, sl_c], origin="lower", extent=ext_z,
                       cmap=REDS_HALF, norm=err_norm)
    ax_e.set_aspect("equal")
    ax_e.add_patch(Rectangle(
        (-half, -half), CENTER_SIZE, CENTER_SIZE,
        fill=False, edgecolor="black", ls="--", lw=1.2,
    ))
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    # right legend states the zoom factor with no handle in front of the text
    blank = Rectangle((0, 0), 1, 1, fill=False, edgecolor="none")
    cax_e = make_axes_locatable(ax_e).append_axes("bottom", size="5%", pad=0.05)
    cb_e = fig.colorbar(im_e, cax=cax_e, orientation="horizontal",
                        label=rf"PSF kernel error ($z = {ERROR_MAP_Z}$)")
    ticks = np.asarray(cb_e.get_ticks(), dtype=float)
    cb_e.set_ticks(ticks[(ticks >= vmin_e) & (ticks <= vmax_e)])

    # shrink both colorbars' label + tick fonts by 1.5x
    cbfs = plt.rcParams["font.size"] / 1.5
    for cb in (cb_k, cb_e):
        cb.ax.tick_params(labelsize=cbfs)
        cb.ax.xaxis.label.set_size(cbfs)

    out = os.path.join(ODIR, "kernel_error_zoom.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# %%
# ---------------------------------------------------------------------------
# Plot 6: expected (analytic) relative memory vs central kernel size k, one
# curve per zoom factor -- the padded-FFT-area model from memory.py.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

for z in Z_FACTORS:
    color = COLORS[z]
    ax.plot(K_GRID, relative_memory(N, K_GRID, z), color=color, lw=2, label=f"z={z}")
    # mark the discrete kernel sizes requested for the paper
    ax.scatter(
        K_SIZES, relative_memory(N, np.array(K_SIZES), z),
        color=color, s=18, zorder=5, edgecolors="0.3", linewidths=0.5,
    )

ax.axhline(1.0, color="0.6", lw=0.8, ls="--", zorder=0)
ax.set_xlabel(r"central kernel size $k$ [pixels]")
ax.set_ylabel("relative FFT memory")
ax.set_xticks(K_SIZES[1::2])  # every second label, starting at 256
ax.set_xlim(K_SIZES[0], K_SIZES[-1])
ax.legend(title=f"$n = {N}$")
ax.grid(True, alpha=0.25)

fig.tight_layout()
out = os.path.join(ODIR, "memory.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("saved", out)

# %%
# Print the values at the discrete kernel sizes for the caption / table.
print(f"\nrelative memory (n = {N}):")
print("  k    " + "  ".join(f"z={z:>4d}" for z in Z_FACTORS))
for k in K_SIZES:
    row = "  ".join(f"{relative_memory(N, k, z):6.3f}" for z in Z_FACTORS)
    print(f"{k:>5d}  {row}")


# %%
# ---------------------------------------------------------------------------
# Plot 7: max downsampling error outside the central crop vs crop size k, one
# curve per zoom factor -- the worst error the split scheme incurs for a given
# (k, z). The downsampled kernel depends only on z, so it is built once per z
# and re-masked per k.
# ---------------------------------------------------------------------------
if ERROR_MAP:
    kimg = np.asarray(kernel[freq])
    coords_k = np.arange(k2) - k2 // 2

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    for z in Z_FACTORS:
        if k2 % z != 0:
            continue
        m = k2 // z
        kup = np.repeat(np.repeat(
            kimg.reshape(m, z, m, z).mean(axis=(1, 3)), z, axis=0), z, axis=1)
        aerr_z = np.abs(kup - kimg)
        max_out = []
        for k in K_SPLIT:
            hk = k // 2
            inside = (coords_k >= -hk) & (coords_k < hk)
            outside = ~(inside[:, None] & inside[None, :])
            max_out.append(float(aerr_z[outside].max()))
        ax.plot(K_SPLIT, max_out, color=COLORS[z], lw=2, marker="o", ms=5,
                markeredgecolor="0.3", markeredgewidth=0.5, label=f"z={z}")

    ax.set_yscale("log")
    ax.set_xlabel(r"central kernel size $k$ [pixels]")
    ax.set_ylabel(r"maximal kernel error")
    ax.set_xticks(K_SIZES[1::2])  # same labels as the memory plot
    ax.set_xlim(K_SIZES[0], K_SIZES[-1])
    ax.legend(title=f"$n = {N_KER}$")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out = os.path.join(ODIR, "split_error_outside.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# %%
# ---------------------------------------------------------------------------
# Plot 8: combined two-row summary of the split trade-off vs crop size k --
# (top) the analytic relative memory (Plot 6) and (bottom) the max downsampling
# error outside the crop (Plot 7), sharing the x-axis (labelled once, bottom).
# ---------------------------------------------------------------------------
if ERROR_MAP:
    fig, (ax_mem, ax_err) = plt.subplots(
        2, 1, figsize=(6.5, 7.2), dpi=300, sharex=True)

    # top panel: analytic relative memory
    for z in Z_FACTORS:
        ax_mem.plot(K_GRID, relative_memory(N, K_GRID, z),
                    color=COLORS[z], lw=2, label=f"z={z}")
        ax_mem.scatter(
            K_SIZES, relative_memory(N, np.array(K_SIZES), z),
            color=COLORS[z], s=18, zorder=5, edgecolors="0.3", linewidths=0.5,
        )
    ax_mem.axhline(1.0, color="0.6", lw=0.8, ls="--", zorder=0)
    ax_mem.set_ylabel("relative FFT memory")
    ax_mem.grid(True, alpha=0.25)  # legend only on the bottom panel

    # bottom panel: max downsampling error outside the crop
    kimg = np.asarray(kernel[freq])
    coords_k = np.arange(k2) - k2 // 2
    for z in Z_FACTORS:
        if k2 % z != 0:
            continue
        m = k2 // z
        kup = np.repeat(np.repeat(
            kimg.reshape(m, z, m, z).mean(axis=(1, 3)), z, axis=0), z, axis=1)
        aerr_z = np.abs(kup - kimg)
        max_out = []
        for k in K_SPLIT:
            hk = k // 2
            inside = (coords_k >= -hk) & (coords_k < hk)
            outside = ~(inside[:, None] & inside[None, :])
            max_out.append(float(aerr_z[outside].max()))
        ax_err.plot(K_SPLIT, max_out, color=COLORS[z], lw=2, marker="o", ms=5,
                    markeredgecolor="0.3", markeredgewidth=0.5, label=f"z={z}")
    ax_err.set_yscale("log")
    ax_err.set_xlabel(r"central kernel size $k$ [pixels]")
    ax_err.set_ylabel(r"maximal kernel error")
    ax_err.set_xticks(K_SIZES[1::2])
    ax_err.set_xlim(K_SIZES[0], K_SIZES[-1])
    ax_err.legend(title=f"$n = {N_KER}$")
    ax_err.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.subplots_adjust(hspace=fig.subplotpars.hspace / 2)  # halve the row gap
    out = os.path.join(ODIR, "memory_maxerror.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

# %%

# %%
# ---------------------------------------------------------------------------
# Memory footprint of the two-step (split) PSF convolution relative to the
# single full-resolution convolution, as used by the fast-resolve response.
#
# The fast-resolve response convolves a sky image of shape (n, n) with a PSF
# kernel of shape (2n, 2n). The single convolution therefore pads the sky to
# (2n, 2n) and FFTs at that shape, so its memory footprint scales with the
# padded area (2n)^2 (see `PSFConvolve` in
# `aim_resolve/fast_resolve/convolve.py`).
#
# The two-step convolution (see `PSFSplitConvolve` / `build_split_kernel`)
# replaces this single large transform by two smaller ones:
#   * high-res: sky padded to (n + k, n + k) and FFT'd at that shape, where k
#     is the size of the central kernel crop;
#   * low-res:  sky and outer kernel downsampled by a factor z, padded to
#     (2n / z, 2n / z) and FFT'd at that shape.
# Since the FFT arrays dominate the memory, the relative footprint is the ratio
# of padded areas:
#
#     mem_split / mem_full = [ (n + k)^2 + (2n / z)^2 ] / (2n)^2
#
# This script reproduces `plots/memory.png` for the full-sky resolution
# n = 3072, the zoom factors z in {2, 3, 4} and central kernel sizes k.
# ---------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np

# %%
# Full-sky image resolution per dimension (sky shape is (n, n), kernel (2n, 2n)).
N = 3072

# Down-sampling (zoom) factors for the low-resolution branch.
Z_FACTORS = (2, 3, 4)

# Discrete central-kernel sizes k: 128 to 1536 in steps of 128.
K_SIZES = tuple(range(128, 1536 + 1, 128))

# Continuous range of k for the curves.
K_GRID = np.linspace(K_SIZES[0], K_SIZES[-1], 512)

# Matplotlib colors per zoom factor (matching the original plot).
COLORS = {2: "tab:blue", 3: "lightgray", 4: "crimson"}

ODIR = os.path.join(os.path.dirname(__file__), "plots")


def relative_memory(n, k, z):
    """Two-step convolution memory relative to the single full convolution.

    Parameters
    ----------
    n : int
        Sky image resolution per dimension (kernel is (2n, 2n)).
    k : array_like
        Central (high-resolution) kernel crop size.
    z : int
        Down-sampling factor of the low-resolution branch.

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


# %%
# ---------------------------------------------------------------------------
# Plot: relative memory vs central kernel size k, one curve per zoom factor.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)

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
ax.set_ylabel("relative memory (two-step / single)")
ax.set_xticks(K_SIZES[1::2])  # every second label, starting at 256
ax.set_xlim(K_SIZES[0], K_SIZES[-1])
ax.legend(title=f"$n = {N}$")
ax.grid(True, alpha=0.25)

fig.tight_layout()
os.makedirs(ODIR, exist_ok=True)
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

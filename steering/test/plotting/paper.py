import numpy as np

from aim_resolve import plot_arrays

# %%
a = np.zeros((512, 512))

labels1 = [
    "Ground truth data",
    "Noisy data",
    "SCM: reconstructed mean",
    "SCM: relative standard deviation",
    "MCM: reconstructed mean",
    "MCM: relative standard deviation",
]

plot_arrays(
    [a, a, a, a, a, a],
    label=labels1,
    figsize=(5.2, 5),
    cols=2,
    dpi=50,
    ticks=0,
    cbar=True,
    vmin=0,
    vmax=1,
)

# %%
a = np.zeros((512, 512))


def callback(fig, axes):
    fig.text(0.025, 0.955, "True components", fontsize=15, c="white")
    fig.text(0.52, 0.955, "Iteration 1", fontsize=15, c="white")
    fig.text(0.52, 0.465, "Iteration 2", fontsize=15, c="white")
    fig.text(0.025, 0.465, "Iteration 3", fontsize=15, c="white")


plot_arrays(
    [a, a, a, a],
    figsize=(5, 5),
    cols=2,
    dpi=50,
    ticks=0,
    cbar=False,
    callback=callback,
    vmin=0,
    vmax=1,
)

# %%
a = np.zeros((512, 512))


def callback(fig, axes):
    fig.text(0.02, 0.94, "True components", fontsize=20, c="white")
    fig.text(0.35, 0.94, "Iteration 1", fontsize=20, c="white")
    fig.text(0.68, 0.94, "Iteration 2", fontsize=20, c="white")
    fig.text(0.02, 0.46, "Iteration 3", fontsize=20, c="white")
    fig.text(0.35, 0.46, "Iteration 4", fontsize=20, c="white")
    fig.text(0.68, 0.46, "Iteration 5", fontsize=20, c="white")


plot_arrays(
    [a, a, a, a, a, a],
    figsize=(5, 5.1),
    cols=3,
    dpi=50,
    ticks=0,
    cbar=False,
    callback=callback,
    vmin=0,
    vmax=1,
)

# %%
a = np.zeros((512, 512))

labels1 = [
    "SCM: reconstructed mean",
    "SCM: relative standard deviation",
    "MCM: reconstructed mean",
    "MCM: relative standard deviation",
]

plot_arrays(
    [a, a, a, a],
    label=labels1,
    figsize=(5.3, 5),
    cols=2,
    dpi=50,
    ticks=0,
    cbar=True,
    vmin=0,
    vmax=1,
)

# %%
a = np.zeros((230, 180))
b = np.zeros((290, 142))

labels1 = [
    "MCM: ESO137-006 galaxy mean",
    "MCM: ESO137-006 galaxy relative standard deviation",
    "MCM: ESO137-007 galaxy mean",
    "MCM: ESO137-007 galaxy relative standard deviation",
]

plot_arrays(
    [a, b, a, b],
    label=labels1,
    figsize=(7, 5),
    cols=1,
    dpi=100,
    ticks=0,
    cbar=True,
    vmin=0,
    vmax=1,
)

# %%

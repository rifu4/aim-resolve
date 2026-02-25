"""Multi-array plotting utilities for images and power spectra."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .image import plot_image
from .power import plot_power
from .util import plot_figure, to_shape


def plot_arrays(
    array,
    grid=None,
    label=None,
    name=None,
    odir=None,
    rows=None,
    cols=None,
    cmap="inferno",
    norm="linear",
    vmin=None,
    vmax=None,
    cbar=True,
    cbar_kwargs=None,
    ticks=5,
    origin="lower",
    marker=None,
    contour=None,
    square=False,
    transpose=False,
    plot_grid=True,
    plot_label=True,
    figsize=(5, 5),
    dpi=100,
    callback=None,
    grid_kwargs=None,
    **kwargs,
):
    """
    Plot arrays or lists containing multiple 2D images or power spectra.

    Parameters
    ----------
    array : np.ndarray or Iterable of np.ndarrays
        The array to plot. Plots an image for 2D arrays and a power spectrum for 1D arrays.
        If an Iterbale of arrays is provided or array.ndim > 2, multiple subplots will be created.
    grid : str, optional
        The grid of the (sub)arrays. Default is None.
    label : str, optional
        The label of the (sub)plot. Default is None.
    name : str, optional
        The name of the (sub)plot. Default is None.
    odir : str, optional
        The output directory to save the plot. Default is None.
    rows : int, optional
        The number of rows in the plot. Default is None.
    cols : int, optional
        The number of columns in the plot. Default is None.
    cmap : str, optional
        The colormap to use. Default is 'inferno'.
    norm : str, optional
        The normalization to use. Default is 'linear'.
    vmin : float, optional
        The minimum value to use for the colormap. Default is None.
    vmax : float, optional
        The maximum value to use for the colormap. Default is None.
    cbar : bool, optional
        Whether to show the colorbar. Default is '{}'.
    ticks : int, optional
        The number of ticks to use. Default is 5. If set to 0, no ticks will be shown.
    origin : str, optional
        The origin parameter for imshow. Default is 'lower'.
    marker : dict or dict containing subdicts, optional
        The markers to plot. For one marker it should look like {'x': [...], 'y': [...], ...}.
        For multiple markers {'m0': {...}, 'm1': {...}, ...}. Default is {}.
    contour : dict, optional
        The contours to plot. Keywords are passed to plt.contour. Default is {}.
    square : bool, optional
        Whether to fillup non-square images with zeros. Default is False.
    transpose : bool, optional
        Whether to transpose the rows, columns of a multi-plot. Default is False.
    plot_grid : bool, optional
        Whether to plot the grid of the array. Default is True.
    plot_label : bool, optional
        Whether to plot the label of the array. Default is True.
    figsize : tuple, optional
        The size of the figure. Default is (5, 5).
    dpi : int, optional
        The dpi of the figure. Default is 300.
    callback : callable, optional
        A callback function. Can be used to customize the plots (e.g. by adding text).
        -> The function should take two arguments: figure and axes.
    grid_kwargs : dict, optional
        The keyword arguments to pass to the GridSpec. Default is {}.
    kwargs : optional
        Additional keyword arguments to pass to the plotting functions.
    """
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if marker is None:
        marker = {}
    if contour is None:
        contour = {}
    if grid_kwargs is None:
        grid_kwargs = {}

    arrays, nums = to_shape(array, None, rows, cols, 0.0, transpose, return_nums=True)
    shape = arrays.shape[:2]
    rows, cols = shape

    shape_T = shape[::-1] if transpose else shape
    grids = to_shape(grid, shape_T, default=None, transpose=transpose)
    labels = to_shape(label, shape_T, default=None, transpose=transpose)
    cmaps = to_shape(cmap, shape_T, default=-1, transpose=transpose)
    vmins = to_shape(vmin, shape_T, default=-1, transpose=transpose)
    vmaxs = to_shape(vmax, shape_T, default=-1, transpose=transpose)
    norms = to_shape(norm, shape_T, default=-1, transpose=transpose)
    cbars = to_shape(cbar, shape_T, default=-1, transpose=transpose)
    cbar_kwargs = to_shape(cbar_kwargs, shape_T, default={}, transpose=transpose)
    markers = to_shape(marker, shape_T, default={}, transpose=transpose)
    contours = to_shape(contour, shape_T, default={}, transpose=transpose)

    figsize = to_shape(figsize, (2,), dtype="float64") * np.array(shape[::-1])
    figure = plt.figure(figsize=figsize, dpi=dpi)
    axes = []
    grid = GridSpec(rows, cols, figure=figure, **grid_kwargs)

    for i, (x, y) in enumerate(product(range(rows), range(cols))):
        if i >= nums:
            continue
        axes.append(figure.add_subplot(grid[x, y]))

        array = arrays[x, y]

        if array.ndim == 2:
            plot_image(
                array=array,
                axes=axes,
                grid=grids[x, y],
                label=labels[x, y],
                cmap=cmaps[x, y],
                norm=norms[x, y],
                vmin=vmins[x, y],
                vmax=vmaxs[x, y],
                cbar=cbars[x, y],
                cbar_kwargs=cbar_kwargs[x, y],
                ticks=ticks,
                origin=origin,
                marker=markers[x, y],
                contour=contours[x, y],
                square=square,
                plot_grid=plot_grid,
                plot_label=plot_label,
                **kwargs,
            )
        elif array.ndim == 1:
            plot_power(
                array=array,
                axes=axes,
                label=labels[x, y],
                plot_label=plot_label,
                **kwargs,
            )
        else:
            continue
            # raise ValueError('`array` has to be 1D or 2D')

    if callable(callback):
        callback(figure, axes)

    grid.tight_layout(figure)

    plot_figure(figure, odir, name)

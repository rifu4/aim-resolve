"""Plotting utility functions for axes, colorbars, ticks, and layout."""

import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from ..model.grid import SignalGrid


def plot_figure(
    figure,
    odir=None,
    name=None,
):
    """
    Plot a figure using plt.show() or save it to a file.

    Parameters
    ----------
    figure : plt.Figure
        The matplotlib figure to display or save.
    odir : str, optional
        The output directory to save the plot. Default is None.
    name : str, optional
        The filename for the saved plot. Default is None.
    """
    if not isinstance(figure, plt.Figure):
        raise TypeError("`fig` has to be of Type `matplotlib.figure.Figure`")

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name))
    else:
        plt.show()
    plt.close()


def set_cbar(
    axes,
    image,
    cbar=True,
    loc="right",
    size="2.5%",
    pad="2.5%",
    label=None,
    labelpad=3,
    labelloc="center",
):
    """
    Set the colorbar for a given axes.

    Parameters
    ----------
    axes : plt.Axes
        The axes to attach the colorbar to.
    image : plt.cm.ScalarMappable
        The image or mappable for the colorbar.
    cbar : bool, optional
        Whether to show the colorbar. Default is True.
    loc : str, optional
        The location of the colorbar. Default is 'right'.
    size : str, optional
        The size of the colorbar axes. Default is '2.5%'.
    pad : str, optional
        The padding between the image and colorbar. Default is '2.5%'.
    label : str, optional
        The label for the colorbar. Default is None.
    labelpad : int, optional
        The padding for the colorbar label. Default is 3.
    labelloc : str, optional
        The location of the colorbar label. Default is 'center'.
    """
    div = make_axes_locatable(axes)
    cax_bottom = div.append_axes("bottom", size=size, pad=pad)
    cax_right = div.append_axes("right", size=size, pad=pad)
    cax_left = div.append_axes("left", size=size, pad=pad)

    if cbar and loc == "bottom":
        cbar = plt.colorbar(image, cax=cax_bottom, orientation="horizontal")
        rot = 0
    else:
        cax_bottom.set_visible(False)

    if cbar and loc == "right":
        cbar = plt.colorbar(image, cax=cax_right)
        rot = 90
    else:
        cax_right.set_visible(False)

    if cbar and loc == "left":
        cbar = plt.colorbar(image, cax=cax_left)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        rot = 90
    else:
        cax_left.set_visible(False)

    if cbar and label:
        cbar.set_label(label, rotation=rot, labelpad=labelpad, loc=labelloc)


def set_ticks(
    axes,
    grid=None,
    ticks=5,
    plot_grid=True,
):
    """
    Set the ticks of the axes.

    Parameters
    ----------
    axes : plt.Axes
        The axes to set ticks for.
    grid : SignalGrid, optional
        The grid used to compute tick positions and labels. Default is None.
    ticks : int, optional
        The number of ticks to use. Default is 5. If set to 0, the axes
        will be turned off.
    plot_grid : bool, optional
        Whether to use the grid for tick labels. Default is True.
    """
    if ticks > 0:
        if plot_grid and isinstance(grid, SignalGrid):
            axes.set_xticks(
                ticks=np.linspace(0, grid.shape[0], ticks) - 0.5,
                labels=np.linspace(grid.lims[0, 0], grid.lims[0, 1], ticks).round(2),
            )
            axes.set_yticks(
                ticks=np.linspace(0, grid.shape[1], ticks) - 0.5,
                labels=np.linspace(grid.lims[1, 0], grid.lims[1, 1], ticks).round(2),
            )
    else:
        axes.axis("off")


def rows_and_cols(
    nums,
    rows=None,
    cols=None,
):
    """
    Calculate the number of rows and columns for a grid of subplots.

    Parameters
    ----------
    nums : int
        The total number of subplots.
    rows : int, optional
        The desired number of rows. Default is None.
    cols : int, optional
        The desired number of columns. Default is None.

    Returns
    -------
    rows : int
        The computed number of rows.
    cols : int
        The computed number of columns.
    """
    match (rows, cols):
        case (None, None):
            rows = int(np.ceil(np.sqrt(nums)))
            cols = int(np.ceil(nums / rows))
        case (_, None):
            rows = min(rows, nums)
            cols = int(np.ceil(nums / rows))
        case _:
            cols = min(cols, nums)
            rows = int(np.ceil(nums / cols))
    return rows, cols


def to_shape(
    input,
    shape=None,
    rows=None,
    cols=None,
    default=-1,
    transpose=False,
    dtype=None,
    return_nums=False,
):
    """
    Convert an input to a specific shape or number of rows and columns.

    Parameters
    ----------
    input : array_like
        The input data to reshape.
    shape : tuple, optional
        The target shape. Default is None.
    rows : int, optional
        The desired number of rows. Default is None.
    cols : int, optional
        The desired number of columns. Default is None.
    default : scalar, optional
        The default fill value for padding. Default is -1.
    transpose : bool, optional
        Whether to transpose the result. Default is False.
    dtype : str or np.dtype, optional
        The desired data type. Default is None.
    return_nums : bool, optional
        Whether to also return the original number of elements.
        Default is False.

    Returns
    -------
    array : np.ndarray
        The reshaped array.
    nums : int
        The original number of elements (only if ``return_nums`` is True).
    """
    array = np.array(input, dtype=object)

    match array.ndim:
        case 0:
            array = array.reshape((1,))
        case 1:
            array = array.reshape((-1,))
        case 2:
            array = array.reshape((-1,) + array.shape)
        case _:
            array = array.reshape((-1,) + array.shape[-2:])

    nums = array.shape[0]
    if shape is None:
        shape = rows_and_cols(nums, rows, cols)
    size = int(np.prod(shape))

    try:
        array = np.broadcast_to(array, shape + array.shape[1:])

    except:
        if nums < size:
            default = array[-1] if default == -1 else default
            if array.ndim == 1:
                for i in range(size - nums):
                    array = np.append(array, default)
            else:
                array = np.append(
                    array, np.zeros((size - nums,) + array.shape[1:]), axis=0
                )
        else:
            array = array[:size]
        array = array.reshape(shape + array.shape[1:])

    if transpose:
        array = array.transpose(1, 0, *range(2, array.ndim))

    if dtype:
        array = array.astype(dtype)

    if return_nums:
        return array, nums
    else:
        return array

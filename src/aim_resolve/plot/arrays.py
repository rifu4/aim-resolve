import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from .image import plot_image
from .power import plot_power
from .util import plot_figure, to_shape



def plot_arrays(
        array,
        space = None,
        label = None,
        name = None,
        odir = None,
        rows = None,
        cols = None,
        cmap = 'inferno',
        norm = 'linear',
        vmin = None,
        vmax = None,
        cbar = True,
        ticks = 5,
        marker = (),
        square = False,
        transpose = False,
        plot_space = True,
        plot_label = True,
        figsize = (5, 5),
        dpi = 300,
        **kwargs,
):
    '''
    Plot arrays or lists containing multiple 2D images or power spectra.
    
    Parameters
    ----------
    array : np.ndarray or Iterable of np.ndarrays
        The array to plot. Plots an image for 2D arrays and a power spectrum for 1D arrays.
        If an Iterbale of arrays is provided or array.ndim > 2, multiple subplots will be created.
    space : str, optional
        The space of the (sub)arrays. Default is None.
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
        Whether to show the colorbar. Default is True.
    ticks : int, optional
        The number of ticks to use. Default is 5. If set to 0, no ticks will be shown.
    marker : tuple, optional
        Plot markers at specific locations in an image. Default is ().
    square : bool, optional
        Whether to fillup non-square images with zeros. Default is False.
    transpose : bool, optional
        Whether to transpose the rows, columns of a multi-plot. Default is False.
    plot_space : bool, optional
        Whether to plot the space of the array. Default is True.
    plot_label : bool, optional
        Whether to plot the label of the array. Default is True.
    figsize : tuple, optional
        The size of the figure. Default is (5, 5).
    dpi : int, optional
        The dpi of the figure. Default is 300.
    kwargs : optional
        Additional keyword arguments to pass to the plotting functions.
    '''
    arrays, nums = to_shape(array, None, rows, cols, 0., transpose, return_nums=True)
    shape = arrays.shape[:2]
    rows, cols = shape

    shape_T = shape[::-1] if transpose else shape
    spaces = to_shape(space, shape_T, default=None, transpose=transpose)
    labels = to_shape(label, shape_T, default=None, transpose=transpose)
    cmaps = to_shape(cmap, shape_T, default=-1, transpose=transpose)
    vmins = to_shape(vmin, shape_T, default=-1, transpose=transpose)
    vmaxs = to_shape(vmax, shape_T, default=-1, transpose=transpose)
    norms = to_shape(norm, shape_T, default=-1, transpose=transpose)

    figsize = to_shape(figsize, (2,), dtype='float64') * np.array(shape[::-1])
    figure = plt.figure(figsize=figsize, dpi=dpi)
    axes = []
    for i,(x,y) in enumerate(product(range(rows), range(cols))):
        if i >= nums:
            continue
        axes.append(figure.add_subplot(rows, cols, i+1))

        array = arrays[x, y]

        if array.ndim == 2:
            plot_image(
                array = array,
                axes = axes,
                space = spaces[x, y],
                label = labels[x, y],
                cmap = cmaps[x, y],
                norm = norms[x, y],
                vmin = vmins[x, y],
                vmax = vmaxs[x, y],
                cbar = cbar,
                ticks = ticks,
                marker = marker,
                square = square,
                plot_space = plot_space,
                plot_label = plot_label,
                **kwargs,
            )
        elif array.ndim == 1:
            plot_power(
                array = array,
                axes = axes,
                label = labels[x,y],
                plot_label = plot_label,
                **kwargs,
            )
        else:
            raise ValueError('`array` has to be 1D or 2D')

    plot_figure(figure, odir, name)

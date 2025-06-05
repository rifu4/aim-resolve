import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from nifty8.re import Vector

from .image import plot_image
from .power import plot_power
from .util import plot_figure, to_shape
from ..model.components import ComponentModel
from ..model.points import PointModel
from ..model.signal import SignalModel
from ..model.tiles import TileModel
from ..optimize.samples import MySamples



def plot_models(
        model,
        samples,
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
        dpi = 100,
        **kwargs,
):
    '''
    Plot the samples mean or of one or multiple models.
    
    Parameters
    ----------
    model : ComponentModel, PointModel, SignalModel, TileModel or Iterable of those
        The model to plot. Plots an image for 2D models and their 1D power spectrum.
        If an Iterbale of models is provided, multiple subplots will be created.
    samples : MySamples, Vector or dict
        The input samples or position vector for the models.
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
    models, nums = to_shape(model, None, rows, cols, 0., transpose, return_nums=True)
    shape = models.shape[:2]
    rows, cols = shape

    shape_T = shape[::-1] if transpose else shape
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
        
        model = models[x,y]
        if not isinstance(model, (ComponentModel, PointModel, SignalModel, TileModel)):
            raise TypeError('`model` has to be of type `ComponentModel`, `PointModel`, `SignalModel` or `TileModel`')
        if isinstance(samples, MySamples):
            array = samples.mean(model)
        elif isinstance(samples, (Vector, dict)):
            array = model(samples)
        else:
            raise TypeError('`samples` has to be of type `MySamples`, `Vector` or `dict`')

        if array.ndim == 2:
            plot_image(
                array = array,
                axes = axes,
                space = models[x, y].space,
                label = models[x, y].prefix,
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
                label = models[x, y].prefix,
                plot_label = plot_label,
            )
        else:
            raise ValueError('`array` has to be 1D or 2D')

    plot_figure(figure, odir, name)

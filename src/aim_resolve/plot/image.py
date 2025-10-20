import matplotlib.pyplot as plt
import numpy as np

from .util import plot_figure, set_cbar, set_ticks
from ..model.grid import SignalGrid
from ..model.map import map_signal 



def plot_image(
        array,
        axes = None,
        grid = None, 
        label = None, 
        name = None,
        odir = None,
        cmap = 'inferno',
        norm = 'linear',
        vmin = None,
        vmax = None,
        cbar = True,
        cbar_kwargs = {},
        ticks = 5,
        origin = 'lower',
        marker = {},
        contour = {},
        square = False,
        plot_grid = True,
        plot_label = True,
        **kwargs,
):
    '''
    Plot a single 2D image using plt.imshow.
    
    Parameters
    ----------
    array : np.ndarray
        The array to plot.
    axes : list of plt.Axes, optional
        The axes to plot on. If not provided, a new figure will be created.
    grid : str, optional
        The grid of the array. Default is None.
    label : str, optional
        The label of the plot. Default is None.
    name : str, optional
        The name of the plot. Default is None.
    odir : str, optional
        The output directory to save the plot. Default is None.
    cmap : str, optional
        The colormap to use. Default is 'inferno'.
    norm : str, optional
        The normalization to use. Default is 'linear'.
    vmin : float, optional
        The minimum value to use for the colormap. Default is None.
    vmax : float, optional
        The maximum value to use for the colormap. Default is None.
    cbar : bool, optional
        Whether to show a colorbar. Default is True.
    cbar_kwargs : dict, optional
        The keyword arguments to pass to the colorbar. Default is {}.
    ticks : int, optional
        The number of ticks to use. Default is 5. If set to 0, no ticks will be shown.
    origin : str, optional
        The origin parameter for imshow. Default is 'lower'.
    marker : dict or dict containing subdicts, optional
        The markers to plot. For one marker it should look like {'x': [...], 'y': [...], ...}. 
        For multiple markers {'m0': {...}, 'm1': {...}, ...}. Default is {}.
    contour : dict, optional:
        The contours to plot. Keywords are passed to plt.contour. Default is {}.
    square : bool, optional
        Whether to plot the image in a square format. Default is False.
    plot_grid : bool, optional
        Whether to plot the grid of the array. Default is True.
    plot_label : bool, optional
        Whether to plot the label of the array. Default is True.
    kwargs : additional keyword arguments
        Additional keyword arguments to pass to plt.imshow.
    '''
    plot_now = False
    if axes is None:
        figure = plt.figure(figsize=(5,5))
        axes = []
        axes.append(figure.add_subplot(1, 1, 1))
        plot_now = True

    array = np.array(array, dtype='float64')

    if square:
        spc_old = SignalGrid.build(shape=array.shape, fov=array.shape)
        spc_new = SignalGrid.build(shape=spc_old.shp.max(), fov=spc_old.fov.max())
        array = map_signal(array, spc_old, spc_new)
        #TODO: fix grid for squared images. Set to None for now
        grid = None

    if norm == 'log':
        amin = array[array > 0].min() if np.any(array > 0) else 1
        array[array <= 0] = amin

    img = plt.imshow(
        X = array.T, 
        cmap = cmap, 
        norm = norm, 
        vmin = vmin, 
        vmax = vmax, 
        origin = origin,
        **kwargs,
    )

    if contour:
        contour_array = contour.pop('array', array)
        plt.contour(contour_array.T, origin='lower', **contour)

    set_cbar(axes[-1], img, cbar, **cbar_kwargs)    

    if plot_label and label:
        axes[-1].set_title(label)
    
    set_ticks(axes[-1], grid, ticks, plot_grid)
    
    marker = {'m0': marker} if all(k in marker for k in ['x', 'y']) else marker
    for mrk in marker.values():
        if isinstance(mrk, dict) and all(k in mrk for k in ['x', 'y']):
            axes[-1].scatter(**mrk)
        else:
            raise TypeError('`marker` has to be a dictionary with keys `x`, `y`.')

    if plot_now:
        plot_figure(figure, odir, name)

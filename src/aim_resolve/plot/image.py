import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from .util import plot_figure, set_ticks
from ..model.map import map_signal 
from ..model.space import SignalSpace



def plot_image(
        array,
        axes = None,
        space = None, 
        label = None, 
        name = None,
        odir = None,
        cmap = 'inferno',
        norm = 'linear',
        vmin = None,
        vmax = None,
        cbar = True,
        ticks = 5,
        marker = (),
        square = False,
        plot_space = True,
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
    space : str, optional
        The space of the array. Default is None.
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
    ticks : int, optional
        The number of ticks to use. Default is 5. If set to 0, no ticks will be shown.
    marker : tuple of dict, optional
        The markers to plot. Default is ().
    square : bool, optional
        Whether to plot the image in a square format. Default is False.
    plot_space : bool, optional
        Whether to plot the space of the array. Default is True.
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
        spc_old = SignalSpace.build(shape=array.shape, fov=array.shape)
        spc_new = SignalSpace.build(shape=spc_old.shp.max(), fov=spc_old.fov.max())
        array = map_signal(array, spc_old, spc_new)
        #TODO: fix space for squared images. Set to None for now
        space = None

    if norm == 'log':
        array[array<=0] = 1.
    
    img = plt.imshow(
        X = array.T, 
        cmap = cmap, 
        norm = norm, 
        vmin = vmin, 
        vmax = vmax, 
        origin = 'lower',
        **kwargs,
    )

    if cbar:
        div = make_axes_locatable(axes[-1])
        cax = div.append_axes('right', size='3%', pad='2%')
        plt.colorbar(img, cax)

    if plot_label and label:
        axes[-1].set_title(label)
    
    set_ticks(axes[-1], space, ticks, plot_space)
    
    marker = (marker, ) if not isinstance(marker, tuple) else marker
    for mrk in marker:
        if not isinstance(mrk, dict):
            raise TypeError('`marker` has to be a dictionary')
        axes[-1].scatter(**mrk)

    if plot_now:
        plot_figure(figure, odir, name)

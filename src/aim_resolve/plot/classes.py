import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colormaps

from .util import plot_figure, set_ticks



def plot_classes(
        points_map = None,
        object_maps = None,
        space = None,
        label = None,
        name = None,
        odir = None,
        cmap = 'inferno',
        ticks = 5,
        plot_space = True,
        plot_label = True,
        figsize = (5, 5),
        dpi = 100,
        **kwargs,
):
    '''
    Plot the classes of a segmentation maps.

    Parameters
    ----------
    points_map : np.ndarray, optional
        The points map to plot. If not provided, a map of zersos will be created.
    object_maps : np.ndarray, optional
        The object maps to plot. If not provided, a map of zeros will be created.
    space : str, optional
        The space of the maps. Default is None.
    label : str, optional
        The label of the plot. Default is None.
    name : str, optional
        The name of the plot. Default is None.
    odir : str, optional
        The output directory to save the plot. Default is None.
    cmap : str, optional
        The colormap to use. Default is 'inferno'.
    ticks : int, optional
        The number of ticks to use. Default is 5. If set to 0, no ticks will be shown.
    plot_space : bool, optional
        Whether to plot the space of the array. Default is True.
    plot_label : bool, optional
        Whether to plot the label of the array. Default is True.
    figsize : tuple, optional
        The size of the figure. Default is (5, 5).
    dpi : int, optional
        The dpi of the figure. Default is 100.
    **kwargs : dict, optional
        Additional keyword arguments for `plt.imshow` plotting. Not used in this function.
    '''
    match (points_map, object_maps):
        case (np.ndarray(), np.ndarray()):
            if points_map.shape[-2:] != object_maps.shape[-2:]:
                raise ValueError('`points_map` and `object_maps` must have the same shape.')
        case (np.ndarray(), _):
            shape = points_map.shape[-2:]
            object_maps = np.zeros((1,) + shape)
        case (_, np.ndarray()):
            shape = object_maps.shape[-2:]
            points_map = np.zeros(shape)
        case (_, _):
            raise ValueError('either `points_map` or `object_maps` must be provided.')

    points_map = points_map.astype('int64')
    object_maps = object_maps.astype('int64')

    figure = plt.figure(figsize=figsize, dpi=dpi)
    axes = []
    axes.append(figure.add_subplot(1, 1, 1))
    axes[-1].imshow(points_map.T, cmap, origin='lower')

    if np.any(object_maps > 0):
        all_map = np.concatenate([np.zeros(points_map[None, :].shape, dtype=int), object_maps], axis=0)
        one_map = np.argmax(all_map, axis=0)

        colours = colormaps.get_cmap('tab20')
        colours = colours(np.linspace(0, 1, max(colours.N, object_maps.shape[0])))
        colours = np.concatenate([np.zeros((1, 4)), colours])

        class_maps = colours[one_map.flatten()]
        class_maps = class_maps.reshape(one_map.shape + (4,))

        axes[-1].imshow(
            X = class_maps.transpose(1,0,2),
            cmap = cmap,
            origin='lower'
        )

    if plot_label and label:
        axes[-1].set_title(label)

    set_ticks(axes[-1], space, ticks, plot_space)

    plot_figure(figure, odir, name)

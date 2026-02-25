"""Extended plotting functions for model analysis and diagnostics."""

import numpy as np

from .arrays import plot_arrays
from ..img_data.data import ImageData
from ..model.map import map_signal
from ..model.util import check_type
from ..optimize.samples import MySamples



def plot_mean_and_std(
        model,
        samples,
        mode = 'mean_and_std',
        freq = False,
        **kwargs,
):
    """
    Plot the mean and standard deviation of samples for a given model.

    Parameters
    ----------
    model : object or list of objects
        The model or list of models to plot. Each model must have a
        ``prefix`` and ``grid`` attribute.
    samples : MySamples
        The samples instance containing posterior samples.
    mode : str, optional
        The plotting mode. Can contain 'mean' and/or 'std'.
        Default is 'mean_and_std'.
    freq : bool, optional
        Whether to plot all frequency slices. If False, only the
        central slice is plotted. Default is False.
    **kwargs
        Additional keyword arguments passed to ``plot_arrays``.
    """
    check_type(samples, MySamples)

    models = [model, ] if not isinstance(model, list) else model

    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)

    arrays, grids, labels, vmins, vmaxs = [], [], [], [], []
    for md in models:
        pf, it = md.prefix.split('.')[0], md.prefix.split('.')[1]
        mean, std = samples.mean_and_std(md)

        if mean.ndim == 2:
            mean = mean[None]
            std = std[None]
        if not freq:
            mean = mean[mean.shape[0]//2:mean.shape[0]//2+1]
            std = std[std.shape[0]//2:std.shape[0]//2+1]

        if 'mean' in mode:
            for i in range(mean.shape[0]):
                arrays += [mean[i], ]
                grids += [md.grid, ] 
                labels += [f'{pf}.{it} mean', ]
                vmins += [vmin, ]
                vmaxs += [vmax, ]

        if 'std' in mode:
            for i in range(mean.shape[0]):
                arrays += [std[i] / mean[i], ]
                grids += [md.grid, ] 
                labels += [f'{pf}.{it} std', ]
                vmins += [None, ]
                vmaxs += [None, ]
            
    plot_arrays(
        array = arrays,
        grid = grids,
        label = labels,
        rows = 2 if all(m in mode for m in ['mean', 'std']) else 1,
        vmin = vmins,
        vmax = vmaxs,
        **kwargs,
    )
    return



def plot_samples(
        model,
        samples,
        **kwargs,
):
    """
    Plot individual posterior samples for a given model.

    Parameters
    ----------
    model : object
        The model to evaluate and plot. Must be callable and have
        ``prefix`` and ``grid`` attributes.
    samples : MySamples
        The samples instance containing posterior samples.
    **kwargs
        Additional keyword arguments passed to ``plot_arrays``.
    """
    check_type(samples, MySamples)

    if len(samples) < 2:
        return
    
    arrays = [model(s) for s in samples]
    
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    if vmin is None:
        vmin = min([a.min() for a in arrays])
    if vmax is None:
        vmax = max([a.max() for a in arrays])

    [kwargs.pop(k, None) for k in ('rows', 'cols')]

    for i,a in enumerate(arrays):
        arrays[i] = a[a.shape[0]//2] if a.ndim == 3 else a

    plot_arrays(
        array = arrays,
        grid = model.grid,
        label = [f'{model.prefix} sample {i}' for i in range(len(samples))],
        vmin = vmin,
        vmax = vmax,
        rows = 1,
        **kwargs,
    )
    return



def plot_agreement(
        model,
        samples,
        data,
        **kwargs,
):
    """
    Plot the agreement between model predictions and data.

    Parameters
    ----------
    model : object
        The model to evaluate. Must have ``prefix`` and ``grid``
        attributes.
    samples : MySamples
        The samples instance containing posterior samples.
    data : ImageData
        The observed image data to compare against.
    **kwargs
        Additional keyword arguments passed to ``plot_arrays``.
    """
    check_type(samples, MySamples)
    
    if not isinstance(data, ImageData):
        return

    mean = samples.mean(model)

    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    if vmin is None:
        vmin = mean.min()
    if vmax is None:
        vmax = mean.max()

    if mean.shape != data.val.shape:
        mean = map_signal(model.grid, data.grid)(np.asarray(mean))

    [kwargs.pop(k, None) for k in ('rows', 'cols')]

    arrays = [mean, mean - data.val, data.val]

    for i,a in enumerate(arrays):
        arrays[i] = a[a.shape[0]//2] if a.ndim == 3 else a

    plot_arrays(
        array = arrays,
        grid = data.grid,
        label = [f'{model.prefix} mean', 'mean - truth', f'{data.prefix} thruth'],
        vmin = [vmin, None, vmin],
        vmax = [vmax, None, vmax],
        rows = 1,
        **kwargs,
    )
    return



def plot_pullplot(
        model,
        samples,
        data,
        **kwargs,
):
    """
    Plot a pullplot ``(mean - truth) / std`` for a given model, samples and ImageData.

    Parameters
    ----------
    model : object
        The model to evaluate. Must have ``prefix`` and ``grid``
        attributes.
    samples : MySamples
        The samples instance containing posterior samples.
    data : ImageData
        The observed image data to compare against.
    **kwargs
        Additional keyword arguments passed to ``plot_arrays``.
    """
    check_type(samples, MySamples)
    
    if not isinstance(data, ImageData) or len(samples) < 2:
        return

    mean, std = samples.mean_and_std(model)

    if mean.shape != data.val.shape:
        mean = map_signal(model.grid, data.grid)(np.asarray(mean))

    [kwargs.pop(k, None) for k in ('vmin', 'vmax', 'norm', 'rows', 'cols')]

    array = (mean - data.val) / std
    array = array[array.shape[0]//2] if array.ndim == 3 else array

    plot_arrays(
        array = array,
        grid = data.grid,
        label = f'{model.prefix} pullplot',
        norm = 'linear',
        vmin = -5.0,
        vmax = 5.0,
        rows = 1,
        **kwargs,
    )
    return

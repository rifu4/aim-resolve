from .arrays import plot_arrays
from ..img_data.data import ImageData
from ..model.map import map_signal
from ..model.util import check_type
from ..optimize.samples import MySamples



def plot_mean_and_std(
        model,
        samples,
        mode = 'mean_and_std',
        **kwargs,
):
    '''Plot the mean and standard deviation of samples for a given model.'''
    check_type(samples, MySamples)

    models = [model, ] if not isinstance(model, list) else model

    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)

    arrays, spaces, labels, vmins, vmaxs = [], [], [], [], []
    for md in models:
        pf, it = md.prefix.split('.')[0], md.prefix.split('.')[1]
        mean, std = samples.mean_and_std(md)

        spaces += [md.space, ]

        if 'mean' in mode:
            arrays += [mean, ]
            labels += [f'{pf}.{it} mean', ]
            vmins += [vmin, ]
            vmaxs += [vmax, ]

        if 'std' in mode:
            arrays += [std / mean, ]
            labels += [f'{pf}.{it} std', ]
            vmins += [None, ]
            vmaxs += [None, ]
            
    plot_arrays(
        array = arrays,
        space = spaces,
        label = labels, 
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
    '''Plot samples for a given model.'''
    check_type(samples, MySamples)

    if len(samples) < 2:
        return
    
    array = [model(s) for s in samples]
    
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    if vmin is None:
        vmin = min([a.min() for a in array])
    if vmax is None:
        vmax = max([a.max() for a in array])

    [kwargs.pop(k, None) for k in ('rows', 'cols')]

    plot_arrays(
        array = array,
        space = model.space,
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
    '''Plot the agreement between model predictions and data.'''
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
        mean = map_signal(mean, model.space, data.space)

    [kwargs.pop(k, None) for k in ('rows', 'cols')]

    plot_arrays(
        array = [mean, mean - data.val, data.val],
        space = data.space,
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
    '''Plot a pullplot `(mean - truth)/std` for a given model, samples and ImageData.'''
    check_type(samples, MySamples)
    
    if not isinstance(data, ImageData) or len(samples) < 2:
        return

    mean, std = samples.mean_and_std(model)

    if mean.shape != data.val.shape:
        mean = map_signal(mean, model.space, data.space)

    [kwargs.pop(k, None) for k in ('vmin', 'vmax', 'norm', 'rows', 'cols')]

    plot_arrays(
        array = (mean - data.val) / std,
        space = data.space,
        label = f'{model.prefix} pullplot',
        norm = 'linear',
        vmin = -5.0,
        vmax = 5.0,
        rows = 1,
        **kwargs,
    )
    return

import os
import pickle
import numpy as np
from jax import random
from functools import partial
from nifty.re import Model, Vector, random_like

from .mask import masks_from_model, masks_to_boxes, remove_freq_axis
from .modeling import get_offset
from .model.components import ComponentModel
from .model.map import map_signal
from .model.noise import NoiseModel
from .model.points import PointModel
from .model.signal import SignalModel
from .model.tiles import TileModel
from .model.util import check_type
from .optimize.opt_dct import callable_optimize_dict
from .optimize.opt_kl import optimize_kl
from .optimize.samples import MySamples, domain_keys, domain_tree, random_init
from .plot import plot_arrays



def transition_func(
        mode,
        **kwargs,
):
    '''
    Versatile transition function -> performs the wanted transition specified in the 'mode' parameter
    
    Parameters:
    -----------
    mode : str
        Transition mode. Can be `anew`, `copy`, `addt` or `zoom`. Default is `addt`.
    kwargs : dict
        Additional keyword arguments passed to the transition functions (see transition functions).
    '''
    if mode == 'anew':
        return partial(transition_anew, **kwargs)
    elif mode == 'freq':
        return partial(transition_freq, **kwargs)
    elif mode == 'addt':
        return partial(transition_util, func=transition_addt, **kwargs)
    elif mode == 'zoom':
        return partial(transition_util, func=transition_zoom, **kwargs)
    else:
        raise TypeError(f'Unknown transition mode. Available modes are `anew`, `freq`, `addt` and `zoom`, but got mode `{mode}`.')



def transition_anew(
        key,
        *args,
        lh_new,
        **kwargs,
):
    '''
    Gnerate new position from random for all parameters

    Parameters:
    -----------
    key : jax.random.PRNGKey
        Random key for the JAX random number generator.
    lh_new : nifty.re.Likelihood
        Likelihood model for the new optimiztion iteration
    '''
    models = [v for v in lh_new.values() if isinstance(v, Model)]
    if domain_keys(models) == set():
        raise ValueError('Check that sky and noise models in the `lh_dict` are of type `nifty.re.Model`')
    
    key, k_p = random.split(key)
    pos_new = random_init(k_p, models, factor=0.01)

    # print ptree
    # print('New model parameters:')
    # for k,v in pos_new.ptree.items():
    #     print(f'  {k}:', v.shape)

    return pos_new



def transition_freq(
        key,
        samples,
        *args,
        lh_old,
        lh_new,
        **kwargs,
):
    '''
    Generate new position from random for all parameters

    Parameters:
    -----------
    key : jax.random.PRNGKey
        Random key for the JAX random number generator.
    samples : nifty.re.Samples
        Samples from the previous iteration.
    lh_old : nifty.re.Likelihood
        Likelihood model for the previous optimiztion iteration.
    lh_new : nifty.re.Likelihood
        Likelihood model for the new optimiztion iteration.
    '''
    sky_old = lh_old['sky_model']
    sky_new = lh_new['sky_model']
    check_type(samples, MySamples)
    check_type(sky_old, ComponentModel)
    check_type(sky_new, ComponentModel)

    # initialize an empty position tree
    ptree = {}

    # copy over matching model components
    for (sky_oi, sky_ni) in zip(sky_old.models, sky_new.models):
        if sky_ni.grid != sky_oi.grid:
            raise ValueError('Old and new sky model components have to match.')
        for key_oi in domain_keys(sky_oi):
            key_ni = key_oi.replace(sky_oi.prefix, sky_ni.prefix)
            ptree[key_ni] = domain_tree(samples)[key_oi]

    # load learned noise scaling of the previous iteration if available
    nm_old = lh_old['noise_model']
    nm_new = lh_new['noise_model']
    if isinstance(nm_old, NoiseModel) and isinstance(nm_new, NoiseModel):
        ptree[nm_new.prefix] = np.broadcast_to(domain_tree(samples)[nm_old.prefix], nm_new.target.shape)

    # print ptree
    print('Copied model parameters:')
    for k,v in ptree.items():
        print(f'  {k}:', v.shape)

    key, k_p = random.split(key)
    pos_new = random_init(k_p, [v for v in lh_new.values() if isinstance(v, Model)], ptree, factor=0.01)

    samples_new = MySamples(pos=pos_new, samples=None, keys=None)

    return samples_new



def transition_util(
        key,
        samples,
        it,
        *,
        func,
        lh_new,
        odir = None,
        **kwargs,
):
        '''
        Check if a transition file exists and loads it. If not, performs the transition and saves the result.

        Parameters:
        -----------
        func : function
            Transition function to be called if no transition file exists.
        key : jax.random.PRNGKey
            Random key for the JAX random number generator.
        samples : nifty.re.Samples
            Samples from the previous iteration.
        it : int
            Current iteration number of the optimization.
        lh_new : nifty.re.Likelihood
            Likelihood model for the new optimiztion iteration.
        odir : str
            Output directory for plots and a transition pickle file.
        '''
        pos_fn = f'{odir}/{it}_trans.pkl' if odir else ''

        if odir:
            os.makedirs(odir, exist_ok=True)
        
        if os.path.isfile(pos_fn):
            samples = pickle.load(open(pos_fn, "rb"))
            models = [v for v in lh_new.values() if isinstance(v, Model)]
            if domain_keys(samples) == domain_keys(models):
                return samples
        
        samples, *_ = func(key, samples, it, lh_new=lh_new, odir=odir, **kwargs)
        
        if pos_fn:
            pickle.dump(samples, open(pos_fn, "wb"))

        return samples



def transition_addt(
        key,
        samples,
        it,
        *,
        lh_old,
        lh_new,
        opt_dct,
        offsets = False,
        odir = None,
        mask = None,
        noise = dict(max_std=1e-5, parameters=dict()),
        plot_dct = dict(norm='log'),
        **kwargs,
):
    '''
    Optimizes the new tile model on the previous reconstruction and separates tile components and point sources from the background.

    Parameters:
    -----------
    key : jax.random.PRNGKey
        Random key for the JAX random number generator.
    samples : nifty.re.Samples
        Samples from the previous iteration.
    it : int
        Current iteration number of the optimization.
    lh_old : nifty.re.Likelihood
        Likelihood model for the previous optimiztion iteration.
    lh_new : nifty.re.Likelihood
        Likelihood model for the new optimiztion iteration.
    offsets : bool
        If True, sets the offsets of the signal components depending on the background reconstruction and returns a dict containing the offsets.
    opt_dct : dict
        Dictionary containing the optimization parameters.
    odir : str
        Output directory for the plotting.
    mask : str
        Path to the mask file. If None, a new mask is created.
    noise : dict
        Dictionary containing the noise parameters.
    plot_dct : dict
        Dictionary containing the plotting parameters.
    '''
    sky_old = lh_old['sky_model']
    sky_new = lh_new['sky_model']
    check_type(samples, MySamples)
    check_type(sky_old, (ComponentModel, SignalModel, PointModel, TileModel))
    check_type(sky_new, ComponentModel)
    plot_dct = plot_dct.copy() | dict(label=None, grid=None, odir=odir)

    # get reconstruction of the previous iteration
    rec_old = np.asarray(samples.mean(sky_old))

    # load or create masks for an efficient separation of the components
    mask_fn = mask if mask else ''
    if os.path.isfile(mask_fn):
        mask_dct = dict(np.load(mask))
    else:
        mask_dct = masks_from_model(sky_new)
    mask_box = masks_to_boxes(sky_new, mask_dct)
    if odir:
        p_dct = plot_dct | dict(name=f'{it}_masks.png', norm='linear', vmin=0, vmax=1)
        mask_val = [remove_freq_axis(v, sky_new.freq) for v in mask_dct.values()]
        plot_arrays([np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_val], **p_dct)

    # initialize an empty position tree
    ptree = {}
    keys = list(random.split(key, 2 + len(sky_new.models)))

    # optimize the new background model on the old reconstruction (mask regions around point sources and the object boxes)
    sky_bg = sky_new.background.copy()
    rec_bg = map_signal(sky_old.grid, sky_bg.grid)(rec_old)
    msk_bg = mask_box[sky_bg.prefix]
    sky_bg.mask = msk_bg
    pos_bg = optimize_and_plot(
        key = keys.pop(),
        sky = sky_bg,
        data = np.where(msk_bg, rec_bg, 0.0),
        noise = noise.copy(),
        opt_dct = opt_dct,
        plot_dct = plot_dct | dict(name=f'{it}_{sky_bg.prefix}.png'),
    )
    ptree |= pos_bg.tree

    sky_bg.mask = None
    rec_sub = map_signal(sky_old.grid, sky_new.grid)(rec_old) - map_signal(sky_bg.grid, sky_new.grid)(np.asarray(sky_bg(pos_bg)))
    rec_sub = rec_sub.clip(0, None)
    ofs_dct = {}

    # optimize the new object and tile models on the corresponding regions of the old reconstruction
    for sky_ci in sky_new.points + sky_new.objects + sky_new.tiles:
        sub_ci = map_signal(sky_new.grid, sky_ci.grid)(rec_sub)
        msk_ci = mask_box[sky_ci.prefix]
        if offsets:
            ofs_dct[sky_ci.prefix] = get_offset(sky_ci, sub_ci, msk_ci, sky_ci.freq)
            sky_ci.set_offset(ofs_dct[sky_ci.prefix])
        if isinstance(sky_ci, (PointModel, TileModel)):
            msk_ci = msk_ci.sum(axis=0).clip(0, 1)
        pos_ci = optimize_and_plot(
            key = keys.pop(),
            sky = sky_ci,
            data = np.where(msk_ci, sub_ci, 0.0),
            noise = noise.copy(),
            opt_dct = opt_dct,
            plot_dct = plot_dct | dict(name=f'{it}_{sky_ci.prefix}.png'),
        )
        ptree |= pos_ci.tree

    rec_sky = map_signal(sky_old.grid, sky_new.grid)(rec_old)
    pos_sky = optimize_and_plot(
        key = keys.pop(),
        sky = sky_new,
        data = rec_sky,
        noise = noise.copy(),
        pos = Vector(ptree),
        opt_dct = None,
        plot_dct = plot_dct | dict(name=f'{it}_{sky_new.prefix}.png'),
    )
    ptree = pos_sky.tree

    # load learned noise scaling of the previous iteration if available
    nm_old = lh_old['noise_model']
    nm_new = lh_new['noise_model']
    if isinstance(nm_old, NoiseModel) and isinstance(nm_new, NoiseModel):
        ptree[nm_new.prefix] = map_signal(sky_old.grid, sky_new.grid)(domain_tree(samples)[nm_old.prefix])

    # print ptree
    # print('New model parameters:')
    # for k,v in ptree.items():
    #     print(f'  {k}:', v.shape)

    samples_new = MySamples(pos=Vector(ptree), samples=None, keys=None)

    return samples_new, ofs_dct



def transition_zoom(
        key,
        samples,
        it,
        *,
        lh_old,
        lh_new,
        opt_dct,
        odir = None,
        noise = dict(max_std=1e-5, parameters=dict()),
        plot_dct = dict(norm='log'),
        **kwargs,
):
    '''
    Optimizes the new tile model on the previous reconstruction and separates tile components and point sources from the background.

    Parameters:
    -----------
    key : jax.random.PRNGKey
        Random key for the JAX random number generator.
    samples : nifty.re.Samples
        Samples from the previous iteration.
    it : int
        Current iteration number of the optimization.
    lh_old : nifty.re.Likelihood
        Likelihood model for the previous optimiztion iteration.
    lh_new : nifty.re.Likelihood
        Likelihood model for the new optimiztion iteration.
    opt_dct : dict
        Dictionary containing the optimization parameters.
    odir : str
        Output directory for the plotting.
    noise : dict
        Dictionary containing the noise parameters.
    plot_dct : dict
        Dictionary containing the plotting parameters.
    '''
    sky_old = lh_old['sky_model']
    sky_new = lh_new['sky_model']
    check_type(samples, MySamples)
    check_type(sky_old, ComponentModel)
    check_type(sky_new, ComponentModel)
    plot_dct = plot_dct.copy() | dict(label=None, grid=None, odir=odir)

    # initialize an empty position tree
    ptree = {}
    keys = list(random.split(key, len(sky_new.models)+1))

    # copy over matching model components
    for (sky_oi, sky_ni) in zip(sky_old.models, sky_new.models):
        if not sky_ni.grid in sky_oi.grid and sky_oi.grid in sky_ni.grid:
            raise ValueError('Old and new sky model components have to match.')
        rec_oi = map_signal(sky_oi.grid, sky_ni.grid)(samples.mean(sky_oi))
        pos_ci = optimize_and_plot(
            key = keys.pop(),
            sky = sky_ni,
            data = rec_oi,
            noise = noise.copy(),
            opt_dct = opt_dct,
            plot_dct = plot_dct | dict(name=f'{it}_{sky_ni.prefix}.png'),
        )
        ptree |= pos_ci.tree

    rec_sky = map_signal(sky_old.grid, sky_new.grid)(samples.mean(sky_old))
    pos_sky = optimize_and_plot(
        key = keys.pop(),
        sky = sky_new,
        data = rec_sky,
        noise = noise.copy(),
        pos = Vector(ptree),
        opt_dct = None,
        plot_dct = plot_dct | dict(name=f'{it}_{sky_new.prefix}.png'),
    )
    ptree = pos_sky.tree

    # load learned noise scaling of the previous iteration if available
    nm_old = lh_old['noise_model']
    nm_new = lh_new['noise_model']
    if isinstance(nm_old, NoiseModel) and isinstance(nm_new, NoiseModel):
        ptree[nm_new.prefix] = map_signal(sky_old.grid, sky_new.grid)(domain_tree(samples)[nm_old.prefix])

    # print ptree
    print('New model parameters:')
    for k,v in ptree.items():
        print(f'  {k}:', v.shape)

    samples_new = MySamples(pos=Vector(ptree), samples=None, keys=None)

    return samples_new, None



def optimize_and_plot(
        key,
        sky, 
        data,
        pos = None,
        opt_dct = None,
        noise = dict(max_std=1e-5, parameters=dict()),
        plot_dct = dict(odir=None, name=None),
):
    '''Optimizes a sky model on the given data with the given optimization parameters and/or plots the results.'''
    if opt_dct:
        max_std = noise['max_std'] if 'max_std' in noise else 1e-5
        noise_model = NoiseModel.build(shape=data.shape, **noise)

        k_n, k_o = random.split(key)
        noise_std = max_std * np.max(data)
        noise_init = np.asarray(noise_std * random_like(k_n, sky.target))
        data = data + noise_init

        lh_dct = dict(
            data = data,
            sky_response = sky,
            noise_cov_inv = None,
            noise_std_inv = noise_std**-1,
            noise_model = noise_model,
        )

        if 'callback' in opt_dct and opt_dct['callback']:
            def callback(samples, opt_state, *_):
                p_dct = plot_dct | dict(odir=plot_dct['odir']+'/callback', name=f'{opt_state.nit}_{sky.prefix}')
                plot_arrays(samples.mean(sky), **p_dct)
        else:
            callback = None

        opt_dct = callable_optimize_dict(opt_dct)

        samples, _ = optimize_kl(lh_dct, key=k_o, position_or_samples=pos, callback=callback, **opt_dct)
        pos = samples.pos
    
    if plot_dct['odir']:
        [plot_dct.pop(k) for k in ['vmin', 'vmax'] if k in plot_dct]
        if isinstance(sky, SignalModel):
            sky.mask = None

        arrays = []
        for a in (data, sky(pos)):
            if a.ndim == 2:
                arrays += [a, ]
            elif a.ndim == 3:
                arrays += [a[i] for i in range(a.shape[0])]

        plot_arrays(
            array = arrays,
            rows = 2,
            vmin = max(sky(pos).min(), 1),
            vmax = sky(pos).max(),
            **plot_dct,
        )

    return pos

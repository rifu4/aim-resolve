import jax.numpy as jnp
import numpy as np
from functools import reduce
from nifty8 import makeOp
from operator import add

from .fast_resolve.response import build_exact_responses
from .fast_resolve.convolve import PSFConvolve, NInvConvolve
from .model.noise import NoiseModel
from .resolve.model import ComponentResponse
from .resolve.observation import Observation



def likelihood_func(
        mode,
        **kwargs,
):
    '''
    Versatile likelihood function -> uses the likelihood specified in the 'mode' parameter
    
    Parameters:
    -----------
    mode : str
        Likelihood mode. Available modes are 'image', `fast`, `radio`, `sum`.
    kwargs : dict
        Additional keyword arguments passed to the likelihood functions (see below).
    '''
    if 'image' in mode:
        return image_likelihood(**kwargs)
    elif 'fast' in mode:
        return fast_likelihood(**kwargs)
    elif 'radio' in mode:
        return radio_likelihood(**kwargs)
    elif 'sum' in mode:
        return likelihood_sum(**kwargs)
    else:
        raise TypeError(f'Unknown likelihood mode. Available modes are `image`, `fast`, `radio`, and `sum`, but got mode `{mode}`.')



def image_likelihood(*,
        sky,
        data,
        noise = dict(max_std=0.001, parameters=dict()),
):    
    '''
    Generate a likelihood function for the image data.
    
    Parameters
    ----------
    sky : Model
        The sky model input to the likelihood function.
    data : ImageData
        The image data to be reconstructed.
    noise : dict
        Dictionary containing the noise parameters (see NoiseModel).
    '''
    max_std = noise['max_std'] if 'max_std' in noise else 0.001
    noise_model = NoiseModel.build(shape=data.grid.shape, **noise)

    sky.set_out_grid(data.grid)

    lh_dct = dict(
        data = data.noisy_val,
        sky_model = sky,
        sky_response = sky,
        noise_cov_inv = None,
        noise_std_inv = (max_std * np.max(data.val))**-1,
        noise_model = noise_model,
    )
    return lh_dct



def radio_likelihood(*,
        sky,
        data,
        noise = dict(wgt_fac=1., parameters=dict()),
        wgridding = False,
):  
    '''
    Generate a likelihood function for the radio data.

    Parameters
    ----------
    sky : Model
        The sky model input to the likelihood function.
    data : Observation
        The radio data to be reconstructed.
    noise : dict
        Dictionary containing the noise parameters (see NoiseModel).
    wgridding : bool
        Whether to use wgridding or not.
    '''
    wgt_fac = noise['wgt_fac'] if 'wgt_fac' in noise else 1.
    noise_model = NoiseModel.build(shape=data.vis.shape, **noise)

    lh_dct = dict(
        data = data.vis,
        sky_model = sky,
        sky_response = ComponentResponse(sky, data, wgridding),
        noise_cov_inv = lambda x: wgt_fac * data.weight * x,
        noise_std_inv = None,
        noise_model = noise_model,
    )
    return lh_dct



def fast_likelihood(*,
        sky,
        data,
        psf_kernel_fn = '',
        n_inv_kernel_fn = '',
        noise = dict(parameters=dict()),
        split = {},
):
    '''
    Generate a fast likelihood function for the radio data (fast-resolve).

    Parameters
    ----------
    sky : Model
        The sky model input to the likelihood function.
    data : Observation
        The radio data to be reconstructed.
    psf_kernel_fn : callable
        The psf kernel filename. Create a new kernel if not specified.
    n_inv_kernel_fn : callable
        The noise kernel filename. Create a new kernel if not specified.
    noise : dict
        Dictionary containing the noise parameters (see NoiseModel).
    split : dict
        The parameters for kernel-splitting. Needs `size` and `factor`. Default is `{}` (no split).
    ''' 
    if isinstance(data, Observation):
        data = data.to_resolve_obs()
    obs = data.to_double_precision()

    print('sky model shape:', sky.target.shape)

    R, R_l, RNR, RNR_l = build_exact_responses(obs, sky.grid, sky.freq)

    N_inv = makeOp(obs.weight)
    data = R.adjoint(N_inv(obs.vis)).val
    print('dirty image shape:', data.shape)

    psf_conv = PSFConvolve.build(
        sky = sky,
        RNR_l = RNR_l,
        psf_kernel_fn = psf_kernel_fn,
        split = split,
    )

    sky_response = NInvConvolve.build(
        psf_conv = psf_conv,
        RNR = RNR,
        n_inv_kernel_fn = n_inv_kernel_fn,
        noise = noise,
    )

    lh_dct = dict(
        data = jnp.array(data),
        sky_model = sky,
        sky_response = sky_response,
        noise_model = sky_response.noise_model,
        RNR = RNR,
    )
    return lh_dct



def likelihood_sum(
        **lhs,
):
    '''
    Generate a likelihood function that is the sum of multiple likelihood functions.

    Parameters
    ----------
    lhs : dict
        Dictionary containing the likelihood functions to sum.
    '''
    return reduce(add, lhs.values())

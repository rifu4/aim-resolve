import jax.numpy as jnp
import numpy as np
from functools import reduce
from nifty8 import makeOp
from nifty.re import Model
from operator import add

from .fast_resolve.response import build_exact_responses
from .fast_resolve.convolve import PSFConvolve, NInvConvolve
from .model.noise import NoiseModel
from .resolve.model import ComponentResponse
from .resolve.observation import Observation



def image_likelihood(*,
        sky,
        data,
        noise = dict(max_std=0.001, parameters=dict()),
        fun = 'exp',
):    
    '''
    Generate a likelihood function for the image data.
    
    Parameters
    ----------
    sky : Model
        The sky model input to the likelihood function.
    data : ImageData
        The data model input to the likelihood function.
    noise : dict
        Dictionary containing the noise parameters (see NoiseModel).
    fun : str, optional
        Used to differentiate between the different likelihood functions.

    '''
    max_std = noise['max_std'] if 'max_std' in noise else 0.001
    noise_model = NoiseModel.build(shape=data.grid.shape, **noise)

    sky.set_out_grid(data.grid)

    lh_dct = dict(
        data = data.noisy_val,
        model = sky,
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
        fun = 'radio',
):  
    '''
    Generate a likelihood function for the radio data.

    Parameters
    ----------
    sky : Model
        The sky model input to the likelihood function.
    data : Observation
        The data model input to the likelihood function.
    noise : dict
        Dictionary containing the noise parameters (see NoiseModel).
    wgridding : bool
        Whether to use wgridding or not.
    fun : str, optional
        Used to differentiate between the different likelihood functions.
    '''
    wgt_fac = noise['wgt_fac'] if 'wgt_fac' in noise else 1.
    noise_model = NoiseModel.build(shape=data.vis.shape, **noise)

    lh_dct = dict(
        data = data.vis,
        model = ComponentResponse(sky, data, wgridding),
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
        split = 0,
        fun = 'fast_radio',
        **kwargs,
):
    '''
    Generate a fast likelihood function for the radio data (fast-resolve).

    Parameters
    ----------
    sky : Model
        The sky model input to the likelihood function.
    data : Observation
        The data model input to the likelihood function.
    psf_pixels : int
        The maximal number of pixels in the PSF.
    response_kernel : callable
        The response kernel file. Create a new kernel if not specified.
    noise_kernel : callable
        The noise kernel file. Create a new kernel if not specified.
    noise : dict
        Dictionary containing the noise parameters (see NoiseModel).
    fun : str, optional
        Used to differentiate between the different likelihood functions.
    ''' 
    if isinstance(data, Observation):
        data = data.to_resolve_obs()
    obs = data.to_double_precision()

    R, R_l, RNR, RNR_l = build_exact_responses(obs, sky.grid, sky.freq)

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

    N_inv = makeOp(obs.weight)
    data = R.adjoint(N_inv(obs.vis))
    data = jnp.array(data.val)

    lh_dct = dict(
        data = data,
        sky = sky,
        RNR = RNR,
        sky_response = sky_response,
        noise_model = sky_response.noise_model,
    )
    return lh_dct



def likelihood_sum(*,
        fun = 'sum',
        **lhs,
):
    '''
    Generate a likelihood function that is the sum of multiple likelihood functions.

    Parameters
    ----------
    fun : str
        Used to differentiate between the different likelihood functions.
    lhs : dict
        Dictionary containing the likelihood functions to sum.
    '''
    return reduce(add, lhs.values())

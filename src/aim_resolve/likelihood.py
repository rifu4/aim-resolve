import jax.numpy as jnp
import numpy as np
from functools import reduce
from nifty8 import makeOp
from nifty8.re import Model
from operator import add

from .model.noise import NoiseModel
from .resolve.fast import build_exact_responses, build_approximation_kernels
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
    noise_model = NoiseModel.build(shape=data.space.shape, **noise)

    lh_dct = dict(
        data = data.noisy_val,
        model = Model(lambda x: sky(x, out_space=data.space), domain=sky.domain, init=sky.init),
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
        psf_pixels = 3000,
        response_kernel = None,
        noise_kernel = None,
        noise = dict(parameters=dict()),
        fun = 'fast_radio',
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

    R, R_l, RNR, RNR_l = build_exact_responses(obs, sky.space, psf_pixels)

    noise_model = NoiseModel.build(shape=R.domain.shape, **noise)

    RNR_approx, N_inv_approx = build_approximation_kernels(
        RNR = RNR,
        RNR_l = RNR_l,
        response_kernel_fn = response_kernel,
        noise_kernel_fn = noise_kernel,
        noise_model = noise_model,
    )

    N_inv = makeOp(obs.weight)
    data = R.adjoint(N_inv(obs.vis))
    data = jnp.array(data.val)

    lh_dct = dict(
        data = data,
        model = sky,
        R = RNR,
        R_approx = RNR_approx,
        N_inv_sqrt = N_inv_approx,
        noise_model = noise_model,
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

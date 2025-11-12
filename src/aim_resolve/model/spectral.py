import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jubik0 import build_simple_spectral_sky
from jubik0.sky_model.multifrequency.spectral_product_utils.frequency_deviations import build_frequency_deviations_model_with_degeneracies
from nifty.re import Model, VModel
from typing import Callable

from .grid import SignalGrid, PointGrid
from .prior import prior_model
from .util import check_type
from ..optimize.samples import domain_tree, model_init



UBIK_KEYS = {'zero_mode', 'spatial_amplitude', 'spectral_index', 'spectral_amplitude', 'deviations', 'nonlinearity'}
MF_KEYS = {'i0', 'alpha', 'deviations', 'nonlinearity'}



def spectral_model(
        prefix,
        space,
        freq = np.ones((1,)),
        nonlinearity = None,
        n_copies = 1,
        **params,
):
    '''
    Initialize one of the spectral models based on the provided parameters.
    
    Parameters
    ----------
    prefix : str
        The prefix for the model.
    space : SignalSpace or PointSpace
        The space for the model.
    freq : np.ndarray
        The freq of the model.
    nonlinearity : str, optional
        The nonlinearity function to apply to the model. Default is None.
    n_copies : int
        The number of copies for the model. Default is 1.
    params : dict
        The parameters for the model (see the specific model for details)
    
    Returns
    -------
    model : Model
        The initialized model.
    '''
    check_type(prefix, str)
    check_type(space, (SignalGrid, PointGrid))
    check_type(freq, np.ndarray)
    check_type(nonlinearity, (Callable, type(None)))
    check_type(n_copies, int)

    match set(params.keys()):
        case k if k.issubset(UBIK_KEYS):
            check_type(space, SignalGrid)
            check_type(nonlinearity, Callable)
            model = spectral_ubik_model(
                prefix=prefix,
                space=space,
                freq=freq,
                nonlinearity=nonlinearity,
                n_copies=n_copies,
                **params
            )
        case k if k.issubset(MF_KEYS):
            model = spectral_prior_model(
                prefix=prefix,
                space=space,
                freq=freq,
                nonlinearity=nonlinearity,
                n_copies=n_copies,
                **params,
            )
        case _:
            print(set(params.keys()))
            raise ValueError('Invalid parameters for spectral model')

    return model



def spectral_ubik_model(*,
        prefix,
        space,
        freq,
        zero_mode,
        spatial_amplitude,
        spectral_index,
        spectral_amplitude = None,
        deviations = None,
        nonlinearity = jnp.exp,
        n_copies = 1,
    ):
    '''
    Function to create a diffuse signal model on a specific space.

    Parameters
    ----------
    '''
    if freq.size == 1:
        raise ValueError('Need at least two frequencies for spectral ubik model.')
    
    log_freq = np.log(freq)
    reference_frequency_index = len(freq) // 2

    model = build_simple_spectral_sky(
        prefix,
        space.shape,
        space.distances,
        log_freq,
        reference_frequency_index,
        zero_mode,
        spatial_amplitude,
        spectral_index,
        spectral_amplitude_settings = spectral_amplitude,
        deviations_settings = deviations,
        nonlinearity = nonlinearity,
    )

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model



def spectral_prior_model(*,
        prefix,
        space,
        freq = np.ones((1,)),
        i0,
        alpha = None,
        deviations = None,
        nonlinearity = jnp.exp,
        n_copies = 1,
):
    '''
    Function to create a point signal model on a specific space.

    Parameters
    ----------
    '''
    i0, _ = prior_model(f'{prefix}i0 ', space, **i0)

    if freq.size == 1:
        model = MultiFrequencyModel(i0, nonlinearity=nonlinearity)

    else:
        log_freq = np.log(freq)
        reference_frequency_index = len(freq) // 2
        log_freq -= log_freq[reference_frequency_index]

        if not alpha:
            raise ValueError('Need alpha parameters to build multi-frequency model.')

        alpha, _ = prior_model(f'{prefix}alpha ', space, **alpha)

        if deviations:
            deviations = build_frequency_deviations_model_with_degeneracies(
                space.shape,
                log_freq,
                reference_frequency_index,
                deviations,
                prefix=f'{prefix}dev ',
            )

        model = MultiFrequencyModel(i0, log_freq, alpha, deviations, nonlinearity)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model



class MultiFrequencyModel(Model):
    def __init__(self, i0, log_freq=None, alpha=None, deviations=None, nonlinearity=jnp.exp):
        check_type(i0, Model)
        check_type(log_freq, (np.ndarray, type(None)))
        check_type(alpha, (Model, type(None)))
        check_type(deviations, (Model, type(None)))
        check_type(nonlinearity, (Callable, type(None)))

        self.i0 = i0
        self.log_freq = log_freq
        self.alpha = alpha
        self.deviations = deviations
        self.nonlinearity = nonlinearity
        super().__init__(
            domain = domain_tree((self.i0, self.alpha, self.deviations), error=False), 
            init = model_init((self.i0, self.alpha, self.deviations), error=False),
        ) 

    def __call__(self, x):
        res = self.i0(x)
        if self.alpha:
            res += jnp.outer(self.log_freq, self.alpha(x)).reshape(self.log_freq.shape + self.alpha.target.shape)
        if self.deviations:
            res += self.deviations(x)
        if self.nonlinearity:
            res = self.nonlinearity(res)
        return res

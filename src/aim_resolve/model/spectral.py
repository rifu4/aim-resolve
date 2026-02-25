"""Spectral modeling utilities for multi-frequency reconstruction."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jubik0 import build_simple_spectral_sky
from jubik0.sky_model.multifrequency.spectral_product_utils.frequency_deviations import (
    build_frequency_deviations_model_with_degeneracies,
)
from nifty.re import Model, VModel

from ..optimize.samples import domain_tree, model_init
from .grid import PointGrid, SignalGrid
from .prior import prior_model
from .util import check_type

UBIK_KEYS = {
    "zero_mode",
    "spatial_amplitude",
    "spectral_index",
    "spectral_amplitude",
    "deviations",
    "nonlinearity",
    "ref_freq_index",
}
MF_KEYS = {"i0", "alpha", "deviations", "nonlinearity", "ref_freq_index"}


def spectral_model(
    prefix,
    grid,
    freq=np.ones((1,)),
    nonlinearity=None,
    n_copies=1,
    **params,
):
    """
    Initialize one of the spectral models based on the provided parameters.

    Parameters
    ----------
    prefix : str
        The prefix for the model.
    grid : SignalGrid or PointGrid
        The grid for the model.
    freq : np.ndarray
        The frequencies of the model.
    nonlinearity : str, optional
        The nonlinearity function to apply to the model. Default is None.
    n_copies : int
        The number of copies for the model. Default is 1.
    params : dict
        The parameters for the model (see the specific model for details).

    Returns
    -------
    model : Model
        The initialized model.
    """
    check_type(prefix, str)
    check_type(grid, (SignalGrid, PointGrid))
    check_type(freq, np.ndarray)
    check_type(nonlinearity, (Callable, type(None)))
    check_type(n_copies, int)

    match set(params.keys()):
        case k if k.issubset(UBIK_KEYS):
            check_type(grid, SignalGrid)
            check_type(nonlinearity, Callable)
            model = spectral_ubik_model(
                prefix=prefix,
                grid=grid,
                freq=freq,
                nonlinearity=nonlinearity,
                n_copies=n_copies,
                **params,
            )
        case k if k.issubset(MF_KEYS):
            model = spectral_prior_model(
                prefix=prefix,
                grid=grid,
                freq=freq,
                nonlinearity=nonlinearity,
                n_copies=n_copies,
                **params,
            )
        case _:
            print(set(params.keys()))
            raise ValueError("Invalid parameters for spectral model")

    return model


def spectral_ubik_model(
    *,
    prefix,
    grid,
    freq,
    zero_mode,
    spatial_amplitude,
    spectral_index,
    spectral_amplitude=None,
    deviations=None,
    nonlinearity=jnp.exp,
    ref_freq_index=None,
    n_copies=1,
):
    """
    Create a diffuse signal model using the ubik spectral sky model.

    Parameters
    ----------
    prefix : str
        The prefix for the model.
    grid : SignalGrid
        The signal grid for the model.
    freq : np.ndarray
        The frequencies of the model (must have at least two entries).
    zero_mode : dict
        Zero-mode configuration for the spectral sky.
    spatial_amplitude : dict
        Spatial amplitude configuration.
    spectral_index : dict
        Spectral index configuration.
    spectral_amplitude : dict, optional
        Spectral amplitude configuration. Default is None.
    deviations : dict, optional
        Deviations configuration. Default is None.
    nonlinearity : callable, optional
        Nonlinearity function. Default is ``jnp.exp``.
    ref_freq_index : int, optional
        Reference frequency index. Default is None (uses midpoint).
    n_copies : int, optional
        The number of copies for the model. Default is 1.

    Returns
    -------
    model : Model or VModel
        The initialized ubik spectral model.
    """
    if freq.size == 1:
        raise ValueError("Need at least two frequencies for spectral ubik model.")

    log_freq = np.log(freq)
    ref_freq_index = (
        ref_freq_index if isinstance(ref_freq_index, int) else len(freq) // 2
    )

    model = build_simple_spectral_sky(
        prefix,
        grid.shape,
        grid.distances,
        log_freq,
        ref_freq_index,
        zero_mode,
        spatial_amplitude,
        spectral_index,
        spectral_amplitude_settings=spectral_amplitude,
        deviations_settings=deviations,
        nonlinearity=nonlinearity,
    )

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model


def spectral_prior_model(
    *,
    prefix,
    grid,
    freq=np.ones((1,)),
    i0,
    alpha=None,
    deviations=None,
    nonlinearity=jnp.exp,
    ref_freq_index=None,
    n_copies=1,
):
    """
    Create a single- or multi-frequency signal model on a specific grid.

    Parameters
    ----------
    prefix : str
        The prefix for the model.
    grid : SignalGrid or PointGrid
        The grid for the model.
    freq : np.ndarray, optional
        The frequencies of the model. Default is ``np.ones((1,))``.
    i0 : dict
        Prior model parameters for the reference frequency distribution.
    alpha : dict, optional
        Prior model parameters for the spectral index. Default is None.
    deviations : dict, optional
        Spectral deviations configuration. Default is None.
    nonlinearity : callable, optional
        Nonlinearity function. Default is ``jnp.exp``.
    ref_freq_index : int, optional
        Reference frequency index. Default is None (uses midpoint).
    n_copies : int, optional
        The number of copies for the model. Default is 1.

    Returns
    -------
    model : Model or VModel
        The initialized spectral prior model.
    """
    i0, _ = prior_model(f"{prefix}i0 ", grid, **i0)

    if freq.size == 1:
        model = MultiFrequencyModel(i0, nonlinearity=nonlinearity)

    else:
        log_freq = np.log(freq)
        ref_freq_index = (
            ref_freq_index if isinstance(ref_freq_index, int) else len(freq) // 2
        )
        log_freq -= log_freq[ref_freq_index]

        if not alpha:
            raise ValueError("Need alpha parameters to build multi-frequency model.")

        alpha, _ = prior_model(f"{prefix}alpha ", grid, **alpha)

        if deviations:
            deviations = build_frequency_deviations_model_with_degeneracies(
                grid.shape,
                log_freq,
                ref_freq_index,
                deviations,
                prefix=f"{prefix}dev ",
            )

        model = MultiFrequencyModel(i0, log_freq, alpha, deviations, nonlinearity)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model


class MultiFrequencyModel(Model):
    """Model combining reference frequency, spectral index, and deviations.

    Parameters
    ----------
    i0 : Model or None, optional
        Reference frequency distribution model. Default is None.
    log_freq : np.ndarray or None, optional
        Log-frequencies array. Default is None.
    alpha : Model or None, optional
        Spectral index model. Default is None.
    deviations : Model or None, optional
        Spectral deviations model. Default is None.
    nonlinearity : callable or None, optional
        Nonlinearity applied after summation. Default is ``jnp.exp``.
    """

    def __init__(
        self, i0=None, log_freq=None, alpha=None, deviations=None, nonlinearity=jnp.exp
    ):
        check_type(i0, (Model, type(None)))
        check_type(log_freq, (np.ndarray, type(None)))
        check_type(alpha, (Model, type(None)))
        check_type(deviations, (Model, type(None)))
        check_type(nonlinearity, (Callable, type(None)))

        self.shape = (
            i0.target.shape
            if i0
            else alpha.target.shape
            if alpha
            else deviations.target.shape
        )
        self.i0 = i0
        self.log_freq = log_freq
        self.alpha = alpha
        self.deviations = deviations
        self.nonlinearity = nonlinearity
        super().__init__(
            domain=domain_tree((self.i0, self.alpha, self.deviations), error=False),
            init=model_init((self.i0, self.alpha, self.deviations), error=False),
        )

    def __call__(self, x):
        """Evaluate the multi-frequency model.

        Parameters
        ----------
        x : dict
            Latent parameter dictionary.

        Returns
        -------
        res : jnp.ndarray
            The evaluated multi-frequency signal.
        """
        res = jnp.zeros(self.shape)
        if self.i0:
            res += self.i0(x)
        if self.alpha:
            res += jnp.outer(self.log_freq, self.alpha(x)).reshape(
                self.log_freq.shape + self.alpha.target.shape
            )
        if self.deviations:
            res += self.deviations(x)
        if self.nonlinearity:
            res = self.nonlinearity(res)
        return res

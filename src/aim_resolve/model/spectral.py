"""Spectral modeling utilities for multi-frequency reconstruction."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from nifty.re import Model, VModel

from ..multi_frequency import (
    build_frequency_deviations_model_with_degeneracies,
    build_simple_spectral_sky,
)
from ..optimize.samples import domain_tree, model_init
from .grid import PointGrid, SignalGrid
from .prior import prior_model
from .util import check_type

UBIK_KEYS = {
    "zero_mode",
    "spatial_amplitude",
    "spectral_index",
    "spectral_curvature",
    "spectral_amplitude",
    "deviations",
    "nonlinearity",
    "ref_freq_index",
}
MF_KEYS = {"i0", "alpha", "beta", "deviations", "nonlinearity", "ref_freq_index"}


def spectral_model(
    prefix,
    grid,
    freq=None,
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
    if freq is None:
        freq = np.ones((1,))
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
    spectral_index=None,
    spectral_amplitude=None,
    deviations=None,
    spectral_curvature=None,
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
        The frequencies of the model. With a single frequency the model is the
        reference-frequency field only, with exactly the parameters (names and
        parametrisation) of the multi-frequency model, so that a frequency
        transition carries the sky over unchanged; the spectral settings are
        then ignored.
    zero_mode : dict
        Zero-mode configuration for the spectral sky.
    spatial_amplitude : dict
        Spatial amplitude configuration.
    spectral_index : dict, optional
        Spectral index configuration (needed for several frequencies).
    spectral_amplitude : dict, optional
        Spectral amplitude configuration. Default is None.
    deviations : dict, optional
        Deviations configuration. Default is None.
    spectral_curvature : dict, optional
        Spectral curvature configuration (``mean``, ``fluctuations``), see
        ``SpectralIndexCurvature``. Default is None (no curvature term).
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
    single = freq.size == 1
    if single:
        # reference-frequency field only: the spectral parts are not part of the
        # model (SpectralProductSky); placeholder settings for the unused index
        spectral_index = dict(mean=(0.0, 1.0), fluctuations=(1.0, 1.0))
        spectral_amplitude = deviations = spectral_curvature = None
        ref_freq_index = 0
    elif not spectral_index:
        raise ValueError("Need spectral_index parameters for a multi-frequency ubik model.")

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
        spectral_curvature_settings=spectral_curvature,
        nonlinearity=nonlinearity,
    )
    if single:
        # SpectralProductSky keeps a frequency axis of length 1; single-frequency models are (*shape)
        mf_model = model
        model = Model(lambda x: mf_model(x)[0], domain=mf_model.domain, init=mf_model.init)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model


def spectral_prior_model(
    *,
    prefix,
    grid,
    freq=None,
    i0,
    alpha=None,
    beta=None,
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
    beta : dict, optional
        Prior model parameters for the spectral curvature (see
        ``MultiFrequencyModel``). Default is None (no curvature term).
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
    if freq is None:
        freq = np.ones((1,))
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
        if beta:
            beta, _ = prior_model(f"{prefix}beta ", grid, **beta)

        if deviations:
            deviations = build_frequency_deviations_model_with_degeneracies(
                grid.shape,
                log_freq,
                ref_freq_index,
                deviations,
                prefix=f"{prefix}dev ",
            )

        model = MultiFrequencyModel(i0, log_freq, alpha, deviations, nonlinearity, beta=beta or None)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model


class MultiFrequencyModel(Model):
    r"""Model combining reference frequency, spectral index, curvature and deviations.

    .. math::
        \log S(\nu) = i_0 + \alpha\, u + \beta\, q(u) + \delta_\perp(u),

    with the relative log-frequencies :math:`u = \log(\nu / \nu_\mathrm{ref})`
    (``log_freq``) and :math:`q(u) = u^2 - (\sum u^3 / \sum u^2)\, u`, i.e.
    :math:`u^2` orthogonalised against :math:`u` (as in
    ``SpectralIndexCurvature``). The deviations are projected onto the
    complement of :math:`u` (and of :math:`q` with a curvature), so they carry
    no slope (curvature) that is degenerate with :math:`\alpha` (:math:`\beta`);
    as both basis functions vanish at the reference frequency, :math:`i_0` stays
    the reference-frequency map.

    Parameters
    ----------
    i0 : Model or None, optional
        Reference frequency distribution model. Default is None.
    log_freq : np.ndarray or None, optional
        Relative log-frequencies array. Default is None.
    alpha : Model or None, optional
        Spectral index model. Default is None.
    deviations : Model or None, optional
        Spectral deviations model. Default is None.
    nonlinearity : callable or None, optional
        Nonlinearity applied after summation. Default is ``jnp.exp``.
    beta : Model or None, optional
        Spectral curvature model (needs at least three frequencies). Default is
        None.
    """

    def __init__(
        self, i0=None, log_freq=None, alpha=None, deviations=None, nonlinearity=jnp.exp, beta=None
    ):
        check_type(i0, (Model, type(None)))
        check_type(log_freq, (np.ndarray, type(None)))
        check_type(alpha, (Model, type(None)))
        check_type(beta, (Model, type(None)))
        check_type(deviations, (Model, type(None)))
        check_type(nonlinearity, (Callable, type(None)))
        if beta is not None and (log_freq is None or log_freq.size < 3):
            raise ValueError("A spectral curvature needs at least three frequencies.")

        self.shape = (
            i0.target.shape
            if i0
            else alpha.target.shape
            if alpha
            else beta.target.shape
            if beta
            else deviations.target.shape
        )
        self.i0 = i0
        self.log_freq = log_freq
        self.alpha = alpha
        self.beta = beta
        self.deviations = deviations
        self.nonlinearity = nonlinearity
        if log_freq is not None and log_freq.size > 1:
            u = np.asarray(log_freq, dtype="float64")
            self.curvature_basis = u**2 - np.sum(u**3) / np.sum(u**2) * u
        else:
            self.curvature_basis = None
        super().__init__(
            domain=domain_tree((self.i0, self.alpha, self.beta, self.deviations), error=False),
            init=model_init((self.i0, self.alpha, self.beta, self.deviations), error=False),
        )

    def _outer(self, basis, field):
        return jnp.outer(basis, field).reshape(basis.shape + field.shape)

    def projected_deviations(self, x):
        """Deviations without their components along u (and q with a curvature)."""
        dev = self.deviations(x)
        if self.log_freq is None or self.log_freq.size < 2:
            return dev
        shp = (-1,) + (1,) * (dev.ndim - 1)
        for b in ([self.log_freq, self.curvature_basis] if self.beta is not None else [self.log_freq]):
            b = jnp.asarray(b).reshape(shp)
            dev = dev - jnp.sum(dev * b, axis=0) / jnp.sum(b**2) * b
        return dev

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
            res += self._outer(self.log_freq, self.alpha(x))
        if self.beta:
            res += self._outer(self.curvature_basis, self.beta(x))
        if self.deviations:
            res += self.projected_deviations(x)
        if self.nonlinearity:
            res = self.nonlinearity(res)
        return res

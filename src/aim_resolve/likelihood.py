"""Likelihood construction utilities for image and radio data."""

from functools import reduce
from operator import add

import jax.numpy as jnp
import numpy as np
from nifty8 import makeOp

from .fast_resolve.convolve import NInvConvolve, PSFConvolve
from .fast_resolve.response import build_exact_responses
from .model.noise import NoiseModel
from .resolve.model import ComponentResponse
from .resolve.observation import Observation


def likelihood_func(
    mode,
    **kwargs,
):
    """Create a likelihood dictionary using the mode-specific builder.

    Parameters
    ----------
    mode : {'image', 'fast', 'radio', 'sum'}
        Likelihood mode.
    **kwargs
        Additional keyword arguments forwarded to the selected builder.

    Returns
    -------
    lh_dct : dict
        Likelihood dictionary consumed by the optimizer.

    Raises
    ------
    TypeError
        If *mode* is not recognised.
    """
    if "image" in mode:
        return image_likelihood(**kwargs)
    elif "fast" in mode:
        return fast_likelihood(**kwargs)
    elif "radio" in mode:
        return radio_likelihood(**kwargs)
    elif "sum" in mode:
        return likelihood_sum(**kwargs)
    else:
        raise TypeError(
            f"Unknown likelihood mode. Available modes are `image`, `fast`, `radio`, and `sum`, but got mode `{mode}`."
        )


def image_likelihood(
    *,
    sky,
    data,
    noise=None,
):
    """Build a likelihood dictionary for image data.

    Parameters
    ----------
    sky : Model
        Sky model whose output grid is set to match *data*.
    data : ImageData
        Image data to reconstruct.
    noise : dict, optional
        Noise configuration forwarded to ``NoiseModel.build``.
        Must contain ``max_std`` and ``parameters``. Default uses
        ``max_std=0.001``.

    Returns
    -------
    lh_dct : dict
        Likelihood dictionary with keys ``data``, ``sky_model``,
        ``sky_response``, ``noise_cov_inv``, ``noise_std_inv`` and
        ``noise_model``.
    """
    if noise is None:
        noise = {"max_std": 0.001, "parameters": {}}
    max_std = noise.get("max_std", 0.001)
    noise_model = NoiseModel.build(shape=data.grid.shape, **noise)

    sky.set_out_grid(data.grid)

    lh_dct = dict(
        data=data.noisy_val,
        sky_model=sky,
        sky_response=sky,
        noise_cov_inv=None,
        noise_std_inv=(max_std * np.max(data.val)) ** -1,
        noise_model=noise_model,
    )
    return lh_dct


def radio_likelihood(
    *,
    sky,
    data,
    noise=None,
    wgridding=False,
):
    """Build a likelihood dictionary for radio visibility data.

    Parameters
    ----------
    sky : Model
        Sky model used as the signal model.
    data : Observation
        Radio observation to reconstruct.
    noise : dict, optional
        Noise configuration. The ``wgt_fac`` key scales the visibility
        weights. Default uses ``wgt_fac=1.0``.
    wgridding : bool, optional
        Whether to use w-gridding for the sky response. Default is False.

    Returns
    -------
    lh_dct : dict
        Likelihood dictionary with keys ``data``, ``sky_model``,
        ``sky_response``, ``noise_cov_inv``, ``noise_std_inv`` and
        ``noise_model``.
    """
    if noise is None:
        noise = {"wgt_fac": 1.0, "parameters": {}}
    wgt_fac = noise.get("wgt_fac", 1.0)
    noise_model = NoiseModel.build(shape=data.vis.shape, **noise)

    lh_dct = dict(
        data=data.vis,
        sky_model=sky,
        sky_response=ComponentResponse(sky, data, wgridding),
        noise_cov_inv=lambda x: wgt_fac * data.weight * x,
        noise_std_inv=None,
        noise_model=noise_model,
    )
    return lh_dct


def fast_likelihood(
    *,
    sky,
    data,
    psf_kernel_fn="",
    n_inv_kernel_fn="",
    noise=None,
    split=None,
):
    """Build a fast-resolve likelihood dictionary for radio data.

    Uses pre-computed PSF and noise-inverse convolution kernels for an
    efficient approximation of the full radio likelihood.

    Parameters
    ----------
    sky : Model
        Sky model used as the signal model.
    data : Observation
        Radio observation to reconstruct.
    psf_kernel_fn : str, optional
        Path to a cached PSF kernel file. A new kernel is created when
        empty. Default is ``''``.
    n_inv_kernel_fn : str, optional
        Path to a cached noise-inverse kernel file. A new kernel is
        created when empty. Default is ``''``.
    noise : dict, optional
        Noise configuration forwarded to ``NoiseModel.build``.
    split : dict, optional
        Kernel-splitting parameters (``size`` and ``factor``). Default is
        ``{}`` (no splitting).

    Returns
    -------
    lh_dct : dict
        Likelihood dictionary with keys ``data``, ``sky_model``,
        ``sky_response``, ``noise_model`` and ``RNR``.
    """
    if noise is None:
        noise = {"parameters": {}}
    if split is None:
        split = {}
    if isinstance(data, Observation):
        data = data.to_resolve_obs()
    obs = data.to_double_precision()

    print("sky model shape:", sky.target.shape)

    R, R_l, RNR, RNR_l = build_exact_responses(obs, sky.grid, sky.freq)

    N_inv = makeOp(obs.weight)
    data = R.adjoint(N_inv(obs.vis)).val
    print("dirty image shape:", data.shape)

    psf_conv = PSFConvolve.build(
        sky=sky,
        RNR_l=RNR_l,
        psf_kernel_fn=psf_kernel_fn,
        split=split,
    )

    sky_response = NInvConvolve.build(
        psf_conv=psf_conv,
        RNR=RNR,
        n_inv_kernel_fn=n_inv_kernel_fn,
        noise=noise,
    )

    lh_dct = dict(
        data=jnp.array(data),
        sky_model=sky,
        sky_response=sky_response,
        noise_model=sky_response.noise_model,
        RNR=RNR,
    )
    return lh_dct


def likelihood_sum(
    **lhs,
):
    """Sum multiple likelihood objects into a single composite likelihood.

    Parameters
    ----------
    **lhs
        Named likelihood objects to be summed.

    Returns
    -------
    likelihood
        The combined likelihood (sum of all inputs).
    """
    return reduce(add, lhs.values())

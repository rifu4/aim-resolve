"""Likelihood construction utilities for image and radio data."""

from functools import reduce
from operator import add

import jax.numpy as jnp
import numpy as np
from nifty8 import makeOp

from .fast_resolve.beam import BeamedSky, build_primary_beam
from .fast_resolve.convolve import NInvConvolve, PSFConvolve
from .fast_resolve.response import ChannelResidual, PaddedRNR, build_exact_responses
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
    wgridding=True,
    pb_correction=False,
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
    pb_correction : bool or dict, optional
        Not supported in radio mode (raises if set); see ``fast_likelihood``.

    Returns
    -------
    lh_dct : dict
        Likelihood dictionary with keys ``data``, ``sky_model``,
        ``sky_response``, ``noise_cov_inv``, ``noise_std_inv`` and
        ``noise_model``.
    """
    if pb_correction:
        raise NotImplementedError(
            "pb_correction is only implemented for mode: fast. The radio response "
            "maps each sky component on its own grid and would need per-component beams."
        )
    if noise is None:
        noise = {"wgt_fac": 1.0, "parameters": {}}
    wgt_fac = noise.get("wgt_fac", 1.0)
    noise_model = NoiseModel.build(shape=data.vis.shape, **noise)

    lh_dct = dict(
        data=data.vis,
        sky_model=sky,
        sky_response=ComponentResponse(sky, data, True, wgridding),
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
    pb_correction=False,
    freq_bounds=None,
    psf_from_n_inv=False,
    channel_residual=False,
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
    pb_correction : bool or dict, optional
        Include the primary beam in the forward model: the response sees the
        apparent sky ``beam * sky`` while ``sky`` stays the intrinsic sky.
        ``True`` uses the defaults of ``build_primary_beam``; a dict sets its
        options (``model``, ``channel_average``, ``n_channels``,
        ``pointing_offset``). The beam is computed once here; PSF and noise
        kernels do not depend on it. Default is ``False``.
    freq_bounds : list of float, optional
        Explicit channel-bin bounds [Hz] for the model frequencies, shape
        ``(n_freq + 1,)``, used by the response and the primary beam. This
        decouples the binning from the model frequencies, e.g. fixed bins with
        the model frequencies at their effective (weighted mean) frequency.
        Default is None (resolve's bins: midpoints between model frequencies).
    psf_from_n_inv : bool or 'padded', optional
        Use the inverse-noise kernel as the PSF as well (circular convolution
        on the sky grid, no PSF kernel, ``split`` is ignored): the linear term
        of the fast energy is then the exact one, which the padded PSF
        together with the unpadded noise kernel does not give. ``'padded'``:
        kernel, residual data and noise model on the 2x padded grid (no
        wrap-around within the field; the kernel from the padded RNR).
        Default is ``False``.
    channel_residual : bool, optional
        Compute the residual of a major iteration with the sky and the primary
        beam per data channel (``ChannelResidual``) instead of per model bin.
        Needs several model frequencies.
        Default is ``False``.

    Returns
    -------
    lh_dct : dict
        Likelihood dictionary with keys ``data``, ``sky_model``,
        ``sky_response``, ``noise_model`` and ``RNR`` (and ``sky_app``, the
        beam-attenuated sky, if ``pb_correction`` is set).
    """
    if noise is None:
        noise = {"parameters": {}}
    if split is None:
        split = {}
    if isinstance(data, Observation):
        data = data.to_resolve_obs()
    obs = data.to_double_precision()

    print("sky model shape:", sky.target.shape)

    R, R_l, RNR, RNR_l = build_exact_responses(obs, sky.grid, sky.freq, freq_bounds)

    N_inv = makeOp(obs.weight)
    padded = psf_from_n_inv == "padded"
    data = (R_l if padded else R).adjoint(N_inv(obs.vis)).val
    print("dirty image shape:", data.shape)

    sky_app = sky
    if pb_correction:
        pb_opts = {} if pb_correction is True else dict(pb_correction)
        beam = build_primary_beam(sky.grid, sky.freq, obs, freq_bounds=freq_bounds, **pb_opts)
        sky_app = BeamedSky(sky, beam)
        pb_axes = tuple(range(beam.ndim - 2, beam.ndim))
        print(
            "primary beam:", pb_opts or "defaults",
            "| min PB per frequency:", np.round(np.min(beam, axis=pb_axes), 4).tolist(),
        )

    psf_conv = None
    if not psf_from_n_inv:
        psf_conv = PSFConvolve.build(
            sky=sky_app,
            RNR_l=RNR_l,
            psf_kernel_fn=psf_kernel_fn,
            split=split,
        )

    sky_response = NInvConvolve.build(
        psf_conv=psf_conv,
        RNR=RNR_l if padded else RNR,
        n_inv_kernel_fn=n_inv_kernel_fn,
        noise=noise,
        sky=sky_app,
        padded=padded,
    )

    lh_dct = dict(
        data=jnp.array(data),
        sky_model=sky,
        sky_response=sky_response,
        noise_model=sky_response.noise_model,
        RNR=PaddedRNR(RNR, RNR_l) if padded else RNR,
    )
    if pb_correction:
        lh_dct["sky_app"] = sky_app
    if channel_residual:
        if sky.freq.size < 2:
            raise ValueError("channel_residual needs several model frequencies.")
        lh_dct["channel_residual"] = ChannelResidual(obs, sky.grid, sky.freq, R_l if padded else R, freq_bounds)
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

"""Exact interferometric response builders for fast-resolve."""

import functools

import nifty8 as ift
import numpy as np


def frequency_domain(freq, freq_bounds=None):
    """Sky frequency domain of the interferometric response.

    resolve assigns every data channel to the model frequency whose bin
    contains it. By default (`freq_bounds` None) the bins are those of
    ``resolve.IRGSpace``: midpoints between neighbouring model frequencies,
    outer bounds mirrored. With `freq_bounds` the bin bounds are given
    explicitly, which decouples them from the model frequencies (e.g. fixed,
    equal-width bins with the model frequencies at the effective, weighted
    mean frequency of each bin).

    Parameters
    ----------
    freq : np.ndarray
        Model frequencies, strictly ascending.
    freq_bounds : array-like, optional
        Bin bounds, shape ``(freq.size + 1,)``, strictly ascending, with
        ``freq_bounds[k] <= freq[k] <= freq_bounds[k + 1]``.

    Returns
    -------
    resolve.IRGSpace
        Frequency domain whose ``binbounds()`` define the channel binning.
    """
    import resolve as rve

    if freq_bounds is None:
        return rve.IRGSpace(freq)

    freq = np.asarray(freq, dtype="float64")
    bounds = np.asarray(freq_bounds, dtype="float64").reshape(-1)
    if bounds.size != freq.size + 1:
        raise ValueError(f"freq_bounds needs {freq.size + 1} entries, got {bounds.size}")
    if np.any(np.diff(bounds) <= 0) or np.any(freq < bounds[:-1]) or np.any(freq > bounds[1:]):
        raise ValueError("freq_bounds must be ascending and contain each model frequency in its bin")
    return _bounded_irg_space()(freq, bounds)


@functools.cache
def _bounded_irg_space():
    """``resolve.IRGSpace`` with explicit bin bounds (class created once, on
    first use, so resolve is only imported when needed)."""
    import resolve as rve

    class BoundedIRGSpace(rve.IRGSpace):
        _needed_for_hash = ["_coordinates", "_binbounds"]

        def __init__(self, coordinates, binbounds):
            super().__init__(coordinates)
            self._binbounds = tuple(np.asarray(binbounds, dtype="float64"))

        def __repr__(self):
            return f"BoundedIRGSpace(coordinates={self._coordinates}, binbounds={self._binbounds})"

        def binbounds(self):
            return np.array(self._binbounds)

    return BoundedIRGSpace


def build_exact_responses(
    observation,
    grid,
    freq=None,
    freq_bounds=None,
):
    """Build exact RNR response operators for fast-resolve.

    Constructs both the normal and padded (2x) interferometric response
    operators together with their corresponding RNR products.

    Parameters
    ----------
    observation : Observation
        Radio observation data.
    grid : SignalGrid
        Spatial grid of the sky model.
    freq : np.ndarray, optional
        Frequency array. Default is ``np.ones((1,))``.
    freq_bounds : array-like, optional
        Explicit channel-bin bounds for the model frequencies (see
        `frequency_domain`). Default is None (resolve's midpoint bins).

    Returns
    -------
    R : Operator
        Normal-grid interferometric response.
    R_l : Operator
        Padded (2x) interferometric response.
    RNR : Operator
        Normal-grid RNR product.
    RNR_l : Operator
        Padded RNR product.
    """
    if freq is None:
        freq = np.ones((1,))
    import resolve as rve

    sdom = ift.RGSpace(grid.shape, distances=grid.dis / grid.fac)
    if freq.size > 1:
        inside = (freq >= observation.freq.min()) & (freq <= observation.freq.max())
        if freq_bounds is not None and not np.all(inside):
            raise ValueError("with freq_bounds all model frequencies must lie within the data frequencies")
        freq = freq[inside]
    fdom = frequency_domain(freq, freq_bounds if freq.size > 1 else None)
    sky_dom = rve.default_sky_domain(sdom=sdom, fdom=fdom)
    R = rve.InterferometryResponse(
        observation, sky_dom, True, 1e-9, verbosity=0, nthreads=8
    )

    sdom_l = ift.RGSpace(tuple(2 * s for s in grid.shape), distances=sdom.distances)
    sky_dom_l = rve.default_sky_domain(sdom=sdom_l, fdom=fdom)
    R_l = rve.InterferometryResponse(
        observation, sky_dom_l, True, 1e-9, verbosity=0, nthreads=8
    )

    if freq.size > 1:
        dch_l = ift.DomainChangerAndReshaper(R_l.domain[2:], R_l.domain)
        dch = ift.DomainChangerAndReshaper(R.domain[2:], R.domain)
    else:
        dch_l = ift.DomainChangerAndReshaper(R_l.domain[3], R_l.domain)
        dch = ift.DomainChangerAndReshaper(R.domain[3], R.domain)

    R_l = R_l @ dch_l
    R = R @ dch

    N_inv = ift.DiagonalOperator(observation.weight)
    RNR = R.adjoint @ N_inv @ R
    RNR_l = R_l.adjoint @ N_inv @ R_l

    return R, R_l, RNR, RNR_l


def apply_exact_response(RNR, val):
    """Apply the exact RNR response to a sky array.

    Parameters
    ----------
    RNR : Operator
        RNR operator.
    val : np.ndarray
        Sky value array.

    Returns
    -------
    np.ndarray
        Response-applied array.
    """
    return RNR(ift.makeField(RNR.domain, np.array(val))).val


class PaddedRNR:
    """Exact padded RNR for a sky on the normal grid.

    The sky is zero-padded centrally to the 2x padded grid (the field at
    the centre, as the padded response sees it) and ``RNR_l`` is applied:
    the residual of a padded fast likelihood (``KernelConvolve`` with
    ``padded``). Usable with ``apply_exact_response``.

    Parameters
    ----------
    RNR : Operator
        Normal-grid RNR operator (defines the domain).
    RNR_l : Operator
        Padded RNR operator.
    """

    def __init__(self, RNR, RNR_l):
        self.domain = RNR.domain
        self.target = RNR_l.target
        self.RNR_l = RNR_l

    def __call__(self, x):
        val = np.asarray(x.val)
        pad = [(0, 0)] * (val.ndim - 2) + [(s // 2, s - s // 2) for s in val.shape[-2:]]
        return self.RNR_l(ift.makeField(self.RNR_l.domain, np.pad(val, pad)))


class ChannelResidual:
    """Exact dirty image of the sky evaluated per data channel.

    The model has one image per frequency bin, the data one sky per channel.
    Within a bin the sky is extrapolated from the bin image with the local
    spectral slope of the model cube, ``s_k (nu / nu_k)^a_k``, and multiplied
    by the primary beam of the channel. Used instead of ``RNR`` in the
    residual of a major iteration, the inner fit (bins) stays as it is, but
    its fixed point is the optimum of the per-channel likelihood.

    Parameters
    ----------
    obs : resolve.Observation
        Radio observation (double precision).
    grid : SignalGrid
        Spatial grid of the sky model.
    freq : array-like
        Model frequencies [Hz], ascending.
    freq_bounds : array-like, optional
        Explicit channel-bin bounds (see ``frequency_domain``).
    R_adjoint : Operator or None
        Response whose adjoint gives the dirty image (the binned ``R``); None
        if only per-channel responses are needed.
    nthreads : int, optional
        Threads of the per-channel responses. Default is 16.
    """

    def __init__(self, obs, grid, freq, R_adjoint, freq_bounds=None, nthreads=16):
        self.obs = obs
        self.grid = grid
        self.freq = np.asarray(freq, dtype="float64")
        self.R = R_adjoint
        self.nthreads = nthreads
        bounds = frequency_domain(self.freq, freq_bounds).binbounds()
        self.nu = np.asarray(obs.freq, dtype="float64")
        self.chan_bin = np.clip(np.searchsorted(bounds, self.nu, side="right") - 1, 0, self.freq.size - 1)
        self.N_inv = ift.makeOp(obs.weight)

    def _response(self, c):
        import resolve as rve

        oc = self.obs.get_freqs_by_slice(slice(c, c + 1))
        sdom = ift.RGSpace(self.grid.shape, distances=self.grid.dis / self.grid.fac)
        dom = rve.default_sky_domain(sdom=sdom, fdom=rve.IRGSpace(oc.freq))
        R = rve.InterferometryResponse(oc, dom, True, 1e-9, verbosity=0, nthreads=self.nthreads)
        return R @ ift.DomainChangerAndReshaper(R.domain[3], R.domain)

    def predict(self, sky):
        """Model visibilities (row, channel) of the intrinsic sky cube ``sky``."""
        from .beam import build_primary_beam

        sky = np.asarray(sky, dtype="float64")
        slope = np.gradient(np.log(np.clip(sky, 1e-300, None)), np.log(self.freq), axis=0)
        vis = np.zeros(self.obs.vis.shape[1:], dtype="complex128")
        for c, nu in enumerate(self.nu):
            k = self.chan_bin[c]
            s_c = sky[k] * (nu / self.freq[k]) ** slope[k]
            s_c = s_c * np.asarray(build_primary_beam(self.grid, [nu], channel_average=False), dtype="float64")
            R = self._response(c)
            vis[:, c] = np.asarray(R(ift.makeField(R.domain, s_c)).val).reshape(-1)
        return vis

    def __call__(self, sky):
        """Dirty image ``R^+ N^-1 R(sky per channel)`` on the bins of ``R``."""
        v = ift.makeField(self.R.target, self.predict(sky)[None])
        return np.asarray(self.R.adjoint(self.N_inv(v)).val)

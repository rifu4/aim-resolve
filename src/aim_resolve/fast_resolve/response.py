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

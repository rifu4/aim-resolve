"""Exact interferometric response builders for fast-resolve."""

import nifty8 as ift
import numpy as np



def build_exact_responses(
        observation,
        grid,
        freq = np.ones((1,)),
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
    import resolve as rve

    sdom = ift.RGSpace(grid.shape, distances=grid.dis / grid.fac)
    if freq.size > 1:
        freq = freq[(freq >= observation.freq.min()) & (freq <= observation.freq.max())]
    fdom = rve.IRGSpace(freq)
    sky_dom = rve.default_sky_domain(sdom=sdom, fdom=fdom)
    R = rve.InterferometryResponse(observation, sky_dom, True, 1e-9, verbosity=0, nthreads=8)

    sdom_l = ift.RGSpace(tuple(2*s for s in grid.shape), distances=sdom.distances)
    sky_dom_l = rve.default_sky_domain(sdom=sdom_l, fdom=fdom)
    R_l = rve.InterferometryResponse(observation, sky_dom_l, True, 1e-9, verbosity=0, nthreads=8)

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

    Handles both single and list-of-operator cases by splitting the
    value along the leading axis.

    Parameters
    ----------
    RNR : Operator or list of Operator
        RNR operator(s).
    val : np.ndarray
        Sky value array.

    Returns
    -------
    np.ndarray
        Response-applied array.

    Raises
    ------
    ValueError
        If any operator domain does not have exactly 3 dimensions.
    """
    results, idx = [], 0
    if isinstance(RNR, list):
        for rnr in RNR:
            if len(rnr.domain.shape) != 3:
                raise ValueError("rnr domain must have 3 dimensions.")
            results.append(apply_exact_response(rnr, val[idx : idx+rnr.domain.shape[0]]))
            idx += rnr.domain.shape[0]
        return np.concatenate(results, axis=0)
    
    return RNR(ift.makeField(RNR.domain, np.array(val))).val

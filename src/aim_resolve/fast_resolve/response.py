import nifty8 as ift
import numpy as np



def build_exact_responses(
        observation,
        grid,
        freq = np.ones((1,)),
):
    '''
    Build the exact `RNR` responses for fast-resolve.

    Parameters
    ----------
    observation : rve.Observation
        The radio observation data.
    grid : SignalGrid
        The grid of the sky model.
    freq : np.ndarray
        The frequencies of the sky model.
    ''' 
    import resolve as rve

    sdom = ift.RGSpace(grid.shape, distances=grid.dis / grid.fac)
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

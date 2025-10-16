import os
import pickle
import jax.numpy as jnp
import nifty8 as ift
import numpy as np

from .convolve import fast_fftconvolve, split_fftconvolve



def build_exact_responses(
        observation,
        grid,
):
    '''
    Build the exact `RNR` responses for fast-resolve.

    Parameters
    ----------
    observation : rve.Observation
        The radio observation data.
    grid : SignalGrid
        The grid of the sky model.
    ''' 
    import resolve as rve

    sdom = ift.RGSpace(grid.shape, distances=grid.dis / grid.fac)
    sky_dom = rve.default_sky_domain(sdom=sdom)
    R = rve.InterferometryResponse(observation, sky_dom, True, 1e-9, verbosity=0, nthreads=8)

    sdom_l = ift.RGSpace(tuple(2*s for s in grid.shape), distances=sdom.distances)
    sky_dom_l = rve.default_sky_domain(sdom=sdom_l)
    R_l = rve.InterferometryResponse(observation, sky_dom_l, True, 1e-9, verbosity=0, nthreads=8)

    dch_l = ift.DomainChangerAndReshaper(R_l.domain[3], R_l.domain)
    R_l = R_l @ dch_l
    dch = ift.DomainChangerAndReshaper(R.domain[3], R.domain)
    R = R @ dch

    N_inv = ift.DiagonalOperator(observation.weight)
    RNR = R.adjoint @ N_inv @ R
    RNR_l = R_l.adjoint @ N_inv @ R_l

    return R, R_l, RNR, RNR_l



def build_approximation_kernels(RNR, RNR_l, response_kernel_fn=None, noise_kernel_fn=None, noise_model=None, split=0):
    '''
    Build approximations for response and noise kernel.

    Parameters
    ----------
    RNR : ift.Operator
        The RNR response operator acting on the model space.
    RNR_l : ift.Operator
        The RNR response operator acting on the padded model space.
    response_kernel_fn : str
        The filename to load or save the response kernel. Default is None.
    noise_kernel_fn : str
        The filename to load or save the noise kernel. Default is None.
    noise_model : ift.Operator
        The noise model that should be used for the inference. Default is None.
    '''
    if os.path.isfile(response_kernel_fn):
        response_kernel = pickle.load(open(response_kernel_fn, "rb"))
    else:
        response_kernel = build_response_kernel(RNR_l)
        if response_kernel_fn:
            pickle.dump(response_kernel, open(response_kernel_fn, "wb"))

    if split > 0:
        RNR_approx = split_fftconvolve(response_kernel, RNR.domain.shape, split, RNR.domain[0].scalar_dvol)
    else:
        RNR_approx = fast_fftconvolve(response_kernel, RNR.domain.shape, RNR.domain[0].scalar_dvol)

    # build approximate inverse noise kernel
    if os.path.isfile(noise_kernel_fn):
        noise_kernel = pickle.load(open(noise_kernel_fn, "rb"))
    else:
        noise_kernel = build_noise_kernel(RNR, 1e-3)
        if noise_kernel_fn:
            pickle.dump(noise_kernel, open(noise_kernel_fn, "wb"))

    fft_s = fft_fun(RNR.domain)
    ifft_s = ifft_fun(RNR.domain)
    nk_inv_sqrt = jnp.array(1. / np.sqrt(noise_kernel))

    if noise_model and noise_model.scaling:
        N_inv_approx = lambda x: ifft_s(noise_model(x) * nk_inv_sqrt * fft_s(x['model'])).real
    elif noise_model and noise_model.varcov:
        FFT_s = ift.FFTOperator(RNR.domain)
        fl = ift.full(FFT_s.target, 1.)
        vol = FFT_s(FFT_s.adjoint(fl)).real.mean().val
        fac = np.sqrt(1/vol)
        N_inv_approx = lambda x: fac * nk_inv_sqrt * fft_s(x['model'])
    else:
        N_inv_approx = lambda x: ifft_s(nk_inv_sqrt * fft_s(x['model'])).real

    return RNR_approx, N_inv_approx



def build_response_kernel(RNR_l):
    '''Build the response kernel for a padded RNR operator.'''
    dom_l = RNR_l.domain
    shp_l = dom_l.shape

    delta = np.zeros(shp_l)
    delta[shp_l[0]//2, shp_l[1]//2] = 1 / dom_l.scalar_weight()
    delta = ift.makeField(dom_l, delta)
    kernel = RNR_l(delta)

    return kernel.val



def build_noise_kernel(RNR, relativ_min_val=0.):
    '''Build the inverse noise kernel for the given RNR operator.'''
    dom = RNR.domain
    shp = dom.shape
    FFT = ift.FFTOperator(RNR.domain)

    delta = np.zeros(shp)
    delta[shp[0]//2, shp[1]//2] = 1 / dom.scalar_weight()
    delta = ift.makeField(dom, delta)
    kernel = RNR(delta).val
    kernel = np.roll(kernel, -shp[0]//2, axis=0)
    kernel = np.roll(kernel, -shp[1]//2, axis=1)
    kernel = ift.makeField(RNR.target, kernel)
    FFT = ift.FFTOperator(RNR.domain)
    max_val = np.max(FFT(kernel).abs().val)
    min_val = relativ_min_val * max_val
    min_val = ift.full(FFT.target, min_val)
    min_val_adder = ift.Adder(min_val)

    pos_eig_val = ift.Operator.identity_operator(FFT.target).exp()
    pos_eig_val = min_val_adder @ pos_eig_val
    rls1 = ift.Realizer(pos_eig_val.target)
    rls2 = ift.Realizer(FFT.domain)

    kernel_pos = rls2 @ FFT.inverse @ rls1.adjoint @ pos_eig_val

    cov = ift.ScalingOperator(kernel_pos.target, 1e-2*max_val)
    lh = ift.GaussianEnergy(data=kernel, inverse_covariance=cov.inverse) @ kernel_pos
    init_pos = (FFT(kernel) - min_val).abs().log()
    energy = ift.EnergyAdapter(position=init_pos, op=lh, want_metric=True)

    ic_newton = ift.DeltaEnergyController(name='Newton', iteration_limit=80, tol_rel_deltaE=0)
    #minimizer = ift.NewtonCG(ic_newton, max_cg_iterations=400, energy_reduction_factor=1e-3)
    minimizer = ift.NewtonCG(ic_newton)
    res = minimizer(energy)[0].position

    return pos_eig_val(res).val



def fft_fun(domain):
    '''Compute the FFT for a given NIFTy domain.'''
    if isinstance(domain, ift.DomainTuple):
        domain = domain[0]
    dvol = domain.scalar_dvol
    return lambda x: dvol * jnp.fft.fftn(x)

    

def ifft_fun(domain):
    '''Compute the inverse FFT for a given NIFTy domain.'''
    if isinstance(domain, ift.DomainTuple):
        domain = domain[0]
    if not domain.harmonic:
        domain = ift.get_default_codomain(domain)
    dvol = domain.scalar_dvol
    npix = domain.size
    return lambda x: dvol * npix * jnp.fft.ifftn(x)

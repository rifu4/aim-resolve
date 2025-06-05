import os
import pickle
import jax.numpy as jnp
import nifty8 as ift
import numpy as np
from jax.lax import slice as jax_slice



def build_exact_responses(
        observation,
        space,
        psf_pixels = 3000,
):
    '''
    Build the exact `RNR` responses for fast-resolve.

    Parameters
    ----------
    observation : rve.Observation
        The radio observation data.
    space : SignalSpace
        The space of the sky model.
    psf_pixels : int
        The maximal number of pixels of the PSF kernel.
    ''' 
    import resolve as rve

    sdom = ift.RGSpace(space.shape, distances=space.distances)
    sky_dom = rve.default_sky_domain(sdom=sdom)
    R = rve.InterferometryResponse(observation, sky_dom, True, 1e-9, verbosity=0, nthreads=8)

    full_psf0 = min(2*psf_pixels, sdom.shape[0])
    full_psf1 = min(2*psf_pixels, sdom.shape[1])
    sdom_l = (sdom.shape[0] + full_psf0, sdom.shape[1] + full_psf1)
    sdom_l = ift.RGSpace(sdom_l, distances=sdom.distances)
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



def build_approximation_kernels(RNR, RNR_l, response_kernel_fn=None, noise_kernel_fn=None, noise_model=None):
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
    shp = RNR.domain.shape
    shp_l = RNR_l.domain.shape
    # assert(shp[0] == shp[1])
    # assert(shp_l[0] == shp_l[1])

    # build approximate response kernel
    n_psf_pix0 = (shp_l[0] - shp[0])
    n_psf_pix1 = (shp_l[1] - shp[1])
    n_padding0 = n_psf_pix0 // 2
    n_padding1 = n_psf_pix1 // 2

    if os.path.isfile(response_kernel_fn):
        psf_kernel = pickle.load(open(response_kernel_fn, "rb"))
    else:
        psf_kernel = build_response_kernel(RNR_l, n_psf_pix0, n_psf_pix1)
        if response_kernel_fn:
            pickle.dump(psf_kernel, open(response_kernel_fn, "wb"))

    fft_l = fft_fun(RNR_l.domain)
    ifft_l = ifft_fun(RNR_l.domain)
    psf_kernel = jnp.array(psf_kernel)
    apply_psf_kern = lambda x: ifft_l(psf_kernel * fft_l(x)).real

    slicer = lambda x: jax_slice(
        x, (n_padding0, n_padding1), (n_padding0+ shp[0], n_padding1+ shp[1])
    )
    padder = lambda x: jnp.pad(x, ((n_padding0, n_padding0), (n_padding1, n_padding1)))

    RNR_approx = lambda x: slicer(apply_psf_kern(padder(x)))

    # build approximate inverse noise kernel
    if os.path.isfile(noise_kernel_fn):
        noise_kernel = pickle.load(open(noise_kernel_fn, "rb"))
    else:
        noise_kernel = build_noise_kernel(RNR, 1e-3)
        if noise_kernel_fn:
            pickle.dump(noise_kernel, open(noise_kernel_fn, "wb"))

    fft_s = fft_fun(RNR.domain)
    ifft_s = ifft_fun(RNR.domain)
    noise_kernel_inv_sqrt = 1. / np.sqrt(noise_kernel)
    noise_kernel_inv_sqrt = jnp.array(noise_kernel_inv_sqrt)

    if noise_model and noise_model.scaling:
        N_inv_approx = lambda x: ifft_s(noise_model(x) * noise_kernel_inv_sqrt * fft_s(x['model'])).real
    elif noise_model and noise_model.varcov:
        FFT_s = ift.FFTOperator(RNR.domain)
        fl = ift.full(FFT_s.target, 1.)
        vol = FFT_s(FFT_s.adjoint(fl)).real.mean().val
        fac = np.sqrt(1/vol)
        N_inv_approx = lambda x: fac * noise_kernel_inv_sqrt * fft_s(x['model'])        
    else:
        N_inv_approx = lambda x: ifft_s(noise_kernel_inv_sqrt * fft_s(x['model'])).real

    return RNR_approx, N_inv_approx



def build_response_kernel(RNR_l, n_pix0, n_pix1):
    '''Build the response kernel for the given padded RNR operator.'''
    dom = RNR_l.domain
    shp = dom.shape
    FFT = ift.FFTOperator(RNR_l.domain)

    delta = np.zeros(shp)
    delta[shp[0]//2, shp[1]//2] = 1 / dom.scalar_weight()
    delta = ift.makeField(dom, delta)
    kernel = RNR_l(delta)

    # zero kernel
    sh0 = shp[0]//2
    sh1 = shp[1]//2
    z_kern = np.zeros_like(kernel.val)
    z_kern[sh0-n_pix0:sh0+n_pix0,sh1-n_pix1:sh1+n_pix1] = kernel.val[sh0 - n_pix0:sh0+n_pix0,sh1-n_pix1:sh1+n_pix1]

    pr_kern = np.roll(z_kern, -shp[0]//2, axis=0)
    pr_kern = np.roll(pr_kern, -shp[1]//2, axis=1)
    pr_kern = ift.makeField(FFT.domain, pr_kern)
    fourier_kern = FFT(pr_kern)

    return fourier_kern.val



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

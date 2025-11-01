import nifty8 as ift
import numpy as np



def build_psf_kernel(RNR_l):
    '''Build the psf kernel for a padded RNR operator.'''
    dom_l = RNR_l.domain
    sdom_l = ift.DomainTuple.make(dom_l[-1:])
    shp_l = sdom_l.shape

    delta = np.zeros(shp_l)
    delta[shp_l[0]//2, shp_l[1]//2] = 1 / sdom_l.scalar_weight()
    delta = np.broadcast_to(delta, dom_l.shape)
    delta = ift.makeField(dom_l, delta)
    kernel = RNR_l(delta).val

    return kernel



def build_n_inv_kernel(RNR, relativ_min_val=1e-3):
    '''Build the inverse noise kernel for the given RNR operator.'''
    dom = RNR.domain
    sdom = ift.DomainTuple.make(dom[-1:])
    shp = sdom.shape
    FFT = ift.FFTOperator(sdom)

    delta = np.zeros(shp)
    delta[shp[0]//2, shp[1]//2] = 1 / sdom.scalar_weight()
    delta = np.broadcast_to(delta, dom.shape)
    delta = ift.makeField(dom, delta)
    kernel = RNR(delta).val

    kernel = np.roll(kernel, -shp[0]//2, axis=-2)
    kernel = np.roll(kernel, -shp[1]//2, axis=-1)
    kernel = kernel[None] if len(dom.shape) == 2 else kernel

    n_inv_kernel = np.zeros_like(kernel)
    for i in range(kernel.shape[0]):
        kernel_i = ift.makeField(RNR.target[-1], kernel[i])
    
        max_val = np.max(FFT(kernel_i).abs().val)
        min_val = relativ_min_val * max_val
        min_val = ift.full(FFT.target, min_val)
        min_val_adder = ift.Adder(min_val)

        pos_eig_val = ift.Operator.identity_operator(FFT.target).exp()
        pos_eig_val = min_val_adder @ pos_eig_val
        rls1 = ift.Realizer(pos_eig_val.target)
        rls2 = ift.Realizer(FFT.domain)

        kernel_pos = rls2 @ FFT.inverse @ rls1.adjoint @ pos_eig_val

        cov = ift.ScalingOperator(kernel_pos.target, 1e-2*max_val)
        lh = ift.GaussianEnergy(data=kernel_i, inverse_covariance=cov.inverse) @ kernel_pos
        init_pos = (FFT(kernel_i) - min_val).abs().log()
        energy = ift.EnergyAdapter(position=init_pos, op=lh, want_metric=True)

        ic_newton = ift.DeltaEnergyController(name='Newton', iteration_limit=80, tol_rel_deltaE=0)
        #minimizer = ift.NewtonCG(ic_newton, max_cg_iterations=400, energy_reduction_factor=1e-3)
        minimizer = ift.NewtonCG(ic_newton)
        res = minimizer(energy)[0].position

        n_inv_kernel[i] = pos_eig_val(res).val

    n_inv_kernel = n_inv_kernel[0] if len(dom.shape) == 2 else n_inv_kernel

    return n_inv_kernel

"""Kernel construction for fast-resolve PSF and noise operators."""

import nifty8 as ift
import numpy as np


def build_psf_kernel(RNR_l):
    """Build the PSF convolution kernel from a padded RNR operator.

    Evaluates the response to a delta function placed at the centre of
    the spatial domain.

    Parameters
    ----------
    RNR_l : Operator or list of Operator
        Padded RNR operator(s). When a list is given the kernels are
        concatenated along the leading axis.

    Returns
    -------
    kernel : np.ndarray
        PSF kernel array.

    Raises
    ------
    ValueError
        If any operator domain does not have exactly 3 dimensions.
    """
    if isinstance(RNR_l, list):
        kernels = []
        for rnr_l in RNR_l:
            if len(rnr_l.domain.shape) != 3:
                raise ValueError("rnr_l domain must have 3 dimensions.")
            kernels.append(build_psf_kernel(rnr_l))
        return np.concatenate(kernels, axis=0)

    dom_l = RNR_l.domain
    sdom_l = ift.DomainTuple.make(dom_l[-1:])
    shp_l = sdom_l.shape

    delta = np.zeros(shp_l)
    delta[shp_l[0] // 2, shp_l[1] // 2] = 1 / sdom_l.scalar_weight()
    delta = np.broadcast_to(delta, dom_l.shape)
    delta = ift.makeField(dom_l, delta)
    kernel = RNR_l(delta).val

    return kernel


def build_n_inv_kernel(RNR, relativ_min_val=1e-3, fit="jax"):
    """Build the inverse-noise kernel from the RNR operator.

    Estimates a positive-definite approximation to the RNR diagonal in
    Fourier space using Newton-CG optimisation.

    Parameters
    ----------
    RNR : Operator or list of Operator
        RNR operator(s). When a list is given the kernels are concatenated
        along the leading axis.
    relativ_min_val : float, optional
        Relative floor added to the eigenvalues to ensure positivity.
        Default is 1e-3.
    fit : {'jax', 'nifty8'}, optional
        Implementation of the Newton-CG fit. ``'jax'`` (default) fits all
        frequencies jointly with ``nifty.re`` on the default JAX device (GPU
        if available); ``'nifty8'`` is the original per-frequency NIFTy8 fit.
        The RNR response itself is evaluated with the given operator(s) in
        both cases.

    Returns
    -------
    n_inv_kernel : np.ndarray
        Inverse-noise kernel array.

    Raises
    ------
    ValueError
        If any operator domain does not have exactly 3 dimensions.
    """
    rnrs = RNR if isinstance(RNR, list) else [RNR]
    if isinstance(RNR, list) and any(len(rnr.domain.shape) != 3 for rnr in rnrs):
        raise ValueError("rnr domain must have 3 dimensions.")

    kernel, sdom = zip(*(_rnr_delta_response(rnr) for rnr in rnrs), strict=True)
    if any(s != sdom[0] for s in sdom):
        raise ValueError("all RNR operators must share the same spatial domain.")
    kernel, sdom = np.concatenate(kernel, axis=0), sdom[0]

    if fit == "jax":
        n_inv_kernel = _fit_n_inv_kernel(kernel, sdom.scalar_dvol, relativ_min_val)
    elif fit == "nifty8":
        n_inv_kernel = _fit_n_inv_kernel_nifty8(kernel, sdom, relativ_min_val)
    else:
        raise ValueError(f"fit must be 'jax' or 'nifty8', got {fit!r}.")

    if not isinstance(RNR, list) and len(RNR.domain.shape) == 2:
        n_inv_kernel = n_inv_kernel[0]
    return n_inv_kernel


def _rnr_delta_response(RNR):
    """RNR applied to a delta at the centre, rolled to the origin: (nfreq, *shape)."""
    dom = RNR.domain
    sdom = ift.DomainTuple.make(dom[-1:])
    shp = sdom.shape

    delta = np.zeros(shp)
    delta[shp[0] // 2, shp[1] // 2] = 1 / sdom.scalar_weight()
    delta = np.broadcast_to(delta, dom.shape)
    delta = ift.makeField(dom, delta)
    kernel = RNR(delta).val

    kernel = np.roll(kernel, -shp[0] // 2, axis=-2)
    kernel = np.roll(kernel, -shp[1] // 2, axis=-1)
    kernel = kernel[None] if len(dom.shape) == 2 else kernel
    return kernel, RNR.target[-1]


def _fit_n_inv_kernel(kernel, dvol, relativ_min_val=1e-3, maxiter=80):
    """Positive Fourier-space approximation of the RNR kernels, all frequencies jointly (JAX).

    Same model as `_fit_n_inv_kernel_nifty8`: for every frequency the kernel
    ``k`` is approximated by ``Re(IFFT(exp(s) + m))`` with the floor ``m =
    relativ_min_val * max|FFT(k)|``, by minimising the Gaussian energy
    ``0.5 * sum((Re(IFFT(exp(s) + m)) - k)^2) / (1e-2 * max|FFT(k)|)`` with
    Newton-CG using the Fisher metric, starting from ``s = log|FFT(k) - m|``.
    The frequencies are independent, so the joint energy is the sum of the
    single-frequency energies and has the same minimum. FFT conventions as
    ``nifty8.FFTOperator``: ``FFT(x) = fftn(x) * dvol``,
    ``FFT.inverse(y) = ifftn(y) / dvol``.

    Parameters
    ----------
    kernel : np.ndarray
        RNR delta responses rolled to the origin, shape ``(nfreq, *shape)``.
    dvol : float
        Pixel volume of the spatial domain.
    relativ_min_val : float, optional
        Relative floor of the eigenvalues. Default is 1e-3.
    maxiter : int, optional
        Newton iterations (as the iteration limit of the NIFTy8 fit). Default 80.

    Returns
    -------
    np.ndarray
        Fourier-space eigenvalues ``exp(s) + m``, shape ``(nfreq, *shape)``.
    """
    import jax
    import jax.numpy as jnp
    import nifty.re as jft
    from jax.experimental import enable_x64

    with enable_x64():
        axes = (-2, -1)
        k = jnp.asarray(kernel, dtype=jnp.float64)
        fk = jnp.fft.fftn(k, axes=axes) * dvol
        max_val = jnp.max(jnp.abs(fk), axis=axes, keepdims=True)
        min_val = relativ_min_val * max_val
        inv_cov = 1.0 / (1e-2 * max_val)

        def model(s):
            return jnp.real(jnp.fft.ifftn(jnp.exp(s) + min_val, axes=axes)) / dvol

        @jax.jit
        def energy(s):
            return 0.5 * jnp.sum(inv_cov * (model(s) - k) ** 2)

        @jax.jit
        def metric(s, t):
            _, jt = jax.jvp(model, (s,), (t,))
            return jax.vjp(model, s)[1](inv_cov * jt)[0]

        s0 = jnp.log(jnp.abs(fk - min_val))
        res = jft.newton_cg(energy, s0, hessp=metric, maxiter=maxiter, xtol=0.0)
        return np.asarray(jnp.exp(res) + min_val)


def _fit_n_inv_kernel_nifty8(kernel, sdom, relativ_min_val=1e-3, maxiter=80):
    """Original NIFTy8 fit of `_fit_n_inv_kernel`, one frequency at a time (reference)."""
    FFT = ift.FFTOperator(sdom)
    n_inv_kernel = np.zeros_like(kernel)
    for i in range(kernel.shape[0]):
        kernel_i = ift.makeField(sdom, kernel[i])

        max_val = np.max(FFT(kernel_i).abs().val)
        min_val = relativ_min_val * max_val
        min_val = ift.full(FFT.target, min_val)
        min_val_adder = ift.Adder(min_val)

        pos_eig_val = ift.Operator.identity_operator(FFT.target).exp()
        pos_eig_val = min_val_adder @ pos_eig_val
        rls1 = ift.Realizer(pos_eig_val.target)
        rls2 = ift.Realizer(FFT.domain)

        kernel_pos = rls2 @ FFT.inverse @ rls1.adjoint @ pos_eig_val

        cov = ift.ScalingOperator(kernel_pos.target, 1e-2 * max_val)
        lh = (
            ift.GaussianEnergy(data=kernel_i, inverse_covariance=cov.inverse)
            @ kernel_pos
        )
        init_pos = (FFT(kernel_i) - min_val).abs().log()
        energy = ift.EnergyAdapter(position=init_pos, op=lh, want_metric=True)

        ic_newton = ift.DeltaEnergyController(
            name="Newton", iteration_limit=maxiter, tol_rel_deltaE=0
        )
        # minimizer = ift.NewtonCG(ic_newton, max_cg_iterations=400, energy_reduction_factor=1e-3)
        minimizer = ift.NewtonCG(ic_newton)
        res = minimizer(energy)[0].position

        n_inv_kernel[i] = pos_eig_val(res).val

    return n_inv_kernel

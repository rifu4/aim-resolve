"""Tests for aim_resolve.fast_resolve.kernel — PSF and noise kernel builders."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from aim_resolve.fast_resolve.kernel import build_n_inv_kernel, build_psf_kernel


class TestBuildPsfKernel:
    """Tests for build_psf_kernel dispatch and validation."""

    def test_list_with_wrong_dims_raises(self):
        op = MagicMock()
        op.domain.shape = (16, 16)  # 2 dims instead of 3
        with pytest.raises(ValueError, match="3 dimensions"):
            build_psf_kernel([op])

    def test_list_concatenates(self):
        """Two operators should produce a concatenated kernel."""
        import nifty8 as ift

        sdom = ift.RGSpace((8, 8))
        fdom = ift.UnstructuredDomain(1)
        dom = ift.DomainTuple.make((fdom, sdom))
        op = ift.ScalingOperator(dom, 1.0)

        kernel = build_psf_kernel([op, op])
        assert kernel.shape[0] == 2  # two concatenated
        assert kernel.shape[1:] == (8, 8)

    def test_single_operator_returns_array(self):
        import nifty8 as ift

        sdom = ift.RGSpace((8, 8))
        fdom = ift.UnstructuredDomain(1)
        dom = ift.DomainTuple.make((fdom, sdom))
        op = ift.ScalingOperator(dom, 1.0)

        kernel = build_psf_kernel(op)
        assert isinstance(kernel, np.ndarray)
        assert kernel.shape == (1, 8, 8)


class TestBuildNInvKernel:
    """Tests for build_n_inv_kernel dispatch and validation."""

    def test_list_with_wrong_dims_raises(self):
        op = MagicMock()
        op.domain.shape = (16, 16)  # 2 dims instead of 3
        with pytest.raises(ValueError, match="3 dimensions"):
            build_n_inv_kernel([op])


def _small_rnr(n_freq=2, n_row=400, shape=(32, 32), seed=0):
    """RNR of a small random observation (w-gridding, as build_exact_responses)."""
    import nifty8 as ift
    import resolve as rve

    rng = np.random.default_rng(seed)
    uvw = rng.uniform(-300, 300, (n_row, 3))
    uvw[:, 2] *= 0.01
    freq = np.linspace(1.0e9, 1.4e9, 2 * n_freq)
    obs = rve.Observation(rve.AntennaPositions(uvw), np.ones((1, n_row, freq.size), np.complex128),
                          rng.uniform(0.5, 2, (1, n_row, freq.size)), rve.Polarization.trivial(), freq)
    sdom = ift.RGSpace(shape, distances=2e-4)
    fdom = rve.IRGSpace(0.5 * (freq[::2] + freq[1::2])) if n_freq > 1 else rve.IRGSpace([1.0])
    R = rve.InterferometryResponse(obs, rve.default_sky_domain(sdom=sdom, fdom=fdom), True, 1e-9)
    R = R @ ift.DomainChangerAndReshaper(R.domain[2:] if n_freq > 1 else R.domain[3], R.domain)
    return R.adjoint @ ift.makeOp(obs.weight) @ R


class TestNInvKernelFit:
    """The JAX fit of the noise kernel against the original NIFTy8 fit."""

    def test_same_minimum_as_nifty8(self):
        from aim_resolve.fast_resolve.kernel import (
            _fit_n_inv_kernel,
            _fit_n_inv_kernel_nifty8,
            _rnr_delta_response,
        )

        kernel, sdom = _rnr_delta_response(_small_rnr())
        # both converge to the same minimum (at 80 iterations neither is fully converged)
        nk_jax = _fit_n_inv_kernel(kernel, sdom.scalar_dvol, maxiter=300)
        nk_ift = _fit_n_inv_kernel_nifty8(kernel, sdom, maxiter=300)
        assert nk_jax.shape == nk_ift.shape == kernel.shape
        assert np.all(nk_jax > 0)
        assert np.linalg.norm(nk_jax - nk_ift) / np.linalg.norm(nk_ift) < 1e-3

    def test_energy_as_nifty8(self):
        from aim_resolve.fast_resolve.kernel import (
            _fit_n_inv_kernel,
            _fit_n_inv_kernel_nifty8,
            _rnr_delta_response,
        )

        kernel, sdom = _rnr_delta_response(_small_rnr())
        dv = sdom.scalar_dvol
        max_val = np.abs(np.fft.fftn(kernel, axes=(-2, -1)) * dv).max(axis=(-2, -1), keepdims=True)

        def energy(nk):
            approx = np.real(np.fft.ifftn(nk, axes=(-2, -1))) / dv
            return np.sum(0.5 * (approx - kernel) ** 2 / (1e-2 * max_val))

        # default 80 iterations: neither fit is fully converged, both energies agree to 0.1 %
        e_jax, e_ift = energy(_fit_n_inv_kernel(kernel, dv)), energy(_fit_n_inv_kernel_nifty8(kernel, sdom))
        assert abs(e_jax - e_ift) / e_ift < 1e-3

    def test_shapes(self):
        rnr = _small_rnr()
        assert build_n_inv_kernel(rnr).shape == (2, 32, 32)
        assert build_n_inv_kernel([rnr, rnr]).shape == (4, 32, 32)
        assert build_n_inv_kernel(_small_rnr(n_freq=1)).shape == (32, 32)

    def test_invalid_fit_raises(self):
        with pytest.raises(ValueError, match="fit must be"):
            build_n_inv_kernel(_small_rnr(), fit="scipy")

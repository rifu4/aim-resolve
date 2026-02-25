"""Tests for aim_resolve.fast_resolve.kernel — PSF and noise kernel builders."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from aim_resolve.fast_resolve.kernel import build_psf_kernel, build_n_inv_kernel


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

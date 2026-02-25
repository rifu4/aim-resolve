"""Tests for aim_resolve.fast_resolve.response — exact response utilities."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from aim_resolve.fast_resolve.response import apply_exact_response


class TestApplyExactResponse:
    """Tests for apply_exact_response dispatch and error handling."""

    def _make_mock_rnr(self, shape):
        """Create a mock RNR operator that returns its input unchanged."""
        import nifty8 as ift

        sdom = ift.RGSpace(shape[-2:])
        fdom = ift.UnstructuredDomain(shape[0]) if len(shape) == 3 else None
        if fdom is not None:
            dom = ift.DomainTuple.make((fdom, sdom))
        else:
            dom = ift.DomainTuple.make(sdom)

        op = ift.ScalingOperator(dom, 1.0)
        return op

    def test_single_operator(self):
        rnr = self._make_mock_rnr((1, 8, 8))
        val = np.ones((1, 8, 8))
        result = apply_exact_response(rnr, val)
        assert result.shape == (1, 8, 8)

    def test_list_of_operators(self):
        rnr1 = self._make_mock_rnr((1, 8, 8))
        rnr2 = self._make_mock_rnr((1, 8, 8))
        val = np.ones((2, 8, 8))
        result = apply_exact_response([rnr1, rnr2], val)
        assert result.shape == (2, 8, 8)

    def test_list_wrong_dims_raises(self):
        op = MagicMock()
        op.domain.shape = (8, 8)  # 2 dims instead of 3
        with pytest.raises(ValueError, match="3 dimensions"):
            apply_exact_response([op], np.ones((8, 8)))

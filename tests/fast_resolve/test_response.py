"""Tests for aim_resolve.fast_resolve.response — exact response utilities."""

from unittest.mock import MagicMock

import numpy as np
import pytest

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

    def test_invalid_operator_raises(self):
        op = MagicMock()
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_exact_response(op, np.ones((8, 8)))

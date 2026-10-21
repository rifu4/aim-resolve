"""Tests for aim_resolve.fast_resolve.response — exact response utilities."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from aim_resolve.fast_resolve.response import apply_exact_response, frequency_domain


class TestFrequencyDomain:
    """Tests for the channel-bin definition of the sky frequency domain."""

    FREQ = np.array([1.0e9, 1.2e9, 1.4e9])

    def test_default_is_resolve_midpoints(self):
        np.testing.assert_allclose(frequency_domain(self.FREQ).binbounds(), [0.9e9, 1.1e9, 1.3e9, 1.5e9])

    def test_explicit_bounds(self):
        bounds = [0.95e9, 1.05e9, 1.25e9, 1.45e9]
        fdom = frequency_domain(self.FREQ, bounds)
        np.testing.assert_allclose(fdom.binbounds(), bounds)
        np.testing.assert_allclose(fdom.coordinates, self.FREQ)
        np.testing.assert_allclose(fdom.dvol, np.diff(bounds))

    def test_hash_depends_on_bounds(self):
        a = frequency_domain(self.FREQ, [0.95e9, 1.05e9, 1.25e9, 1.45e9])
        b = frequency_domain(self.FREQ, [0.95e9, 1.15e9, 1.25e9, 1.45e9])
        assert a != b
        assert a == frequency_domain(self.FREQ, [0.95e9, 1.05e9, 1.25e9, 1.45e9])

    @pytest.mark.parametrize(
        "bounds",
        [
            [0.95e9, 1.05e9, 1.25e9],  # wrong size
            [0.95e9, 1.25e9, 1.05e9, 1.45e9],  # not ascending
            [0.95e9, 1.25e9, 1.3e9, 1.45e9],  # 1.2 GHz outside its bin
        ],
    )
    def test_invalid_bounds_raise(self, bounds):
        with pytest.raises(ValueError):
            frequency_domain(self.FREQ, bounds)

    def test_response_channel_assignment(self):
        """resolve's response assigns the channels by the explicit bounds."""
        import resolve as rve

        n_row, ch = 4, np.array([0.97e9, 1.09e9, 1.21e9, 1.42e9])
        rng = np.random.default_rng(0)
        uvw = rng.uniform(-100, 100, (n_row, 3))
        ant = rve.AntennaPositions(uvw)
        vis = np.ones((1, n_row, ch.size), dtype=np.complex64)
        pol = rve.Polarization.trivial()
        obs = rve.Observation(ant, vis, np.ones(vis.shape, np.float32), pol, ch)
        import nifty8 as ift

        sdom = ift.RGSpace((16, 16), distances=1e-4)
        for bounds, expected in [(None, [0, 0, 1, 2]), ([0.95e9, 1.05e9, 1.25e9, 1.45e9], [0, 1, 1, 2])]:
            fdom = frequency_domain(self.FREQ, bounds)
            R = rve.InterferometryResponse(obs, rve.default_sky_domain(sdom=sdom, fdom=fdom), True, 1e-5)
            got = np.empty(ch.size, int)
            for k in range(self.FREQ.size):
                sky = np.zeros(R.domain.shape)
                sky[0, 0, k, 8, 8] = 1.0  # point source at the centre, only in model frequency k
                hit = np.abs(R(ift.makeField(R.domain, sky)).val[0, 0]) > 0
                got[hit] = k
            assert got.tolist() == expected


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

"""Tests for aim_resolve.model.grid — SignalGrid and PointGrid."""

import numpy as np
import pytest

from aim_resolve.model.grid import SignalGrid, PointGrid


# ---------- SignalGrid ----------

class TestSignalGridBuild:
    def test_default_build(self):
        g = SignalGrid.build(space=(64, 64))
        assert g.space == (64, 64)
        assert g.shape == (64, 64)
        assert g.factor == 1
        assert g.n_copies == 1

    def test_factor(self):
        g = SignalGrid.build(space=(32, 32), factor=2)
        assert g.shape == (64, 64)
        assert g.factor == 2

    def test_distances_from_fov(self):
        g = SignalGrid.build(space=(10, 10), fov=(1.0, 2.0))
        np.testing.assert_allclose(g.distances, (0.1, 0.2))

    def test_default_distances(self):
        g = SignalGrid.build(space=(8, 8))
        np.testing.assert_allclose(g.distances, (1/8, 1/8))

    def test_n_copies(self):
        g = SignalGrid.build(space=(16, 16), n_copies=3, center=[(0, 0), (1, 1), (2, 2)])
        assert g.n_copies == 3


class TestSignalGridProperties:
    def test_spc(self):
        g = SignalGrid.build(space=(16, 32))
        np.testing.assert_array_equal(g.spc, [16, 32])

    def test_shp(self):
        g = SignalGrid.build(space=(16, 32), factor=2)
        np.testing.assert_array_equal(g.shp, [32, 64])

    def test_fov(self):
        g = SignalGrid.build(space=(10, 10), distances=(0.5, 0.5))
        np.testing.assert_allclose(g.fov, [5.0, 5.0])

    def test_ndim(self):
        g = SignalGrid.build(space=(8, 8))
        assert g.ndim == 2

    def test_size(self):
        g = SignalGrid.build(space=(4, 8))
        assert g.size == 32

    def test_dvol(self):
        g = SignalGrid.build(space=(10, 10), distances=(2.0, 2.0))
        assert g.dvol == 4.0


class TestSignalGridEquality:
    def test_equal(self):
        a = SignalGrid.build(space=(16, 16))
        b = SignalGrid.build(space=(16, 16))
        assert a == b

    def test_not_equal_shape(self):
        a = SignalGrid.build(space=(16, 16))
        b = SignalGrid.build(space=(32, 32))
        assert not (a == b)


class TestSignalGridContainment:
    def test_contains(self):
        big = SignalGrid.build(space=(64, 64))
        small = SignalGrid.build(space=(32, 32))
        assert small in big

    def test_not_contains(self):
        small = SignalGrid.build(space=(32, 32))
        big = SignalGrid.build(space=(64, 64))
        assert not (big in small)


class TestSignalGridRefine:
    def test_refine(self):
        g = SignalGrid.build(space=(16, 16))
        r = g.refine(2)
        assert r.factor == 2
        assert r.shape == (32, 32)
        assert r.space == (16, 16)


class TestSignalGridMultiply:
    def test_multiply(self):
        g = SignalGrid.build(space=(16, 16))
        r = g * 2
        assert r.space == (32, 32)

    def test_rmultiply(self):
        g = SignalGrid.build(space=(16, 16))
        r = 2 * g
        assert r.space == (32, 32)

    def test_divide(self):
        g = SignalGrid.build(space=(16, 16))
        r = g / 2
        assert r.space == (8, 8)


class TestSignalGridToDict:
    def test_roundtrip(self):
        g = SignalGrid.build(space=(16, 16), factor=2)
        d = g.to_dict()
        g2 = SignalGrid.build(**d)
        assert g == g2


class TestSignalGridRepr:
    def test_repr(self):
        g = SignalGrid.build(space=(8, 8))
        r = repr(g)
        assert "SignalGrid" in r
        assert "(8, 8)" in r


# ---------- PointGrid ----------

class TestPointGridBuild:
    def test_single_point(self):
        pg = PointGrid.build(coordinates=(0.5, 0.5))
        assert pg.n_copies == 1
        assert pg.factor == 1
        assert pg.shape == (1, 1)

    def test_multi_point(self):
        pg = PointGrid.build(coordinates=[(0.5, 0.5), (0.5, 0.5)], n_copies=2)
        assert pg.n_copies == 2

    def test_invalid_coordinates_raises(self):
        with pytest.raises(ValueError):
            PointGrid.build(coordinates=(0.3, 0.3), factor=2)


class TestPointGridProperties:
    def test_coos(self):
        pg = PointGrid.build(coordinates=(0.5, 0.5))
        np.testing.assert_array_equal(pg.coos, [0.5, 0.5])

    def test_ndim(self):
        pg = PointGrid.build(coordinates=(0.5, 0.5))
        assert pg.ndim == 2

    def test_repr(self):
        pg = PointGrid.build(coordinates=(0.5, 0.5))
        assert "PointGrid" in repr(pg)

    def test_to_dict(self):
        pg = PointGrid.build(coordinates=(0.5, 0.5))
        d = pg.to_dict()
        assert "coordinates" in d

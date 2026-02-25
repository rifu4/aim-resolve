"""Tests for aim_resolve.resolve.constants — physical constants and str2rad."""

import numpy as np
import pytest

from aim_resolve.resolve.constants import (
    ARCMIN2RAD,
    AS2RAD,
    DEG2RAD,
    SPEEDOFLIGHT,
    str2rad,
)


class TestPhysicalConstants:
    """Sanity-check the constant values."""

    def test_deg2rad(self):
        assert DEG2RAD == pytest.approx(np.pi / 180)

    def test_arcmin2rad(self):
        assert ARCMIN2RAD == pytest.approx(np.pi / 60 / 180)

    def test_as2rad(self):
        assert AS2RAD == pytest.approx(np.pi / 3600 / 180)

    def test_speed_of_light(self):
        assert SPEEDOFLIGHT == pytest.approx(299792458.0)


class TestStr2Rad:
    """Test the str2rad unit-conversion helper."""

    def test_deg(self):
        assert str2rad("180deg") == pytest.approx(np.pi)

    def test_rad(self):
        assert str2rad("3.14rad") == pytest.approx(3.14)

    def test_amin(self):
        assert str2rad("60amin") == pytest.approx(DEG2RAD)

    def test_as(self):
        assert str2rad("3600as") == pytest.approx(DEG2RAD)

    def test_mas(self):
        assert str2rad("1000mas") == pytest.approx(AS2RAD)

    def test_muas(self):
        assert str2rad("1000000muas") == pytest.approx(AS2RAD)

    def test_plain_float(self):
        assert str2rad("1.5") == pytest.approx(1.5)

    def test_unknown_unit_raises(self):
        with pytest.raises(RuntimeError, match="Unit not understood"):
            str2rad("10parsec")

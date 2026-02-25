"""Physical and unit conversion constants for radio astronomy."""

import numpy as np

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
SPEEDOFLIGHT = 299792458.0


def str2rad(s):
    """Convert string of number and unit to radian.

    Supports the following units: muas, mas, as, amin, deg, rad.
    If no unit is found, attempts to parse the string as a plain float.

    Parameters
    ----------
    s : str
        String of number and unit, e.g. ``'1.5deg'`` or ``'300mas'``.

    Returns
    -------
    float
        The value converted to radians.

    Raises
    ------
    RuntimeError
        If the unit suffix is not recognised and the string cannot be
        converted to a float.
    """
    c = {
        "muas": AS2RAD * 1e-6,
        "mas": AS2RAD * 1e-3,
        "as": AS2RAD,
        "amin": ARCMIN2RAD,
        "deg": DEG2RAD,
        "rad": 1,
    }
    for k in c:
        if s.endswith(k):
            return float(s[: -len(k)]) * c[k]
    try:
        return float(s)
    except ValueError as err:
        raise RuntimeError("Unit not understood") from err

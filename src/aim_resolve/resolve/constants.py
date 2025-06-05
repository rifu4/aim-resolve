import numpy as np



ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
SPEEDOFLIGHT = 299792458.



def str2rad(s):
    '''
    Convert string of number and unit to radian supporting the following units: muas, mas, as, amin, deg, rad.
    Requires the value in Radian.

    Parameters
    ----------
    s : str
        String of number and unit.
    '''
    c = {
        'muas': AS2RAD * 1e-6,
        'mas': AS2RAD * 1e-3,
        'as': AS2RAD,
        'amin': ARCMIN2RAD,
        'deg': DEG2RAD,
        'rad': 1,
    }
    for k in c.keys():
        if s.endswith(k):
            return float(s[:-len(k)]) * c[k] 
    try:
        return float(s)
    except:
        raise RuntimeError('Unit not understood')

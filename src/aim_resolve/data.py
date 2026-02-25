"""Data loading utilities for image and radio observations."""

from .img_data.data import ImageData, ImageDataGenerator
from .resolve.observation import Observation



def data_func(
        mode,
        **kwargs,
):
    """Load observation data using the mode-specific loader.

    Parameters
    ----------
    mode : str
        Data mode. Supported values are ``'image'`` and ``'radio'``.
    **kwargs
        Additional keyword arguments forwarded to the selected loader.

    Returns
    -------
    data : ImageData or Observation
        The loaded observation data.

    Raises
    ------
    TypeError
        If *mode* is not recognised.
    """
    if 'image' in mode:
        return image_data(**kwargs)
    elif 'radio' in mode:
        return radio_data(**kwargs)
    else:
        raise TypeError(f'Unknown data mode. Available modes are `image` and `radio`, but got mode `{mode}`.')



def image_data(*,
        fname,
        odir = '',
        idx = None,
        key = 42,
        max_std = 0.001,
):
    """Load image data from a file and add synthetic noise.

    Tries ``ImageDataGenerator`` first and falls back to ``ImageData``.

    Parameters
    ----------
    fname : str
        Path to the image data file.
    odir : str, optional
        Output directory prefix for the file. Default is ``''``.
    idx : int or None, optional
        Index of the sample to extract (used with ``ImageDataGenerator``).
        Default is None.
    key : int, optional
        Random seed for noise generation. Default is 42.
    max_std : float, optional
        Maximum standard deviation of the added noise. Default is 0.001.

    Returns
    -------
    data : ImageData
        The loaded image data with noise added.
    """
    try:
        img_data = ImageDataGenerator.load(fname, odir, dtype='float64')
        data = img_data.get_sample(idx)
    except:
        data = ImageData.load(fname, odir, dtype='float64')

    data.add_noise(key, max_std)

    return data



def radio_data(*, 
        fname,
        freq = None,
        nrow = None,
        prec = 'double',
):
    """Load a radio observation from a measurement set.

    The data are Stokes-I averaged and optionally sub-sampled in frequency
    or row count.

    Parameters
    ----------
    fname : str
        Path to the measurement set file.
    freq : list or None, optional
        Subset of frequencies to keep. Default is None (all frequencies).
    nrow : int, float or None, optional
        Number (or fraction) of rows to keep. Default is None (all rows).
    prec : {'single', 'double'}, optional
        Floating-point precision of the loaded data. Default is ``'double'``.

    Returns
    -------
    obs : Observation
        The loaded and pre-processed radio observation.

    Raises
    ------
    TypeError
        If *freq* is not a list or *nrow* is not int/float.
    ValueError
        If *prec* is not ``'single'`` or ``'double'``.
    """
    obs = Observation.load(fname)

    obs = obs.average_stokesi()
    obs = obs.to_double_precision()

    if freq:
        if not isinstance(freq, list):
            raise TypeError('`freq` has to be of Type `list`')
        obs = obs.get_freqs(freq)

    if nrow:
        if not isinstance(nrow, (int, float)):
            raise TypeError('`nvis` has to be of Type `int` or `float`')
        obs = obs.subsample_rows(nrow)

    match prec:
        case 'single':
            obs = obs.to_single_precision()
        case 'double':
            obs = obs.to_double_precision()
        case _:
            raise ValueError('`precision` has to be either `single` or `double`')
    
    return obs

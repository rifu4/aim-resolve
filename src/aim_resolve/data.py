from .img_data.data import ImageData, ImageDataGenerator
from .resolve.observation import Observation



def image_data(*,
        fname,
        odir = '',
        idx = None,
        key = 42,
        max_std = 0.001,
        fun = 'exp',
):
    '''
    Load image data from a file and add noise to it.
    Uses either the ImageData or ImageDataGenerator class to load the data.
    
    Parameters
    ----------
    fname : str
        The name of the file to load the data from.
    odir : str, optional
        The output directory for the file, by default
    idx : int, optional
        The index of the image to load, by default None
    key : int, optional
        The random seed to use for generating noise, by default 42
    max_std : float, optional
        The maximum standard deviation of the noise to add, by default 0.001
    fun : str, optional
        Used to differentiate between the different data functions.
    '''
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
        fun = 'radio',
):
    '''
    Load a radio observation from a file. Uses the Observation class to load the data.

    Parameters
    ----------
    fname : str
        The name of the file to load the data from.
    freq : list, optional
        Use only a subset of frequencies, by default None
    nrow : int or float, optional
        Use only a subset of rows, by default None
    prec : str, optional
        The precision of the data, by default 'double'
    fun : str, optional
        Used to differentiate between the different data functions.
    '''
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

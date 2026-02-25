"""Utility functions for the model subpackage."""

from collections.abc import Iterable

import numpy as np


def check_type(value, *types, uppers=()):
    """Check if the value is of the given type(s).

    Parameters
    ----------
    value : any
        The value to check.
    types : tuple
        The types to check against. Each element of the tuple can be a
        type or an iterable of types. If the value itself is an iterable,
        the first type checks the iterable and subsequent types check its
        elements recursively.
    uppers : tuple
        Do not use this parameter directly. It is used to produce more
        informative error messages.
    """
    if not isinstance(value, types[0]):
        err = f"`{value}`"
        for up in uppers[::-1]:
            err += f" in `{up}`"
        raise TypeError(f"{err} has to be of type `{types[0]}`")
    if isinstance(value, Iterable) and len(types) > 1:
        for v in value:
            check_type(v, *types[1:], uppers=uppers + (value,))


def flatten_list(lst):
    """Flatten nested iterables to a single list.

    Parameters
    ----------
    lst : iterable
        A possibly nested iterable to flatten.

    Returns
    -------
    list
        A flat list of all leaf elements.
    """
    new_lst = []
    for val in lst:
        if isinstance(val, Iterable) and not isinstance(val, str):
            new_lst += flatten_list(val)
        else:
            new_lst += [
                val,
            ]
    return new_lst


def to_shape(array, shape, dtype="float64"):
    """Convert the input to an array with the given shape.

    Parameters
    ----------
    array : scalar, str, or iterable
        Input value(s) to convert.
    shape : tuple of int
        Desired output shape.
    dtype : str, optional
        Data type of the output array, by default ``'float64'``.

    Returns
    -------
    np.ndarray
        Array reshaped or broadcast to *shape* with the given *dtype*.
    """
    from ..resolve.constants import str2rad

    lst = (
        array
        if isinstance(array, Iterable) and not isinstance(array, str)
        else [
            array,
        ]
    )
    lst = flatten_list(lst)
    lst = [str2rad(li) if isinstance(li, str) else li for li in lst]
    array = np.array(lst)

    if array.size == np.prod(shape):
        res = np.reshape(array, shape)
    else:
        res = np.broadcast_to(array, shape)

    return res.astype(dtype)


def is_val(array):
    """Check if the array contains any non-zero values.

    Parameters
    ----------
    array : np.ndarray
        Input array (may contain NaNs).

    Returns
    -------
    bool
        True if any non-NaN element is non-zero.
    """
    if np.any(array[~np.isnan(array)] != 0):
        return True
    else:
        return False


def extend_shape(n_copies, freq, shape, *, offset=False):
    """Expand a shape depending on number of copies and frequencies.

    Parameters
    ----------
    n_copies : int
        Number of tile copies.
    freq : array-like
        Frequency array.
    shape : tuple of int
        Base spatial shape.
    offset : bool, optional
        If True, add a length-1 frequency axis instead of the full
        frequency dimension, by default False.

    Returns
    -------
    tuple of int
        Extended shape.
    """
    if len(freq) > 1 and offset:
        shape = (1,) + shape
    elif len(freq) > 1:
        shape = (len(freq),) + shape
    if n_copies > 1:
        shape = (n_copies,) + shape
    return shape

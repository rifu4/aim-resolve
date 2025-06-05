import numpy as np
from collections.abc import Iterable



def check_type(value, *types, uppers=()):
    '''
    Check if the value is of the given type(s).

    Parameters
    ----------
    value : any
        The value to check.
    types : tuple
        The types to check against. Each element of the tuple can be a type or an iterable of types.
        -> if the value itself is an iterable, the first type in the tuple is used to check the type of the iterable itself.
        -> the second type in the tuple is used to check the type of the elements of the iterable. And so on.
    uppers : tuple
        Do not use this parameter directly. It is used to produce more informative error messages.
    '''
    if not isinstance(value, types[0]):
        err = f'`{value}`'
        for up in uppers[::-1]:
            err += f' in `{up}`'
        raise TypeError(f'{err} has to be of type `{types[0]}`')
    if isinstance(value, Iterable) and len(types) > 1:
        for v in value:
            check_type(v, *types[1:], uppers=uppers+(value,))



def flatten_list(lst):
    '''flatten nested iterables to a single list'''
    new_lst = []
    for val in lst:
        if isinstance(val, Iterable) and not isinstance(val, str):
            new_lst += flatten_list(val)
        else:
            new_lst += [val, ]
    return new_lst



def to_shape(array, shape, dtype='float64'):
    '''convert the input to an array with the given shape'''
    from ..resolve.constants import str2rad

    lst = array if isinstance(array, Iterable) and not isinstance(array, str) else [array, ]
    lst = flatten_list(lst)
    lst = [str2rad(li) if isinstance(li, str) else li for li in lst]
    array = np.array(lst) 

    if array.size == np.prod(shape):
        res = np.reshape(array, shape)
    else:
        res = np.broadcast_to(array, shape)

    return res.astype(dtype)
    


def is_val(array):
    '''check if the array contains any non-zero values'''
    if np.any(array[~np.isnan(array)] != 0):
        return True
    else:
        return False

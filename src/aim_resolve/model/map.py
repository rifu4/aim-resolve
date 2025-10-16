import jax.numpy as jnp
from jax import vmap
from jax.lax import dynamic_slice, dynamic_update_slice
from functools import partial

from .grid import SignalGrid, PointGrid
from .util import check_type



def downsample(array, factor):
    return array.reshape(array.shape[0]//factor, factor, array.shape[1]//factor, factor).mean(axis=(1,3))
    

def upsample(array, factor):
    return array.repeat(factor, axis=0).repeat(factor, axis=1)



def map_signal(in_grid, out_grid, sum_up=True):
    '''
    Create a function to map a signal from one grid to another.

    Parameters
    ----------
    in_grid : SignalGrid
        The input signal grid.
    out_grid : SignalGrid
        The output signal grid.
    sum_up : bool, optional
        Whether to sum up the input copies when mapping to a single output copy, by default True
    '''
    check_type(in_grid, SignalGrid)
    check_type(out_grid, SignalGrid)

    in_shape, out_shape = signal_slice_shapes(in_grid, out_grid)
    in_start, out_start = signal_slice_indices(in_grid, out_grid)
    sum_func = partial(jnp.sum, axis=0) if sum_up else lambda x: x

    def map_2d(x, in_start, out_start):
        if out_grid.fac < in_grid.fac:
            x = downsample(x, in_grid.fac // out_grid.fac)
        x = dynamic_slice(x, in_start, in_shape)
        x = dynamic_update_slice(jnp.zeros(out_shape, dtype=x.dtype), x, out_start)
        if out_grid.fac > in_grid.fac:
            x = upsample(x, out_grid.fac // in_grid.fac)
        return x
    
    def map_fun(x):
        if x.ndim == 2:
            if out_grid.n_copies == 1:
                return map_2d(x, in_start, out_start)
            else:
                return vmap(map_2d, in_axes=(None, 0, 0))(x, in_start, out_start)
        else:
            if in_grid.n_copies == 1:
                return sum_func(vmap(map_2d, in_axes=(0, None, None))(x, in_start, out_start))
            else:
                return sum_func(vmap(map_2d, in_axes=(0, 0, 0))(x, in_start, out_start))
    
    return map_fun


def signal_slice_shapes(in_grid, out_grid):
    fac = min(in_grid.fac, out_grid.fac)
    out_shape = tuple((out_grid.spc * fac).tolist())
    in_shape = tuple((in_grid.spc * fac).tolist())
    if in_grid in out_grid:
        in_shape = tuple((in_grid.spc * fac).tolist())
    elif out_grid in in_grid:
        in_shape = out_shape
    else:
        raise ValueError('One grid must contain the other to map signals.')
    return in_shape, out_shape


def signal_slice_indices(in_grid, out_grid):
    fac = min(in_grid.fac, out_grid.fac)

    def indices(in_spc, in_cen, out_spc, out_cen, fac):
        dif = (out_cen - in_cen) - (out_spc - in_spc) / 2
        dif = dif.astype('int64')
        in_start = jnp.maximum(dif, 0) * fac
        out_start = jnp.maximum(-dif, 0) * fac
        return in_start, out_start
    
    if in_grid.n_copies == 1:
        return indices(in_grid.spc, in_grid.cen, out_grid.spc, out_grid.cen, fac)
    else:
        return vmap(indices, in_axes=(None, 0, None, None, None))(in_grid.spc, in_grid.cen, out_grid.spc, out_grid.cen, fac)



def map_points(in_grid, out_grid, sum_up=True):
    '''
    Create a function to map points to a signal grid.

    Parameters
    ----------
    in_grid : PointGrid
        The input point grid.
    out_grid : SignalGrid
        The output signal grid.
    sum_up : bool, optional
        Whether to sum up the input copies when mapping to a single output copy, by default True
    '''
    check_type(in_grid, PointGrid)
    check_type(out_grid, SignalGrid)

    out_shape = point_slice_shapes(in_grid, out_grid)
    out_start = point_slice_indices(in_grid, out_grid)
    sum_func = partial(jnp.sum, axis=0) if sum_up else lambda x: x

    def map_2d(x, out_start):
        if out_grid.fac < in_grid.fac:
            x /= (in_grid.fac // out_grid.fac)**2
        x = dynamic_update_slice(jnp.zeros(out_shape, dtype=x.dtype), x, out_start)
        if out_grid.fac > in_grid.fac:
            x = upsample(x, out_grid.fac // in_grid.fac)
        return x
    
    def map_fun(x):
        if x.ndim == 2:
            if out_grid.n_copies == 1:
                return map_2d(x, out_start)
            else:
                return vmap(map_2d, in_axes=(None, 0))(x, out_start)
        else:
            if in_grid.n_copies == 1:
                return sum_func(vmap(map_2d, in_axes=(0, None))(x, out_start))
            else:
                return sum_func(vmap(map_2d, in_axes=(0, 0))(x, out_start))
    
    return map_fun


def point_slice_shapes(in_grid, out_grid):
    fac = min(in_grid.fac, out_grid.fac)
    if in_grid in out_grid:
        out_shape = tuple((out_grid.spc * fac).tolist())
    else:
        raise ValueError('Point grid must be contained in the signal grid to map points.')
    return out_shape


def point_slice_indices(in_grid, out_grid):
    fac = min(in_grid.fac, out_grid.fac)

    def indices(in_coos, out_spc, out_cen, fac):
        out_lims = out_cen - (out_spc - 1/fac) / 2
        in_coos = jnp.floor(in_coos * fac) / fac + 0.5 / fac
        out_start = (in_coos - out_lims) * fac
        return out_start.astype('int64')
    
    if in_grid.n_copies == 1:
        return indices(in_grid.coos, out_grid.spc, out_grid.cen, fac)
    else:
        return vmap(indices, in_axes=(0, None, None, None))(in_grid.coos, out_grid.spc, out_grid.cen, fac)



def map_tiles(in_grid, out_grid, sum_up=True):
    '''
    Create a function to map a signal from one grid to another.

    Parameters
    ----------
    in_grid : SignalGrid
        The input signal grid.
    out_grid : SignalGrid
        The output signal grid.
    sum_up : bool, optional
        Whether to sum up the input copies when mapping to a single output copy, by default True
    '''
    check_type(in_grid, SignalGrid)
    check_type(out_grid, SignalGrid)

    in_shape, out_shape = tile_slice_shapes(in_grid, out_grid)
    in_start, out_start = signal_slice_indices(in_grid, out_grid)
    sum_func = partial(jnp.sum, axis=0) if sum_up else lambda x: x

    def map_2d(x, in_start, out_start):
        if out_grid.fac < in_grid.fac:
            x = downsample(x, in_grid.fac // out_grid.fac)
        x = dynamic_slice(x, in_start, in_shape)
        x = dynamic_update_slice(jnp.zeros(out_shape, dtype=x.dtype), x, out_start)
        if out_grid.fac > in_grid.fac:
            x = upsample(x, out_grid.fac // in_grid.fac)
        return x

    if in_grid.n_copies == out_grid.n_copies == 1:
        return lambda x: map_2d(x, in_start, out_start)
    elif in_grid.n_copies > 1:
        return lambda x: sum_func(vmap(map_2d, in_axes=(0,0,0))(x, in_start, out_start))
    elif out_grid.n_copies > 1:
        return lambda x: vmap(map_2d, in_axes=(None,0,0))(x, in_start, out_start)


def tile_slice_shapes(in_grid, out_grid):
    fac = min(in_grid.fac, out_grid.fac)
    out_shape = tuple((out_grid.spc * fac).tolist())
    in_shape = tuple((in_grid.spc * fac).tolist())
    return in_shape, out_shape

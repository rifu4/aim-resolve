import jax
import jax.numpy as jnp
import numpy as np
import numpy as np
from jax import vmap
from jax.lax import dynamic_slice, dynamic_update_slice, fori_loop
from functools import partial

from .grid import SignalGrid, PointGrid



def downsample(array, factor):
    h, w = array.shape[-2], array.shape[-1]
    assert h % factor == 0 and w % factor == 0, "Dims must be divisible by factor."
    shape = array.shape[:-2] + (h // factor, factor, w // factor, factor)
    return array.reshape(shape).mean(axis=(-3, -1))
    

def upsample(array, factor):
    return array.repeat(factor, axis=-2).repeat(factor, axis=-1)



def map_signal(in_grid, out_grid):
    '''
    Create a function to map a signal from one grid to another.

    Parameters
    ----------
    in_grid : SignalGrid or PointGrid
        The input grid.
    out_grid : SignalGrid
        The output signal grid.
    '''
    match (in_grid, out_grid):
        case (SignalGrid(), SignalGrid()):
            in_shape, out_shape = signal_slice_shapes(in_grid, out_grid)
            in_start, out_start = signal_slice_indices(in_grid, out_grid)
        case (PointGrid(), SignalGrid()):
            in_shape, out_shape = point_slice_shapes(in_grid, out_grid)
            in_start, out_start = point_slice_indices(in_grid, out_grid)
        case _:
            raise TypeError('Mapping only supported from SignalGrid to SignalGrid, or from PointGrid to SignalGrid.')
        
    fun = partial(map_array,
        in_copies=in_grid.n_copies,
        out_copies=out_grid.n_copies,
        in_shape=in_shape,
        out_shape=out_shape,
        in_start=in_start,
        out_start=out_start,
        zoom=out_grid.fac / in_grid.fac,
    )
    return fun



def map_array(in_array, in_copies, out_copies, in_shape, out_shape, in_start, out_start, zoom):
    if isinstance(in_array, np.ndarray):
        np_type = True
        device = jax.devices('cpu')[0]
    else:
        np_type = False
        device = jax.devices()[0]

    with jax.default_device(device):
        in_array = jnp.asarray(in_array)
        in_start = jnp.asarray(in_start)
        out_start = jnp.asarray(out_start)

        if zoom < 1 and in_shape == (1, 1):
            in_array /= int(1/zoom)**2
        elif zoom < 1:
            in_array = downsample(in_array, int(1/zoom))

        match (in_array.ndim, in_copies, out_copies):
            # (x, y) -> (x', y')
            case (2, 1, 1):
                out_array = jnp.zeros(out_shape, dtype=in_array.dtype)
                out_array = map_array_2d(in_array, out_array, in_start, out_start, in_shape)
            # (x, y) -> (n_out, x', y')
            case (2, 1, n_out):
                out_array = jnp.zeros((n_out,) + out_shape, dtype=in_array.dtype)
                loop_over_n_out = lambda i,o: o.at[i].set(map_array_2d(in_array, o[i], in_start[i], out_start[i], in_shape))
                out_array = fori_loop(0, n_out, loop_over_n_out, out_array)

            # (n/f, x, y) -> (n/f, x', y')
            case (3, 1, 1):
                n = in_array.shape[0]
                out_array = jnp.zeros((n,) + out_shape, dtype=in_array.dtype)
                out_array = vmap(map_array_2d, in_axes=(0, 0, None, None, None))(in_array, out_array, in_start, out_start, in_shape)
            # (n_in, x, y) -> (x', y')
            case (3, n_in, 1):
                out_array = jnp.zeros(out_shape, dtype=in_array.dtype)
                loop_over_n_in = lambda i,o : map_array_2d(in_array[i], o, in_start[i], out_start[i], in_shape)
                out_array = fori_loop(0, n_in, loop_over_n_in, out_array)
            # (n_in, x, y) -> (n_out, x', y') if n_in == n_out
            case (3, n_in, n_out) if n_in == n_out:
                out_array = jnp.zeros((n_out,) + out_shape, dtype=in_array.dtype)
                loop_over_n_out = lambda i,o: o.at[i].set(map_array_2d(in_array[i], o[i], in_start[i], out_start[i], in_shape))
                out_array = fori_loop(0, n_out, loop_over_n_out, out_array)
            # (f, x, y) -> (n_out, f, x, y)
            case (3, 1, n_out):
                f = in_array.shape[0]
                out_array = jnp.zeros((n_out, f) + out_shape, dtype=in_array.dtype)
                vmap_over_f = vmap(map_array_2d, in_axes=(0, 0, None, None, None))
                loop_over_n_out = lambda i,o: o.at[i].set(vmap_over_f(in_array, o[i], in_start[i], out_start[i], in_shape))
                out_array = fori_loop(0, n_out, loop_over_n_out, out_array)

            # (n, f, x, y) -> (n, f, x', y')
            case (4, 1, 1):
                n, f = in_array.shape[:2]
                out_array = jnp.zeros((n, f) + out_shape, dtype=in_array.dtype)
                vmap_over_f = vmap(map_array_2d, in_axes=(0, 0, None, None, None))
                loop_over_n = lambda i,o: o.at[i].set(vmap_over_f(in_array[i], o[i], in_start, out_start, in_shape))
                out_array = fori_loop(0, n, loop_over_n, out_array)
            # (n_in, f, x, y) -> (f, x', y')
            case (4, n_in, 1):
                f = in_array.shape[1]
                out_array = jnp.zeros((f,) + out_shape, dtype=in_array.dtype)
                vmap_over_f = vmap(map_array_2d, in_axes=(0, 0, None, None, None))
                loop_over_n_in = lambda i,o : vmap_over_f(in_array[i], o, in_start[i], out_start[i], in_shape)
                out_array = fori_loop(0, n_in, loop_over_n_in, out_array)
            # (n_in, f, x, y) -> (n_out, f, x', y') if n_in == n_out
            case (4, n_in, n_out) if n_in == n_out:
                f = in_array.shape[1]
                out_array = jnp.zeros((n_out, f) + out_shape, dtype=in_array.dtype)
                vmap_over_f = vmap(map_array_2d, in_axes=(0, 0, None, None, None))
                loop_over_n_out = lambda i,o: o.at[i].set(vmap_over_f(in_array[i], o[i], in_start[i], out_start[i], in_shape))
                out_array = fori_loop(0, n_out, loop_over_n_out, out_array) 

        if zoom > 1:
            out_array = upsample(out_array, int(zoom)) 

        if np_type:
            return np.asarray(out_array); del out_array

    return out_array


def map_array_2d(in_array, out_array, in_start, out_start, in_shape):
    in_slice = dynamic_slice(in_array, in_start, in_shape)
    in_slice += dynamic_slice(out_array, out_start, in_shape)
    out_array = dynamic_update_slice(out_array, in_slice, out_start)
    return out_array



def signal_slice_shapes(in_grid, out_grid):
    factor = min(in_grid.fac, out_grid.fac)
    out_shape = tuple((out_grid.spc * factor).tolist())
    in_shape = tuple((in_grid.spc * factor).tolist())
    if in_grid in out_grid:
        in_shape = tuple((in_grid.spc * factor).tolist())
    elif out_grid in in_grid:
        in_shape = tuple((out_grid.spc * factor).tolist())
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
    
    match (in_grid.n_copies, out_grid.n_copies):
        case (1, 1):
            return indices(in_grid.spc, in_grid.cen, out_grid.spc, out_grid.cen, fac)
        case (1, n_out):
            return vmap(indices, in_axes=(None, None, None, 0, None))(in_grid.spc, in_grid.cen, out_grid.spc, out_grid.cen, fac)
        case (n_in, 1):
            return vmap(indices, in_axes=(None, 0, None, None, None))(in_grid.spc, in_grid.cen, out_grid.spc, out_grid.cen, fac)
        case (n_in, n_out) if n_in == n_out:
            return vmap(indices, in_axes=(None, 0, None, 0, None))(in_grid.spc, in_grid.cen, out_grid.spc, out_grid.cen, fac)
        case _:
            raise ValueError('Number of copies in input and output grids must match, or one of them must be 1.')


def point_slice_shapes(in_grid, out_grid):
    fac = min(in_grid.fac, out_grid.fac)
    in_shape = (1, 1)
    if in_grid in out_grid:
        out_shape = tuple((out_grid.spc * fac).tolist())
    else:
        raise ValueError('Point grid must be contained in the signal grid to map points.')
    return in_shape, out_shape
      

def point_slice_indices(in_grid, out_grid):
    fac = min(in_grid.fac, out_grid.fac)

    def indices(in_coos, out_spc, out_cen, fac):
        out_lims = out_cen - (out_spc - 1/fac) / 2
        in_coos = jnp.floor(in_coos * fac) / fac + 0.5 / fac
        out_start = (in_coos - out_lims) * fac
        out_start = out_start.astype('int64')
        in_start = jnp.zeros_like(out_start)
        return in_start, out_start
    
    match (in_grid.n_copies, out_grid.n_copies):
        case (1, 1):
            return indices(in_grid.coos, out_grid.spc, out_grid.cen, fac)
        case (1, n_out):
            return vmap(indices, in_axes=(None, None, 0, None))(in_grid.coos, out_grid.spc, out_grid.cen, fac)
        case (n_in, 1):
            return vmap(indices, in_axes=(0, None, None, None))(in_grid.coos, out_grid.spc, out_grid.cen, fac)
        case (n_in, n_out) if n_in == n_out:
            return vmap(indices, in_axes=(0, None, 0, None))(in_grid.coos, out_grid.spc, out_grid.cen, fac)
        case _:
            raise ValueError('Number of copies in input and output grids must match, or one of them must be 1.')

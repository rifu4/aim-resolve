import jax.numpy as jnp
from jax import vmap
from jax.lax import dynamic_slice, dynamic_update_slice, fori_loop
from functools import partial

from .grid import SignalGrid, PointGrid



def downsample(array, factor):
    return array.reshape(array.shape[0]//factor, factor, array.shape[1]//factor, factor).mean(axis=(1,3))
    

def upsample(array, factor):
    return array.repeat(factor, axis=0).repeat(factor, axis=1)



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
            in_shape, out_shape = array_slice_shapes(in_grid, out_grid)
            in_start, out_start = array_slice_indices(in_grid, out_grid)
            map_fun = partial(map_array,
                in_copies=in_grid.n_copies,
                out_copies=out_grid.n_copies,
                in_shape=in_shape,
                out_shape=out_shape,
                in_start=in_start,
                out_start=out_start,
                zoom=out_grid.fac / in_grid.fac,
            )
        case (PointGrid(), SignalGrid()):
            out_shape = point_slice_shapes(in_grid, out_grid)
            out_start = point_slice_indices(in_grid, out_grid)
            map_fun = partial(map_point,
                in_copies=in_grid.n_copies,
                out_copies=out_grid.n_copies,
                out_shape=out_shape,
                out_start=out_start,
                zoom=out_grid.fac / in_grid.fac,
            )
        case _:
            raise TypeError('Mapping only supported from SignalGrid to SignalGrid, or from PointGrid to SignalGrid.')

    return map_fun



def array_slice_shapes(in_grid, out_grid):
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


def array_slice_indices(in_grid, out_grid):
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
        

def map_array_2d(in_array, out_array, in_start, out_start, in_shape, zoom):
    if zoom < 1:
        in_array = downsample(in_array, int(1/zoom))

    in_slice = dynamic_slice(in_array, in_start, in_shape)
    in_slice += dynamic_slice(out_array, out_start, in_shape)
    out_array = dynamic_update_slice(out_array, in_slice, out_start)

    if zoom > 1:
        out_array = upsample(out_array, int(zoom))

    return out_array


def map_array(in_array, in_copies, out_copies, in_shape, out_shape, in_start, out_start, zoom):
    in_array = jnp.asarray(in_array)
    in_start = jnp.asarray(in_start)
    out_start = jnp.asarray(out_start)
    match (in_array.ndim, in_copies, out_copies):
        case (2, 1, 1):
            out_array = jnp.zeros(out_shape, dtype=in_array.dtype)
            return map_array_2d(in_array, out_array, in_start, out_start, in_shape, zoom)
        case (2, 1, n_out):
            out_array = jnp.zeros((n_out,) + out_shape, dtype=in_array.dtype)
            loop_fun = lambda i,o: o.at[i].set(map_array_2d(in_array, o[i], in_start[i], out_start[i], in_shape, zoom))
            return fori_loop(0, n_out, loop_fun, out_array)
        case (3, 1, 1):
            out_array = jnp.zeros((in_array.shape[0],) + out_shape, dtype=in_array.dtype)
            loop_fun = lambda i,o: o.at[i].set(map_array_2d(in_array[i], o[i], in_start, out_start, in_shape, zoom))
            return fori_loop(0, in_array.shape[0], loop_fun, out_array)
        case (3, n_in, 1):
            out_array = jnp.zeros(out_shape, dtype=in_array.dtype)
            loop_fun = lambda i,o : map_array_2d(in_array[i], o, in_start[i], out_start[i], in_shape, zoom)
            return fori_loop(0, n_in, loop_fun, out_array)
        case (3, n_in, n_out) if n_in == n_out:
            out_array = jnp.zeros((n_out,) + out_shape, dtype=in_array.dtype)
            loop_fun = lambda i,o: o.at[i].set(map_array_2d(in_array[i], o[i], in_start[i], out_start[i], in_shape, zoom))
            return fori_loop(0, n_out, loop_fun, out_array)



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


def map_point_2d(in_point, out_array, out_start, zoom):
    if zoom < 1:
        in_point /= int(1/zoom)**2

    in_point += dynamic_slice(out_array, out_start, (1, 1))
    out_array = dynamic_update_slice(out_array, in_point, out_start)

    if zoom > 1:
        out_array = upsample(out_array, int(zoom))

    return out_array
        

def map_point(in_point, in_copies, out_copies, out_shape, out_start, zoom):
    in_point = jnp.asarray(in_point)
    out_start = jnp.asarray(out_start)
    match (in_point.ndim, in_copies, out_copies):
        case (2, 1, 1):
            out_array = jnp.zeros(out_shape, dtype=in_point.dtype)
            return map_point_2d(in_point, out_array, out_start, zoom)
        case (2, 1, n_out):
            out_array = jnp.zeros((n_out,) + out_shape, dtype=in_point.dtype)
            loop_fun = lambda i,o: o.at[i].set(map_point_2d(in_point, o[i], out_start[i], zoom))
            return fori_loop(0, n_out, loop_fun, out_array)
        case (3, 1, 1):
            out_array = jnp.zeros((in_point.shape[0],) + out_shape, dtype=in_point.dtype)
            loop_fun = lambda i,o: o.at[i].set(map_array_2d(in_point[i], o[i], out_start, zoom))
            return fori_loop(0, in_point.shape[0], loop_fun, out_array)
        case (3, n_in, 1):
            out_array = jnp.zeros(out_shape, dtype=in_point.dtype)
            loop_fun = lambda i,o : map_point_2d(in_point[i], o, out_start[i], zoom)
            return fori_loop(0, n_in, loop_fun, out_array)
        case (3, n_in, n_out) if n_in == n_out:
            out_array = jnp.zeros((n_out,) + out_shape, dtype=in_point.dtype)
            loop_fun = lambda i,o: o.at[i].set(map_point_2d(in_point[i], o[i], out_start[i], zoom))
            return fori_loop(0, n_out, loop_fun, out_array)

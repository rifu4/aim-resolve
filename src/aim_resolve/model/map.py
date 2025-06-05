import jax.numpy as jnp
from jax import vmap
from jax.scipy.ndimage import map_coordinates

from .space import SignalSpace, jax_rotate
from .util import check_type


    
def map_signal(x, in_space, out_space, order=0, vmap_sum=True):
    '''
    Map one or more signals from a SignalSpace to another SignalSpace.
    
    Parameters
    ----------
    x : np.ndarray
        The signal to be mapped
    in_space : SignalSpace
        The input space of the signal
    out_space : SignalSpace
        The output space of the signal
    order : int, optional
        The order of the interpolation, by default 0
    '''
    check_type(in_space, SignalSpace)
    check_type(out_space, SignalSpace)

    if x.ndim == 2:
        return map_one_signal(x, in_space.dis, in_space.cen, in_space.rot, out_space.coos, order)
    else:
        if in_space.n_copies > 1:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, 0, 0, None, None))
            res = vmap_one_signal(x, in_space.dis, in_space.cen, in_space.rot, out_space.coos, order)            
        else:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, None, None, None, None))
            res = vmap_one_signal(x, in_space.dis, in_space.cen, in_space.rot, out_space.coos, order)
        if vmap_sum:
            return jnp.sum(res, axis=0)
        else:
            return res
    

def map_one_signal(x, in_dis, in_cen, in_rot, out_coos, order=0): 
    out_coos_T = out_coos.T.reshape(-1, 2)
    out_coos_T -= in_cen
    out_coos_T = jax_rotate(out_coos_T, -in_rot)
    out_coos_T /= in_dis
    out_coos_T += 0.5 * (jnp.array(x.shape) - 1)
    out_coos = out_coos_T.reshape(out_coos.T.shape).T
    return map_coordinates(x, out_coos, order)



def map_points(x, in_coos, out_space, order=0, vmap_sum=True):
    '''
    Map one or more point sources from their coordinates to a SignalSpace.

    Parameters
    ----------
    x : np.ndarray
        The point sources to be mapped
    in_coos : np.ndarray
        The coordinates of the point sources
    out_space : SignalSpace
        The output space the point sources are mapped to
    order : int, optional
        The order of the interpolation, by default 0
    '''
    check_type(out_space, SignalSpace)

    if x.ndim == 2:
        return map_one_point(x, in_coos, out_space.coos, out_space.dis, order)
    else:
        vmap_one_point = vmap(map_one_point, in_axes=(0, 0, None, None, None))
        res = vmap_one_point(x, in_coos, out_space.coos, out_space.dis, order)
        if vmap_sum:
            return jnp.sum(res, axis=0)
        else:
            return res


def map_one_point(x, in_coos, out_coos, out_dis, order=0):
    out_coos_T = out_coos.T.reshape(-1, 2)
    out_coos_T -= in_coos
    out_coos_T = out_coos_T / out_dis
    out_coos_T -= jnp.diag(out_coos_T[jnp.abs(out_coos_T).argmin(axis=0)])
    out_coos = out_coos_T.reshape(out_coos.T.shape).T
    return map_coordinates(x, out_coos, order)



def map_tiles(x, in_dis, in_cen, in_rot, out_space, n_copies=1, order=0, vmap_sum=True):
    '''
    Map one or more signals from a SignalSpace to another SignalSpace.
    
    Parameters
    ----------
    x : np.ndarray
        The signal to be mapped
    in_dis: np.ndarray
        The input distances of the signals
    in_cen: np.ndarray
        The input centers of the signals
    in_rot: np.ndarray
        The input rotations of the signals
    out_space : SignalSpace
        The output space of the signal
    n_copies : int, optional
        The number of copies of the signal, by default 1
    order : int, optional
        The order of the interpolation, by default 0
    vmap_sum : bool, optional
        If True, the output will be summed over the first axis, by default True
    '''
    check_type(out_space, SignalSpace)

    if x.ndim == 2:
        return map_one_signal(x, in_dis, in_cen, in_rot, out_space.coos, order)
    else:
        if n_copies > 1:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, 0, 0, None, None))
            res = vmap_one_signal(x, in_dis, in_cen, in_rot, out_space.coos, order)            
        else:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, None, None, None, None))
            res = vmap_one_signal(x, in_dis, in_cen, in_rot, out_space.coos, order)
        if vmap_sum:
            return jnp.sum(res, axis=0)
        else:
            return res

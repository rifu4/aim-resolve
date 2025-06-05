import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import vmap

from .observation import Observation
from ..model.space import SignalSpace
from ..model.util import check_type



def point_response(x, in_coos, in_space, observation):
    '''
    Map one or more point sources from their coordinates to the UV-space.

    Parameters
    ----------
    x : np.ndarray
        The points to be mapped
    in_coos : np.ndarray
        The coordinates of the points
    obs : Observation
        The radio observation
    space : SignalSpace
        The SignalSpace the point sources are defined in
    '''
    check_type(in_space, SignalSpace)
    check_type(observation, Observation)

    if x.ndim == 2:
        return one_point_response(x, in_coos, in_space.dis, observation)
    else:
        vmap_one_point = vmap(one_point_response, in_axes=(0, 0, None, None))
        res = vmap_one_point(x, in_coos, in_space.dis, observation)
        return jnp.sum(res, axis=0)
    

def one_point_response(
        x,
        in_coos,
        in_dis,
        observation,
):
    speedoflight = 299792458.0
    freq = observation.freq
    uvw = observation.uvw
    vol = in_dis.prod()

    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    uv = (2 * np.pi * uvw[:, :2] * in_dis * np.array([1, -1])) % (2 * np.pi)
    u, v = uv.T
    
    res = vol * x * jnp.exp(-1j * (u * in_coos[0] + v * in_coos[1]))

    return jnp.expand_dims(res.reshape(-1, len(freq)), 0)



def signal_response(in_space, observation, wgridding=False, epsilon=1e-9):
    '''
    Apply the signal response to one or more signals
    
    Parameters
    ----------
    in_space : SignalSpace
        The input space of the signal
    observation : Observation
        The radio observation
    wgridding : bool, optional
        Whether to use wgridding (ducc response), by default False
    epsilon : float, optional
        The tolerance for the response function, by default 1e-9
    '''
    check_type(in_space, SignalSpace)
    check_type(observation, Observation)

    if wgridding:
        return ducc_response(in_space, observation, wgridding, epsilon)
    else:
        return finu_response(in_space, observation, epsilon)
    


def ducc_response(in_space, observation, wgridding=True, epsilon=1e-9):
    '''
    Apply the ducc response to one signal. Does not work with multiple signals.
    
    Parameters
    ----------
    in_space : SignalSpace
        The input space of the signal
    observation : Observation
        The radio observation
    wgridding : bool, optional
        Whether to use wgridding, by default True
    epsilon : float, optional
        The tolerance for the ducc response, by default 1e-9
    '''
    from jaxbind.contrib import jaxducc0
    check_type(in_space, SignalSpace)
    check_type(observation, Observation)
    if in_space.n_copies > 1:
        raise ValueError('ducc response cannot vmap over multiple signals')

    freq = observation.freq
    uvw = observation.uvw
    uvw[:,:2] = rotate(uvw[:,:2], in_space.rot)
    cen = in_space.cen * np.array([1,-1])
    cen = rotate(cen, in_space.rot)
    vol = in_space.dis.prod()

    wg = jaxducc0.get_wgridder(
        pixsize_x = in_space.dis[0],
        pixsize_y = in_space.dis[1],
        npix_x = in_space.shape[0],
        npix_y = in_space.shape[1],
        center_x = cen[0],
        center_y = cen[1],
        do_wgridding = wgridding,
        epsilon = epsilon,
        nthreads = 1,
        verbosity = 0,
        flip_v = True,
    )
    wgridder = partial(wg, uvw, freq)
    
    def apply_ducc(x):
        res = vol * wgridder(x)[0]
        return jnp.expand_dims(res, 0)

    return apply_ducc



def finu_response(in_space, observation, epsilon=1e-9):
    '''
    Apply the finufft response to one or more signals
    
    Parameters
    ----------
    in_space : SignalSpace
        The input space of the signal
    observation : Observation
        The radio observation
    epsilon : float, optional
        The tolerance for the finufft response, by default 1e-9
    '''
    from jax_finufft import nufft2
    check_type(in_space, SignalSpace)
    check_type(observation, Observation)
    if in_space.n_copies > 1:
        raise ValueError('finu response cannot vmap over multiple signals yet')

    speedoflight = 299792458.0
    freq = observation.freq
    uvw = observation.uvw
    uvw[:,:2] = rotate(uvw[:,:2], in_space.rot)
    cen = in_space.cen * np.array([1,-1])
    cen = rotate(cen, in_space.rot)
    vol = in_space.dis.prod()

    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    uv = (2 * np.pi * uvw[:, :2] * in_space.dis * np.array([1, -1])) % (2 * np.pi)
    u, v = uv.T

    def apply_finu(x):
        res = vol * nufft2(x.astype(np.complex128), u, v, eps=epsilon)
        res *= jnp.exp(-1j * (u * cen[0] + v * cen[1]))
        return jnp.expand_dims(res.reshape(-1, len(freq)), 0)

    return apply_finu



def rotate(xy, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[c, s], [-s, c]])
    return xy @ R

"""Response operator construction for radio interferometric imaging."""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import vmap

from ..model.grid import SignalGrid
from ..model.util import check_type
from .observation import Observation


def point_response(x, in_coos, in_grid, observation):
    """Map one or more point sources from their coordinates to UV-space.

    Parameters
    ----------
    x : np.ndarray
        The point amplitudes to be mapped.
    in_coos : np.ndarray
        The coordinates of the point sources.
    in_grid : SignalGrid
        The signal grid the point sources are defined on.
    observation : Observation
        The radio observation.

    Returns
    -------
    jnp.ndarray
        The visibilities corresponding to the point sources.
    """
    check_type(in_grid, SignalGrid)
    check_type(observation, Observation)

    if x.ndim == 2:
        return one_point_response(x, in_coos, in_grid.dis, observation)
    else:
        vmap_one_point = vmap(one_point_response, in_axes=(0, 0, None, None))
        res = vmap_one_point(x, in_coos, in_grid.dis, observation)
        return jnp.sum(res, axis=0)


def one_point_response(
    x,
    in_coos,
    in_dis,
    observation,
):
    """Compute the visibility response of a single point source.

    Parameters
    ----------
    x : jnp.ndarray
        Amplitude of the point source (2-d array).
    in_coos : jnp.ndarray
        Coordinates of the point source.
    in_dis : np.ndarray
        Pixel distances of the signal grid.
    observation : Observation
        The radio observation.

    Returns
    -------
    jnp.ndarray
        Visibility contribution of the point source.
    """
    speedoflight = 299792458.0
    freq = observation.freq
    uvw = observation.uvw
    vol = in_dis.prod()

    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    uv = (2 * np.pi * uvw[:, :2] * in_dis * np.array([1, -1])) % (2 * np.pi)
    u, v = uv.T

    res = vol * x * jnp.exp(-1j * (u * in_coos[0] + v * in_coos[1]))

    return jnp.expand_dims(res.reshape(-1, len(freq)), 0)


def signal_response(in_grid, observation, wgridding=False, epsilon=1e-9):
    """Apply the signal response to one or more signals.

    Parameters
    ----------
    in_grid : SignalGrid
        The input space of the signal.
    observation : Observation
        The radio observation.
    wgridding : bool, optional
        Whether to use wgridding (ducc response), by default False.
    epsilon : float, optional
        The tolerance for the response function, by default 1e-9.

    Returns
    -------
    callable
        A function that maps a sky image to visibilities.
    """
    check_type(in_grid, SignalGrid)
    check_type(observation, Observation)

    if wgridding:
        return ducc_response(in_grid, observation, wgridding, epsilon)
    else:
        return finu_response(in_grid, observation, epsilon)


def ducc_response(in_grid, observation, wgridding=True, epsilon=1e-9):
    """Apply the ducc response to one signal.

    Does not work with multiple signals.

    Parameters
    ----------
    in_grid : SignalGrid
        The input space of the signal.
    observation : Observation
        The radio observation.
    wgridding : bool, optional
        Whether to use wgridding, by default True.
    epsilon : float, optional
        The tolerance for the ducc response, by default 1e-9.

    Returns
    -------
    callable
        A function that maps a sky image to visibilities using ducc.

    Raises
    ------
    ValueError
        If ``in_grid.n_copies > 1`` since ducc cannot vmap over
        multiple signals.
    """
    from jaxbind.contrib import jaxducc0

    check_type(in_grid, SignalGrid)
    check_type(observation, Observation)
    if in_grid.n_copies > 1:
        raise ValueError("ducc response cannot vmap over multiple signals")

    freq = observation.freq
    uvw = observation.uvw
    vol = in_grid.dis.prod()

    wg = jaxducc0.get_wgridder(
        pixsize_x=in_grid.dis[0],
        pixsize_y=in_grid.dis[1],
        npix_x=in_grid.shape[0],
        npix_y=in_grid.shape[1],
        center_x=in_grid.cen[0],
        center_y=-in_grid.cen[1],
        do_wgridding=wgridding,
        epsilon=epsilon,
        nthreads=1,
        verbosity=0,
        flip_v=True,
    )
    wgridder = partial(wg, uvw, freq)

    def apply_ducc(x):
        res = vol * wgridder(x)[0]
        return jnp.expand_dims(res, 0)

    return apply_ducc


def finu_response(in_grid, observation, epsilon=1e-9):
    """Apply the finufft response to one or more signals.

    Parameters
    ----------
    in_grid : SignalGrid
        The input space of the signal.
    observation : Observation
        The radio observation.
    epsilon : float, optional
        The tolerance for the finufft response, by default 1e-9.

    Returns
    -------
    callable
        A function that maps a sky image to visibilities using finufft.

    Raises
    ------
    ValueError
        If ``in_grid.n_copies > 1`` since finu response cannot vmap
        over multiple signals yet.
    """
    from jax_finufft import nufft2

    check_type(in_grid, SignalGrid)
    check_type(observation, Observation)
    if in_grid.n_copies > 1:
        raise ValueError("finu response cannot vmap over multiple signals yet")

    speedoflight = 299792458.0
    freq = observation.freq
    uvw = observation.uvw
    cen = in_grid.cen * np.array([1, -1])
    vol = in_grid.dis.prod()

    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1)).reshape(-1, 3)
    uv = (2 * np.pi * uvw[:, :2] * in_grid.dis * np.array([1, -1])) % (2 * np.pi)
    u, v = uv.T

    def apply_finu(x):
        res = vol * nufft2(x.astype(np.complex128), u, v, eps=epsilon)
        res *= jnp.exp(-1j * (u * cen[0] + v * cen[1]))
        return jnp.expand_dims(res.reshape(-1, len(freq)), 0)

    return apply_finu


def rotate(xy, phi):
    """Rotate 2-d coordinates by a given angle.

    Parameters
    ----------
    xy : np.ndarray
        Array of shape ``(N, 2)`` with 2-d coordinates.
    phi : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Rotated coordinates with the same shape as *xy*.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[c, s], [-s, c]])
    return xy @ R

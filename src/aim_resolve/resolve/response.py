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

    A single point source is passed straight to :func:`one_point_response`,
    which maps each model frequency to exactly one data frequency. Multiple
    point sources (``in_coos`` of shape ``(n_points, 2)``) are handled by
    vmapping the single-point response over the point (copies) axis and summing
    their visibility contributions.

    Parameters
    ----------
    x : jnp.ndarray
        Point amplitudes. For a single source the leading axis (if present) is
        the frequency axis; for ``n_points`` sources the leading axis is the
        point axis (an optional frequency axis follows).
    in_coos : jnp.ndarray
        Point-source coordinates: ``(2,)`` for one source, ``(n_points, 2)``
        for several.
    in_grid : SignalGrid
        The signal grid the point sources are defined on.
    observation : Observation
        The radio observation.

    Returns
    -------
    jnp.ndarray
        The visibilities corresponding to the point sources, shape
        ``(1, nrow, nfreq)``.
    """
    check_type(in_grid, SignalGrid)
    check_type(observation, Observation)

    in_coos = jnp.asarray(in_coos)
    if in_coos.ndim == 1:
        # Single point source; one_point_response maps model -> data freqs.
        return one_point_response(x, in_coos, in_grid.dis, observation)

    # Multiple point sources: apply the single-point response to each (vmap over
    # the point axis of both amplitudes and coordinates) and sum their
    # contributions. The frequency axis, if any, is handled inside the response.
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

    Supports multi-frequency point models: the per-frequency amplitude is
    mapped onto the observation frequencies one-to-one. The amplitude ``x`` may
    carry singleton spatial dimensions (the point grid is ``(1, 1)``); it is
    flattened and must reduce either to a scalar (the same amplitude at every
    data frequency) or to ``(nfreq,)`` (one amplitude per data frequency).

    Parameters
    ----------
    x : jnp.ndarray
        Amplitude of the point source. Broadcastable to the data frequencies
        after flattening: shape ``(1, 1)`` / scalar, or ``(nfreq, 1, 1)``.
    in_coos : jnp.ndarray
        Coordinates of the point source, shape ``(2,)``.
    in_dis : np.ndarray
        Pixel distances of the signal grid.
    observation : Observation
        The radio observation.

    Returns
    -------
    jnp.ndarray
        Visibility contribution of the point source, shape ``(1, nrow, nfreq)``.
    """
    speedoflight = 299792458.0
    freq = observation.freq
    uvw = observation.uvw
    vol = in_dis.prod()

    # Per-(row, freq) uv sample points; keep the frequency axis so a
    # per-frequency amplitude maps one model channel onto one data frequency.
    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1))
    uv = (2 * np.pi * uvw[..., :2] * in_dis * np.array([1, -1])) % (2 * np.pi)
    u = uv[..., 0]  # (nrow, nfreq)
    v = uv[..., 1]  # (nrow, nfreq)

    amp = jnp.reshape(x, (-1,))  # scalar / (1,) or (nfreq,)
    phase = jnp.exp(-1j * (u * in_coos[0] + v * in_coos[1]))  # (nrow, nfreq)
    res = vol * amp[None, :] * phase  # broadcast (1, k) over (nrow, nfreq)

    return jnp.expand_dims(res, 0)  # (1, nrow, nfreq)


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
    """Apply the finufft (GPU, no wgridding) response to a sky model.

    The returned callable maps a sky image to visibilities via a type-2
    non-uniform FFT. It handles three axes of batching:

    * **Frequency.** A single 2-d image ``(nx, ny)`` is evaluated at every data
      frequency, while a multi-frequency cube ``(nfreq, nx, ny)`` maps each sky
      channel to exactly one data frequency (the frequency axis is vmapped so
      channel ``i`` is sampled at the uv-coordinates of data frequency ``i``).
    * **Extended components (tiles).** When ``in_grid.n_copies > 1`` the leading
      axis of the input holds the tile copies ``(n_copies, [nfreq,] nx, ny)``.
      Every copy shares the same grid shape and pixel size, so they share the
      uv-sample points and differ only in their phase center
      (``in_grid.cen[i]``). The single-grid response is vmapped over the copies
      axis and the contributions are summed -- exactly like
      :func:`point_response` sums several point sources.

    Parameters
    ----------
    in_grid : SignalGrid
        The input space of the signal. For tile models this is the (small) tile
        grid with ``n_copies > 1`` and one center per copy.
    observation : Observation
        The radio observation.
    epsilon : float, optional
        The tolerance for the finufft response, by default 1e-9.

    Returns
    -------
    callable
        A function that maps a sky image / cube (optionally with a leading
        copies axis) to visibilities of shape ``(1, nrow, nfreq)`` using
        finufft.
    """
    from jax_finufft import nufft2

    check_type(in_grid, SignalGrid)
    check_type(observation, Observation)

    speedoflight = 299792458.0
    freq = observation.freq
    uvw = observation.uvw
    dis = in_grid.dis
    vol = dis.prod()
    nfreq = len(freq)
    n_copies = in_grid.n_copies

    # Per-(row, freq) uv sample points on the grid, kept as (nrow, nfreq) so we
    # can either flatten them (one shared image over all frequencies) or index
    # the frequency axis (one sky channel per data frequency). Every tile copy
    # shares these points since they share the grid shape and pixel size.
    uvw = np.transpose((uvw[..., None] * freq / speedoflight), (0, 2, 1))
    uv = (2 * np.pi * uvw[..., :2] * dis * np.array([1, -1])) % (2 * np.pi)
    u = uv[..., 0]  # (nrow, nfreq)
    v = uv[..., 1]  # (nrow, nfreq)

    # Per-copy phase centers: (2,) for a single grid, (n_copies, 2) for tiles.
    cen = in_grid.cen * np.array([1, -1])

    def apply_image(image, u1, v1, cen1):
        # image: (nx, ny); u1, v1: (nrow,) uv coords of one data frequency;
        # cen1: (2,) phase center of this grid copy.
        res = vol * nufft2(image.astype(np.complex128), u1, v1, eps=epsilon)
        return res * jnp.exp(-1j * (u1 * cen1[0] + v1 * cen1[1]))  # (nrow,)

    def apply_grid(x, cen1):
        # x: sky for one grid copy, (nx, ny) or (nfreq, nx, ny). -> (nrow, nfreq)
        if x.ndim == 2:
            # Single sky image evaluated at every data frequency.
            return apply_image(x, u.reshape(-1), v.reshape(-1), cen1).reshape(-1, nfreq)
        # One sky channel per data frequency: vmap the leading freq axis.
        if x.shape[0] != nfreq:
            raise ValueError(
                f"multi-frequency sky has {x.shape[0]} channels but the "
                f"observation has {nfreq} frequencies"
            )
        return vmap(apply_image, in_axes=(0, 1, 1, None))(x, u, v, cen1).T

    def apply_finu(x):
        if n_copies == 1:
            res = apply_grid(x, cen)
        else:
            # Multiple extended components (tiles): apply the single-grid
            # response to each copy (vmap over the leading copies axis, one
            # center per copy) and sum their visibility contributions.
            res = vmap(apply_grid, in_axes=(0, 0))(x, cen)  # (n_copies, nrow, nfreq)
            res = jnp.sum(res, axis=0)
        return jnp.expand_dims(res, 0)  # (1, nrow, nfreq)

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

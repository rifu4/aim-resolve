"""JAX-compatible image transformation utilities."""

import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import correlate2d
from jax.typing import ArrayLike


def gaussian_kernel2d(sigma, radius):
    """Create a 2-D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    radius : int
        Half-width of the kernel in pixels.

    Returns
    -------
    jnp.ndarray
        Kernel of shape ``(2*radius+1, 2*radius+1)``.
    """
    x, y = jnp.meshgrid(
        jnp.arange(-radius, radius + 1), jnp.arange(-radius, radius + 1)
    )
    dst = jnp.sqrt(x**2 + y**2)
    normal = 1 / (2 * jnp.pi * sigma**2)
    return jnp.exp(-(dst**2 / (2.0 * sigma**2))) * normal


def gaussian_filter2d(x, sigma, radius=5, normalize=False):
    """Apply a 2-D Gaussian filter (JAX-compatible).

    Parameters
    ----------
    x : jnp.ndarray
        2-D input array.
    sigma : float
        Standard deviation of the Gaussian kernel.
    radius : int, optional
        Kernel half-width. Default is 5.
    normalize : bool, optional
        If True, normalise the output by its maximum. Default is False.

    Returns
    -------
    jnp.ndarray
        Filtered array of the same shape as *x*.
    """

    def true_branch(x, sigma):
        k = gaussian_kernel2d(sigma, radius)
        y = correlate2d(x, k, "same")
        if not normalize:
            y *= lax.cond(
                y.max() > 0, lambda x, y: x.max() / y.max(), lambda x, y: 1.0, x, y
            )
        return y

    def false_branch(x, sigma):
        return x

    return lax.cond(sigma > 0, true_branch, false_branch, x, sigma)


def rotate_data(
    m: ArrayLike,
    k: int = 1,
    axes: tuple[int, int] = (0, 1),
):
    """Rotate a 2-D array by ``k * 90`` degrees (JAX-compatible).

    Parameters
    ----------
    m : array_like
        2-D input array.
    k : int, optional
        Number of 90-degree counter-clockwise rotations. Default is 1.
    axes : tuple of int, optional
        Rotation plane. Default is ``(0, 1)``.

    Returns
    -------
    jnp.ndarray
        Rotated array.
    """
    k = k % 4
    return lax.switch(
        k,
        [
            lambda: m,
            lambda: jnp.rot90(m, k=1, axes=axes),
            lambda: jnp.rot90(m, k=2, axes=axes),
            lambda: jnp.rot90(m, k=3, axes=axes),
        ],
    )


def flip_data(
    m: ArrayLike,
    axis: int = 0,
):
    """Flip a 2-D array along the given axis (JAX-compatible).

    Parameters
    ----------
    m : array_like
        2-D input array.
    axis : int, optional
        Flip mode: 0 = no flip, 1 = rows, 2 = columns, 3 = both.
        Default is 0.

    Returns
    -------
    jnp.ndarray
        Flipped array.
    """
    axis = axis % 4
    return lax.switch(
        axis,
        [
            lambda: m,
            lambda: jnp.flip(m, axis=0),
            lambda: jnp.flip(m, axis=1),
            lambda: jnp.flip(m, axis=(0, 1)),
        ],
    )

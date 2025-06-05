import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import correlate2d
from jax.typing import ArrayLike



def gaussian_kernel2d(sigma, radius):
    '''
    Re-implementation of scipy.ndimage.gaussian_kernel2d
    
    Parameters
    ----------
    sigma : float
        standard deviation of the gaussian kernel
    radius : int
        radius of the kernel

    Returns
    -------
    y : ndarray
        2D gaussian kernel
    '''
    x, y = jnp.meshgrid(jnp.arange(-radius, radius+1),
                        jnp.arange(-radius, radius+1))
    dst = jnp.sqrt(x**2+y**2)
    normal = 1/(2 * jnp.pi * sigma**2)
    return jnp.exp(-(dst**2 / (2.0 * sigma**2))) * normal
 


def gaussian_filter2d(x, sigma, radius=5, normalize=False):
    '''
    Re-implementation of scipy.ndimage.gaussian_filter2d

    Parameters
    ----------
    x : ndarray
        2D array to be filtered
    sigma : float
        standard deviation of the gaussian kernel
    radius : int
        radius of the kernel. Should be something like `int(4*sigma + 0.5)`. Default is 5.
    normalize : bool
        if True, normalize the output by the maximum value. Default is False.
    
    Returns
    -------
    y : ndarray
        2D gaussian filter
    '''
    def true_branch(x, sigma):
        k = gaussian_kernel2d(sigma, radius)
        y = correlate2d(x, k, 'same')
        if not normalize:
            y *= lax.cond(y.max() > 0, lambda x,y: x.max() / y.max(), lambda x,y: 1., x, y)
        return y

    def false_branch(x, sigma):
        return x

    return lax.cond(sigma > 0, true_branch, false_branch, x, sigma)



def rotate_data(
        m : ArrayLike,
        k : int = 1,
        axes: tuple[int, int] = (0, 1),
):
    '''
    Rotate a 2D array by k * 90 degrees

    Parameters
    ----------
    m : ndarray
        2D array to be rotated
    k : int
        number of 90 degree rotations. Default is 1.
    axes : tuple of int
        axes to rotate. Default is (0, 1).
    '''
    k = k % 4
    return lax.switch(
        k,
        [lambda: m,
         lambda: jnp.rot90(m, k=1, axes=axes),
         lambda: jnp.rot90(m, k=2, axes=axes),
         lambda: jnp.rot90(m, k=3, axes=axes),]
    )



def flip_data(
        m : ArrayLike,
        axis: int = 0,
):
    '''
    Flip a 2D array along the given axis

    Parameters
    ----------
    m : ndarray
        2D array to be flipped
    axis : int
        axis to flip. 0 for no axis, 1 for x-axis, 2 for y-axis, 3 for both axes. Default is 0.
    '''
    axis = axis % 4
    return lax.switch(
        axis,
        [lambda: m,
         lambda: jnp.flip(m, axis=0),
         lambda: jnp.flip(m, axis=1),
         lambda: jnp.flip(m, axis=(0, 1)),],
    )

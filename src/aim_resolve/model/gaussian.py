"""Gaussian spatial model for AIM-Resolve."""

import jax.numpy as jnp
import numpy as np
from collections.abc import Mapping
from functools import partial
from typing import Union
from nifty.re import Model, VModel, WrappedCall, random_like, lognormal_prior, normal_prior



def gaussian_model(*,
        prefix: str,
        shape: np.ndarray,
        distances: np.ndarray,
        cov_x: Union[tuple, float],
        cov_y: Union[tuple, float],
        scale: Union[tuple, float] = 1.,
        theta: Union[tuple, float] = 0.,
        n_copies: int = 1,
) -> Model:
    """
    Define a Gaussian model with the given parameters.

    For all shape parameters (cov_x, cov_y, scale, theta): if a tuple is
    given, a normal prior is defined with ``(mean, std)``; if a float is
    given, the parameter is set to a constant.

    Parameters
    ----------
    prefix : str
        The prefix for the model.
    shape : tuple
        The shape of the model.
    distances : tuple
        The distances of the model.
    cov_x : tuple or float
        The covariance in x-direction.
    cov_y : tuple or float
        The covariance in y-direction.
    scale : tuple or float, optional
        The scale of the field. Default is 1.
    theta : tuple or float, optional
        The angle of the field. Default is 0.
    n_copies : int, optional
        The number of copies of the model. Default is 1.

    Returns
    -------
    Model
        A NIFTy Model (or VModel if n_copies > 1) representing the
        Gaussian spatial profile.
    """
    ptree = {}
    cov_x = prior_or_const(cov_x, ptree, prefix+'cov_x', normal_prior)
    cov_y = prior_or_const(cov_y, ptree, prefix+'cov_y', normal_prior)
    scale = prior_or_const(scale, ptree, prefix+'scale', normal_prior)
    theta = prior_or_const(theta, ptree, prefix+'theta', normal_prior)

    coordinates = centered_coos(np.array(shape), np.array(distances))

    def gaussian(primals: Mapping) -> jnp.ndarray:
        cx = cov_x(primals) if prefix+'cov_x' in primals else cov_x
        cy = cov_y(primals) if prefix+'cov_y' in primals else cov_y
        sc = scale(primals) if prefix+'scale' in primals else scale
        th = theta(primals) if prefix+'theta' in primals else theta

        x, y = coordinates

        a = jnp.cos(th)**2/(2*cx**2) + jnp.sin(th)**2/(2*cy**2)
        b = -jnp.sin(2*th)/(4*cx**2) + jnp.sin(2*th)/(4*cy**2)
        c = jnp.sin(th)**2/(2*cx**2) + jnp.cos(th)**2/(2*cy**2)
        return sc * jnp.exp(-(a*x**2 + 2*b*x*y + c*y**2))
    
    init = {
        k: partial(random_like, primals=v) for k, v in ptree.items()
    }
    model = Model(gaussian, domain=ptree.copy(), init=init)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model



def prior_or_const(value, ptree, name, prior=lognormal_prior):
    """Generate a prior or a constant value depending on the input type.

    Parameters
    ----------
    value : tuple, list, int, or float
        If a two-element tuple/list, a prior is created with
        ``prior(value[0], value[1])``; otherwise used as a constant.
    ptree : dict
        Parameter tree to update with the new prior domain.
    name : str
        Name key for the wrapped prior.
    prior : callable, optional
        Prior constructor, by default ``lognormal_prior``.

    Returns
    -------
    WrappedCall or scalar
        A callable prior or the unchanged constant value.
    """
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = prior(*value)
        value = WrappedCall(value, name=name)
        ptree.update(value.domain)
    elif not isinstance(value, (int, float)):
        raise TypeError(f'`{value}` must be of type `tuple`, `list`, `int` or `float`')
    return value


def centered_coos(shp, dis):
    """Generate centred coordinates for a given shape and pixel distances.

    The coordinate origin is placed at the centre of the grid.

    Parameters
    ----------
    shp : array_like
        Spatial shape of the grid (2-element).
    dis : array_like
        Pixel distances for each spatial dimension.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(2, *shp)`` with centred x and y coordinates.
    """
    coos = jnp.indices(shp).astype(float)
    coos_T = coos.T.reshape(-1, 2)
    coos_T -= 0.5 * (shp - 1)
    coos_T *= dis
    return coos_T.reshape(coos.T.shape).T

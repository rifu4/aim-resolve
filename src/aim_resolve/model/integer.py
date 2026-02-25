"""Integer-valued prior model for AIM-Resolve."""

import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.tree_util import Partial, tree_map
from nifty.re import VModel, WrappedCall


class IntegerPrior(WrappedCall):
    """Uniformly distributed prior with only integer values.

    Parameters
    ----------
    a_min : int
        Minimum value of the distribution.
    a_max : int
        Maximum value of the distribution.
    step : int, optional
        Step size between allowed values, by default 1.
    **kwargs
        Additional keyword arguments passed to ``WrappedCall``.
    """

    def __init__(self, a_min, a_max, step=1, **kwargs):
        self.low = self.a_min = a_min
        self.high = self.a_max = a_max
        self.step = step
        call = random_int(self.a_min, self.a_max, step)
        super().__init__(call, white_init=True, **kwargs)


def random_int(a_min=0.0, a_max=1.0, step=1):
    """Transform a standard normal distribution to a uniform integer distribution.

    Parameters
    ----------
    a_min : float, optional
        Minimum value, by default 0.
    a_max : float, optional
        Maximum value, by default 1.
    step : int, optional
        Step size, by default 1.

    Returns
    -------
    callable
        A function mapping standard normal samples to integer values.
    """
    norm_cdf = Partial(tree_map, norm.cdf)
    scale = a_max - a_min

    def standard_to_uniform(xi):
        return (a_min + jnp.floor(scale * norm_cdf(xi) / step) * step).astype(int)

    return standard_to_uniform


def integer_model(
    *,
    prefix,
    shape,
    i_min,
    i_max,
    step=1,
    n_copies=1,
):
    """Initialize a uniformly distributed prior with only integer values.

    Parameters
    ----------
    prefix : str
        The prefix for the model.
    shape : tuple
        The shape of the model.
    i_min : int
        The minimum value of the uniform integer distribution.
    i_max : int
        The maximum value of the uniform integer distribution.
    step : int, optional
        The step size for the uniform integer distribution, by default 1.
    n_copies : int, optional
        The number of copies for the model, by default 1.

    Returns
    -------
    IntegerPrior or VModel
        The integer prior model, optionally wrapped for multiple copies.
    """
    model = IntegerPrior(i_min, i_max, step, shape=shape, name=prefix)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model

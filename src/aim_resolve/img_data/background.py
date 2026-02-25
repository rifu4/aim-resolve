"""Background generator model for synthetic sky images."""

import jax.numpy as jnp
from nifty.re import Model, Vector
from typing import Callable

from ..model.prior import prior_model
from ..model.grid import SignalGrid
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init



class BackgroundGenerator(Model):
    """Generative model for a smooth sky background.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    grid : SignalGrid
        Spatial grid of the background.
    i0 : Model
        Prior model for the log-intensity field.
    gaussian : Model or None
        Optional Gaussian envelope multiplied with the signal.
    func : callable or None
        Point-wise activation function (e.g. ``jnp.exp``).
    """

    def __init__(self, grid, i0, gaussian=None, func=jnp.exp):
        check_type(grid, SignalGrid)
        check_type(i0, Model)
        check_type(gaussian, (Model, type(None)))
        check_type(func, (Callable, type(None)))

        self.grid = grid
        self.i0 = i0
        self.gaussian = gaussian
        self.func = func
        super().__init__(
            domain=Vector(domain_tree((self.i0, self.gaussian), error=False)),
            init=model_init((self.i0, self.gaussian), error=False),
        )

    def __call__(self, x):
        x_val = self.i0(x)
        y_val = jnp.zeros(x_val.shape)

        if self.func:
            x_val = self.func(x_val)

        if self.gaussian:
            x_val *= self.gaussian(x)

        return jnp.stack((x_val, y_val, y_val), axis=0)
    
    @classmethod
    def build(cls, *, grid, i0, gaussian=None, func='exp'):
        """Build a background generator from configuration dictionaries.

        Parameters
        ----------
        grid : dict
            Signal grid parameters (see ``SignalGrid``).
        i0 : dict
            Prior model parameters for the intensity field.
        gaussian : dict or None, optional
            Gaussian envelope parameters. Default is None.
        func : str or None, optional
            Name of a ``jax.numpy`` activation function. Default is
            ``'exp'``.

        Returns
        -------
        BackgroundGenerator
            The constructed background generator model.
        """
        grid = SignalGrid.build(**grid)

        i0, _ = prior_model('bg i0 ', grid, **i0)

        if gaussian:
            gaussian, _ = prior_model('bg gm ', grid, **gaussian)

        if func:
            func = getattr(jnp, func, None)

        return cls(grid, i0, gaussian, func)

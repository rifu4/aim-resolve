import jax.numpy as jnp
from nifty.re import Model, Vector
from typing import Callable

from ..model.prior import prior_model
from ..model.grid import SignalGrid
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init



class BackgroundGenerator(Model):
    '''Generate a background model. Use `build` function to create the model.'''

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
        '''
        Build a background generator model.
        
        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid)
        i0 : dict
            Dictionary containing the prior model parameters of the signal (see prior_model)
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters (see gaussian_model), by default None
            -> multiply the signal with a gaussian
        func : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        grid = SignalGrid.build(**grid)

        i0, _ = prior_model('bg i0 ', grid, **i0)

        if gaussian:
            gaussian, _ = prior_model('bg gm ', grid, **gaussian)

        if func:
            func = getattr(jnp, func, None)

        return cls(grid, i0, gaussian, func)

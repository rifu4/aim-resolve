import dataclasses
import jax.numpy as jnp
from jax.typing import ArrayLike
from nifty8.re import Model, VModel, Vector
from typing import Any, Callable

from .map import map_signal
from .prior import prior_model
from .grid import SignalGrid, PointGrid
from .util import check_type, to_shape
from ..optimize.samples import domain_tree, model_init



class SignalModel(Model):
    '''Generate a signal model. Use `build` function to create the model.'''

    mask: Any = dataclasses.field(default=None, metadata=dict(static=False))

    def __init__(self, grid, i0, offset=0, prefix='sm', func=jnp.exp, zero_pad=None, gaussian=None, pspec=None):
        check_type(grid, (SignalGrid, PointGrid))
        check_type(i0, (Model, VModel))
        check_type(offset, ArrayLike)
        check_type(prefix, str)
        check_type(func, (Callable, type(None)))
        check_type(zero_pad, (Callable, type(None)))
        check_type(gaussian, (Model, type(None)))
        check_type(pspec, (Model, VModel, type(None)))

        self.grid = grid
        self.i0 = i0
        self.offset = offset
        self.prefix = prefix
        self.func = func
        self.zero_pad = zero_pad
        self.gaussian = gaussian
        self.pspec = pspec
        self.map_function = lambda x: x
        super().__init__(
            domain = Vector(domain_tree((self.i0, self.gaussian), error=False)), 
            init = model_init((self.i0, self.gaussian), error=False),
        )

    def __call__(self, x):
        res = self.i0(x)
        res += self.offset
        if self.zero_pad:
            res = self.zero_pad(res)
        if self.func:
            res = self.func(res)
        if self.gaussian:
            res *= self.gaussian(x)
        if isinstance(self.mask, ArrayLike):
            res = jnp.where(self.mask, res, 0.0)
        return self.map_function(res)

    @classmethod
    def build(cls, *, grid, i0, offset=0, prefix='sm', func='exp', zero_pad=1.0, gaussian=None):
        '''
        Build a SignalModel from the given parameters.

        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid)
        i0 : dict
            Dictionary containing the prior model parameters of the signal (see prior_model)
        offset : float, optional
            Offset to add to the signal, by default 0
        prefix : str, optional
            Prefix for the model, by default 'sig'
        func : str, optional
            Function to apply to the signal, by default 'exp'
        zero_pad : float, optional
            Zero padding factor, by default 1.0
            -> pad the signal with zeros, 1.0 means no padding
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters (see gaussian_model), by default None
            -> multiply the signal with a gaussian
        '''
        if 'coordinates' in grid:
            grid = PointGrid.build(**grid)
        else:
            grid = SignalGrid.build(**grid)
        
        offset = to_shape(offset, (), 'float64')

        check_type(prefix, str)

        check_type(zero_pad, (int, float))
        pad_grid, pad_func = grid, None
        if zero_pad != 1.0 and isinstance(grid, SignalGrid):
            pad_func = zero_pad_func(grid, zero_pad)
            pad_grid = zero_pad * grid
        
        i0, pspec = prior_model(f'{prefix} i0 ', pad_grid, **i0)

        if func:
            func = getattr(jnp, func, None)

        if gaussian != None and isinstance(grid, SignalGrid):
            gaussian, _ = prior_model(f'{prefix} gm ', grid, **gaussian)

        return cls(grid, i0, offset, prefix, func, pad_func, gaussian, pspec)
    
    def set_out_grid(self, out_grid):
        check_type(out_grid, SignalGrid)
        self.map_function = map_signal(self.grid, out_grid)
        return

    @property
    def shape(self):
        return self.grid.shape
    
    def set_offset(self, offset):
        '''
        Set the offset for the signal model.

        Parameters
        ----------
        offset : float
            Offset to add to the signal model
        '''
        self.offset = to_shape(offset, (), 'float64')
        return
    
    def copy(self):
        return SignalModel(self.grid, self.i0, self.offset, self.prefix, self.func, self.zero_pad, self.gaussian, self.pspec)



def zero_pad_func(grid, zero_pad=1):
    '''Zero pad the signal with the given factor.'''
    if zero_pad == 1:
        return None
    elif not 1 < zero_pad <= 2:
        raise ValueError('zero_pad must be between 1 and 2')
    
    pad_grid = zero_pad * grid
    pad_slice = tuple(slice((os-ss)//2, ss+(os-ss)//2) for os,ss in zip(pad_grid.shape, grid.shape))
    return lambda x: x[pad_slice]

import dataclasses
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from nifty.re import Model, VModel, Vector
from typing import Any, Callable

from .grid import SignalGrid, PointGrid
from .map import map_signal
from .prior import prior_model
from .spectral import spectral_model
from .util import check_type, to_shape, extend_shape
from ..optimize.samples import domain_tree, model_init



class SignalModel(Model):
    '''Generate a signal model. Use `build` function to create the model.'''

    mask: Any = dataclasses.field(default=None, metadata=dict(static=False))

    def __init__(self, grid, freq, model, prefix='sm', offset=0, nonlinearity=jnp.exp, gaussian=None):
        check_type(grid, (SignalGrid, PointGrid))
        check_type(freq, np.ndarray)
        check_type(model, (Model, VModel))
        check_type(prefix, str)
        check_type(offset, ArrayLike)
        check_type(nonlinearity, (Callable, type(None)))
        check_type(gaussian, (Model, type(None)))

        self.grid = grid
        self.freq = freq
        self.model = model
        self.prefix = prefix
        self.offset = offset
        self.nonlinearity = nonlinearity
        self.gaussian = gaussian
        self.map_function = lambda x: x
        super().__init__(
            domain = Vector(domain_tree((self.model, self.gaussian), error=False)), 
            init = model_init((self.model, self.gaussian), error=False),
        )

    def __call__(self, x, *, map=False):
        res = self.model(x)
        if self.nonlinearity:
            res *= self.nonlinearity(self.offset)
        else:
            res += self.offset
        if self.gaussian:
            gsm = self.gaussian(x)
            res *= gsm[None] if res.ndim == gsm.ndim + 1 else gsm
        if isinstance(self.mask, ArrayLike):
            res = jnp.where(self.mask, res, 0.0)
        if map:
            res = self.map_function(res)
        return res

    @classmethod
    def build(cls, *, grid, freq=[1.], params, prefix='sm', offset=0, nonlinearity='exp', gaussian=None):
        '''
        Build a SignalModel from the given parameters.

        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid)
        freq : list or np.ndarray or Observation
            Frequencies of the signal model. If an Observation is given, the frequencies are extracted from it, by default '[1.]'
        params : dict
            Dictionary containing the spectral model parameters of the signal (see spectral_model)
        prefix : str, optional
            Prefix for the model, by default 'sm'
        offset : float, optional
            Offset to add to the signal, by default 0
        nonlinearity : str, optional
            Function to apply to the signal, by default 'exp'
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters (see gaussian_model), by default None
            -> multiply the signal with a gaussian
        '''
        from ..resolve.observation import Observation

        if 'coordinates' in grid:
            grid = PointGrid.build(**grid)
        else:
            grid = SignalGrid.build(**grid)

        if isinstance(freq, Observation):
            freq = freq.freq
        freq = to_shape(freq, (len(freq),), 'float64')

        if nonlinearity:
            nonlinearity = getattr(jnp, nonlinearity, None)

        model = spectral_model(f'{prefix} ', grid, freq, nonlinearity, **params)

        offset = to_shape(offset, (), 'float64')

        if gaussian != None and isinstance(grid, SignalGrid):
            gaussian, _ = prior_model(f'{prefix} gm ', grid, **gaussian)

        return cls(grid, freq, model, prefix, offset, nonlinearity, gaussian)
    
    def set_out_grid(self, out_grid):
        check_type(out_grid, SignalGrid)
        self.map_function = map_signal(self.grid, out_grid)
        return

    @property
    def shape(self):
        return extend_shape(1, self.freq, self.grid.shape)
    
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
        return SignalModel(self.grid, self.freq, self.model, self.prefix, self.offset, self.nonlinearity, self.gaussian)
    
    @property
    def spectral_index(self):
        '''Return the spectral index model.'''
        if self.freq.size > 1:
            model = self.model
            n_copies = 1
            if isinstance(model, VModel):
                n_copies = model.target.shape[0]
                model = model.model
            if hasattr(model, 'alpha'):
                alpha = model.alpha
            else:
                alpha = Model(lambda x: model.spectral_index_distribution(x), domain=model.domain, init=model.init)
            if n_copies > 1:
                alpha = VModel(alpha, n_copies)
            return SignalModel(self.grid, self.freq, alpha, self.prefix, 0, None, None)
        else:
            raise ValueError('Spectral index is only defined for multi-frequency models.')
    
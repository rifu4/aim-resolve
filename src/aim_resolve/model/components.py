import jax.numpy as jnp
import numpy as np
from itertools import product
from nifty.re import Model, Vector

from .points import PointModel
from .signal import SignalModel
from .grid import SignalGrid
from .tiles import TileModel
from .util import check_type, extend_shape
from ..optimize.samples import domain_keys, domain_tree, model_init



class ComponentModel(Model):
    '''Generate a component model. Use `build` function to create the model.'''

    def __init__(self, grid, background, prefix='cm', *components):
        models = (background, ) + components
        check_type(grid, SignalGrid)
        check_type(prefix, str)
        [check_type(m, (SignalModel, PointModel, TileModel)) for m in models]
        [check_type(m.grid, SignalGrid) for m in models]

        self.grid = grid
        self.freq = models[0].freq
        self.models = models
        self.prefix = prefix
        self.set_out_grid(grid)
        super().__init__(
            domain = Vector(domain_tree(self.models)), 
            init = model_init(self.models),
        )

    def __call__(self, x, nans=False):
        res = jnp.zeros(self.out_shape)
        for m in self.models:
            res += m(x, map=True)
        if nans:
            res = jnp.where(self.mask, res, jnp.nan)
        return res
    
    @classmethod
    def build(cls, *, background, prefix='cm', **components):
        '''
        Build a ComponentModel from the given parameters.
        
        Parameters
        ----------
        background : SignalModel
            Model for the background signal 
        prefix : str, optional
            Prefix for the model, by default 'cm'
        components : keyword arguments
            Key/Value pairs containing the component models ({'key': model})
        '''
        models = (background, ) + tuple(components.values())
        check_type(background, SignalModel)
        check_type(prefix, str)
        [check_type(m, (SignalModel, PointModel, TileModel)) for m in models]
        [check_type(m.grid, SignalGrid) for m in models]

        for (i,mi), (j,mj) in product(enumerate(models), enumerate(models)):
            if i != j and domain_keys(mi) == domain_keys(mj):
                raise ValueError(f'Two models have the same prefix `{mi.prefix}`.')
            if i != j and np.any(mi.freq != mj.freq):
                raise ValueError(f'Two models have different frequencies: `{mi.prefix}` and `{mj.prefix}`.')
            #TODO: ensure that ref_freq_indices are the same

        if len(models) == 1:
            grid = background.grid
        else:
            factor = max([m.grid.factor for m in models])
            grid = background.grid.refine(factor // background.grid.factor)
        
        return cls(grid, background, prefix, *models[1:])
    
    def set_out_grid(self, out_grid):
        check_type(out_grid, SignalGrid)
        self.out_grid = out_grid
        self.out_shape = extend_shape(1, self.freq, self.out_grid.shape)
        for m in self.models:
            m.set_out_grid(out_grid)
        return
    
    @property
    def shape(self):
        return extend_shape(1, self.freq, self.out_grid.shape)
    
    @property
    def mask(self):
        res = np.zeros(self.out_shape)
        for m in self.models:
            res += m.map_function(np.ones(m.shape))
        return res > 0
    
    def copy(self):
        return ComponentModel(self.grid, self.background, self.prefix, *self.components)

    @property
    def background(self):
        return self.models[0]

    @property
    def components(self):
        return self.models[1:]

    @property
    def objects(self):
        return tuple(c for c in self.components if isinstance(c, SignalModel))
    
    @property
    def points(self):
        return tuple(c for c in self.components if isinstance(c, PointModel))
    
    @property
    def tiles(self):
        return tuple(c for c in self.components if isinstance(c, TileModel))
    
    @property
    def signals(self):
        return (self.background, ) + self.objects
    
    @property
    def diffuse(self):
        return ComponentModel(self.grid, self.background, self.prefix, *self.objects)
    
    @property
    def separate(self):
        return (self.diffuse, ) + self.points
    
    @property
    def points_and_objects(self):
        return ComponentModel(self.grid, self.components[0], self.prefix, *self.components[1:])
    
    @property
    def ref_freq_model(self):
        '''Return the reference frequency model.'''
        return self._spectral_property('ref_freq_model')

    @property
    def spectral_index(self):
        '''Return the spectral index model.'''
        return self._spectral_property('spectral_index')

    @property
    def spectral_deviations(self):
        '''Return the spectral deviations model.'''
        return self._spectral_property('spectral_deviations')

    @property
    def spectral_model(self):
        '''Return the spectral model.'''
        return self._spectral_property('spectral_model')

    def _spectral_property(self, attr):
        '''Helper function to create spectral properties.'''
        models = []
        for m in self.models:
            models += [getattr(m, attr)]
        return ComponentModel(self.grid, models[0], self.prefix, *models[1:])

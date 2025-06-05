import jax.numpy as jnp
from itertools import product
from nifty8.re import Model, Vector

from .points import PointModel
from .signal import SignalModel
from .space import SignalSpace
from .tiles import TileModel
from .util import check_type
from ..optimize.samples import domain_keys, domain_tree, model_init



class ComponentModel(Model):
    '''Generate a component model. Use `build` function to create the model.'''

    def __init__(self, space, background, prefix='cm', *components):
        models = (background, ) + components
        check_type(space, SignalSpace)
        check_type(background, SignalModel)
        check_type(prefix, str)
        [check_type(m, (SignalModel, PointModel, TileModel)) for m in models]
        [check_type(m.space, SignalSpace) for m in models]

        self.space = space
        self.prefix = prefix
        self.background = background
        self.components = components
        self.models = models
        super().__init__(
            domain = Vector(domain_tree(self.models)), 
            init = model_init(self.models),
        )

    def __call__(self, x, *, out_space=None):
        out_space = out_space if out_space else self.space
        res = jnp.zeros(out_space.shape)
        #TODO: speed up the for loop with jax
        for m in self.models:
            res += m(x, out_space=out_space)
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
        [check_type(m.space, SignalSpace) for m in models]

        for (i,mi), (j,mj) in product(enumerate(models), enumerate(models)):
            if i != j and domain_keys(mi) == domain_keys(mj):
                raise ValueError(f'Two models have the same prefix `{mi.prefix}`.')

        if len(models) == 1:
            space = background.space
        else:
            bg_space = background.space
            distances = min([m.space.distances for m in models])
            shape = tuple(int(fi/di) for fi,di in zip(bg_space.fov, distances))
            space = SignalSpace(shape, distances, center=bg_space.center, rotation=bg_space.rotation)
        
        return cls(space, background, prefix, *models[1:])

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
        return ComponentModel(self.space, self.background, self.prefix, *self.objects)
    
    @property
    def separate(self):
        return (self.diffuse, ) + self.points

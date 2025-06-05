import jax.numpy as jnp
from jax import random
from nifty8.re import Model, Vector

from .background import BackgroundGenerator
from .objects import ObjectGenerator
from .points import PointGenerator
from .tiles import TileGenerator
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init



class ComponentGenerator(Model):
    '''Generate a component model. Use `build` function to create the model.'''

    def __init__(self, background, points=None, tiles=None, objects=None):
        check_type(background, BackgroundGenerator)
        check_type(points, (PointGenerator, type(None)))
        check_type(tiles, (TileGenerator, type(None)))
        check_type(objects, (ObjectGenerator, type(None)))

        self.space = background.space
        self.background = background
        self.points = points
        self.tiles = tiles
        self.objects = objects
        super().__init__(
            domain = Vector(domain_tree((self.background, self.points, self.tiles, self.objects), error=False)), 
            init = model_init((self.background, self.points, self.tiles, self.objects), error=False),
        )

    def __call__(self, x, *, key=random.PRNGKey(0)):
        val = self.background(x)

        if self.points:
            val += self.points(x, key=key)
        
        if self.tiles:
            val += self.tiles(x)
        
        if self.objects:
            val += self.objects(x, key=key)

        val = val.at[1:].set(jnp.clip(val[1:], 0, 1))
        
        return val
    
    @classmethod
    def build(cls, *, space, background, points=None, tiles=None, objects=None, func='exp'):
        '''
        Build a component generator model.

        Parameters
        ----------
        space : dict
            Dictionary containing the signal space parameters (see SignalSpace)
        background : dict
            Dictionary containing the background model parameters (see BackgroundGenerator)
        points : dict, optional 
            Dictionary containing the point model parameters (see PointGenerator), by default None
        tiles : dict, optional
            Dictionary containing the tile model parameters (see TileGenerator), by default None
        objects : dict, optional
            Dictionary containing the object model parameters (see ObjectGenerator), by default None
        func : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        background = BackgroundGenerator.build(space=space, func=func, **background)

        if points:
            points = PointGenerator.build(space=space, func=func, **points)

        if tiles:
            tiles = TileGenerator.build(space=space, func=func, **tiles)
        
        if objects:
            objects = ObjectGenerator.build(space=space, func=func, **objects)

        return cls(background, points, tiles, objects)

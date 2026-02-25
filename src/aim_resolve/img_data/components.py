"""Composite component generator combining background, points, tiles and objects."""

import jax.numpy as jnp
from jax import random
from nifty.re import Model, Vector

from .background import BackgroundGenerator
from .objects import ObjectGenerator
from .points import PointGenerator
from .tiles import TileGenerator
from ..model.grid import SignalGrid
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init



class ComponentGenerator(Model):
    """Generative model combining background, point, tile and object components.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    grid : SignalGrid
        Spatial grid shared by all components.
    background : BackgroundGenerator
        Background component model.
    points : PointGenerator or None
        Optional point-source component model.
    tiles : TileGenerator or None
        Optional tile component model.
    objects : ObjectGenerator or None
        Optional extended-object component model.
    """

    def __init__(self, grid, background, points=None, tiles=None, objects=None):
        check_type(grid, SignalGrid)
        check_type(background, BackgroundGenerator)
        check_type(points, (PointGenerator, type(None)))
        check_type(tiles, (TileGenerator, type(None)))
        check_type(objects, (ObjectGenerator, type(None)))

        self.grid = grid
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
    def build(cls, *, grid, background, points=None, tiles=None, objects=None, func='exp'):
        """Build a composite component generator from configuration.

        Parameters
        ----------
        grid : dict
            Signal grid parameters (see ``SignalGrid``).
        background : dict
            Background model parameters (see ``BackgroundGenerator``).
        points : dict or None, optional
            Point-source model parameters. Default is None.
        tiles : dict or None, optional
            Tile model parameters. Default is None.
        objects : dict or None, optional
            Extended-object model parameters. Default is None.
        func : str or None, optional
            Name of a ``jax.numpy`` activation function applied to all
            sub-models. Default is ``'exp'``.

        Returns
        -------
        ComponentGenerator
            The constructed composite component generator.
        """
        background = BackgroundGenerator.build(grid=grid, func=func, **background)

        if points:
            points = PointGenerator.build(grid=grid, func=func, **points)

        if tiles:
            tiles = TileGenerator.build(grid=grid, func=func, **tiles)
        
        if objects:
            objects = ObjectGenerator.build(grid=grid, func=func, **objects)

        grid = SignalGrid.build(**grid)

        return cls(grid, background, points, tiles, objects)

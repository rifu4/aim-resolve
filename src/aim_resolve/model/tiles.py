import jax.numpy as jnp
from nifty8.re import Model, VModel, Vector

from .map import map_tiles
from .prior import prior_model
from .signal import SignalModel
from .grid import SignalGrid
from .util import check_type, to_shape
from ..optimize.samples import domain_tree, model_init



class TileModel(Model):
    '''Generate a tile model. Use `build` function to create the model.'''

    def __init__(self, grid, tiles, prefix='tm', gaussian=None):
        check_type(grid, SignalGrid)
        check_type(tiles, SignalModel)
        check_type(tiles.grid, SignalGrid)
        check_type(prefix, str)
        check_type(gaussian, (Model, VModel, type(None)))

        self.grid = grid
        self.tiles = tiles
        self.prefix = prefix
        self.gaussian = gaussian
        super().__init__(
            domain = Vector(domain_tree((self.tiles, self.gaussian), error=False)), 
            init = model_init((self.tiles, self.gaussian), error=False),
        )

    def __call__(self, x, *, out_grid=None):
        out_grid = out_grid if out_grid else self.grid
        res = self.tiles(x)
        if self.gaussian:
            res *= self.gaussian(x)
        return map_tiles(self.tiles.grid, out_grid)(res)

    @classmethod
    def build(cls, *, grid, tile_grid, i0, offset=0, prefix='tm', func='exp', gaussian=None):
        '''
        Build a TileModel from the given parameters.
        
        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid)
        tile_grid : dict
            Dictionary containing the tile grid parameters (see SignalGrid, n_copies > 1)
        i0 : dict
            Dictionary containing the prior model parameters (see prior_model)
        offset : float or list of floats, optional
            Offsets for the individual tile signals, by default '0'
        prefix : str, optional
            Prefix for the model, by default 'ps'
        func : str, optional
            Function to apply to the signal, by default 'exp'
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters (see gaussian_model)
        '''
        tile_grid = SignalGrid.build(**tile_grid)

        grid = SignalGrid.build(**{'factor': tile_grid.factor} | grid)

        i0, pspec = prior_model(f'{prefix} i0 ', tile_grid, tile_grid.n_copies, **i0)

        offset_shape = (tile_grid.n_copies, 1, 1) if tile_grid.n_copies > 1 else (1, 1)
        offset = to_shape(offset, offset_shape, 'float64')

        check_type(prefix, str)

        if func:
            func = getattr(jnp, func, None)

        #TODO: maybe add possibility to vmap over different covs (similar to mean/std in normal model)
        if gaussian != None and isinstance(grid, SignalGrid):
            gaussian, _ = prior_model(f'{prefix} gm ', tile_grid, tile_grid.n_copies, **gaussian)

        #TODO: maybe add mask operation (cut out some part of the signal)
        tiles = SignalModel(tile_grid, i0, offset, prefix, func, pspec=pspec)

        return cls(grid, tiles, prefix, gaussian)
    
    @property
    def shape(self):
        return (self.tiles.grid.n_copies, ) + self.tiles.grid.shape
    
    def set_offset(self, offset):
        '''
        Set the offset for the tile model.
        
        Parameters
        ----------
        offset : float or list of floats
            Offsets for the individual tile signals
        '''
        offset_shape = (self.tiles.grid.n_copies, 1, 1) if self.tiles.grid.n_copies > 1 else (1, 1)
        self.tiles.offset = to_shape(offset, offset_shape, 'float64')
        return

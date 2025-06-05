import jax.numpy as jnp
from nifty8.re import Model, VModel, Vector

from .map import map_signal
from .prior import prior_model
from .signal import SignalModel
from .space import SignalSpace
from .util import check_type, to_shape
from ..optimize.samples import domain_tree, model_init



class TileModel(Model):
    '''Generate a tile model. Use `build` function to create the model.'''

    def __init__(self, space, tiles, prefix='tm', gaussian=None, n_copies=1):
        check_type(space, SignalSpace)
        check_type(tiles, SignalModel)
        check_type(tiles.space, SignalSpace)
        check_type(prefix, str)
        check_type(gaussian, (Model, VModel, type(None)))
        check_type(n_copies, int)

        self.space = space
        self.tiles = tiles
        self.prefix = prefix
        self.gaussian = gaussian
        self.n_copies = n_copies
        super().__init__(
            domain = Vector(domain_tree((self.tiles, self.gaussian), error=False)), 
            init = model_init((self.tiles, self.gaussian), error=False),
        )

    def __call__(self, x, *, out_space=None):
        out_space = out_space if out_space else self.space
        res = self.tiles(x)
        if self.gaussian:
            res *= self.gaussian(x)
        return map_signal(res, self.tiles.space, out_space)

    @classmethod
    def build(cls, *, space, tile_spaces, i0, offset=0, n_copies=1, prefix='tm', func='exp', gaussian=None):
        '''
        Build a TileModel from the given parameters.
        
        Parameters
        ----------
        space : dict
            Dictionary containing the signal space parameters (see SignalSpace)
        tile_spaces : dict
            Dictionary containing the tile space parameters (see SignalSpace, n_copies > 1)
        i0 : dict
            Dictionary containing the prior model parameters (see prior_model)
        offset : float or list of floats, optional
            Offsets for the individual tile signals, by default '0'
        n_copies : int, optional
            Number of point sources, by default 1
        prefix : str, optional
            Prefix for the model, by default 'ps'
        func : str, optional
            Function to apply to the signal, by default 'exp'
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters (see gaussian_model)
        '''
        space = SignalSpace.build(**space)

        tile_space = SignalSpace.build(**tile_spaces, n_copies=n_copies)

        i0, pspec = prior_model(f'{prefix} i0 ', tile_space, n_copies, **i0)

        offset_shape = (n_copies, 1, 1) if n_copies > 1 else (1, 1)
        offset = to_shape(offset, offset_shape, 'float64')

        check_type(prefix, str)

        if func:
            func = getattr(jnp, func, None)

        #TODO: maybe add possibility to vmap over different covs (similar to mean/std in normal model)
        if gaussian != None and isinstance(space, SignalSpace):
            gaussian, _ = prior_model(f'{prefix} gm ', tile_space, n_copies, **gaussian)

        #TODO: maybe add mask operation (cut out some part of the signal)
        tiles = SignalModel(tile_space, i0, offset, prefix, func, pspec=pspec)

        return cls(space, tiles, prefix, gaussian, n_copies)
    
    @property
    def shape(self):
        return (self.n_copies, ) + self.tiles.space.shape
    
    def set_offset(self, offset):
        '''
        Set the offset for the tile model.
        
        Parameters
        ----------
        offset : float or list of floats
            Offsets for the individual tile signals
        '''
        offset_shape = (self.n_copies, 1, 1) if self.n_copies > 1 else (1, 1)
        self.tiles.offset = to_shape(offset, offset_shape, 'float64')
        return

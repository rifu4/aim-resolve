import jax.numpy as jnp
from nifty8.re import Model, Vector, VModel
from typing import Callable

from ..model.integer import integer_model
from ..model.map import map_tiles
from ..model.prior import prior_model, uniform_model
from ..model.space import SignalSpace
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init



class TileGenerator(Model):
    '''Generate a tile model. Use `build` function to create the model.'''

    def __init__(self, space, i0, centers, n_copies, gaussian=None, func=jnp.exp):
        check_type(space, SignalSpace)
        check_type(i0, (Model, VModel))
        check_type(centers, Model)
        check_type(n_copies, Model)
        check_type(gaussian, (Model, VModel, type(None)))
        check_type(func, (Callable, type(None)))

        self.space = space
        self.i0 = i0
        self.centers = centers
        self.n_copies = n_copies
        self.gaussian = gaussian
        self.func = func
        super().__init__(
            domain = Vector(domain_tree((self.i0, self.centers, self.n_copies, self.gaussian), error=False)), 
            init = model_init((self.i0, self.centers, self.n_copies, self.gaussian), error=False),
        )

    def __call__(self, x):
        i0_val = self.i0(x)
        nc_val = self.n_copies(x)
        nc_mask = jnp.arange(i0_val.shape[0]) < nc_val[0]
        nc_mask = nc_mask.reshape(-1, 1, 1)

        x_val = i0_val * nc_mask
        y_val = jnp.ones(x_val.shape) * nc_mask

        if self.func:
            x_val = self.func(x_val)

        if self.gaussian:
            gm_val = self.gaussian(x)
            gm_max = jnp.max(gm_val, axis=(1, 2)).reshape(-1, 1, 1)
            gm_val /= jnp.where(gm_max > 0, gm_max, 1)
            x_val *= gm_val
            y_val *= gm_val

        y_val = jnp.where(y_val > 0.1, 1, 0)

        rot = jnp.zeros(i0_val.shape[0])
        x_val = map_tiles(x_val, self.space.dis, self.centers(x), rot, self.space, i0_val.shape[0])
        y_val = map_tiles(y_val, self.space.dis, self.centers(x), rot, self.space, i0_val.shape[0])
        
        return jnp.stack((x_val, jnp.zeros(self.space.shape), y_val), axis=0)

    @classmethod
    def build(cls, *, n_min=0, n_max=0, space, tile_size, i0, gaussian=None, func='exp'):
        '''
        Build a tile generator model.

        Parameters
        ----------
        n_min : int
            Minimum number of tiles to generate
        n_max : int
            Maximum number of tiles to generate
        space : dict
            Dictionary containing the signal space parameters (see SignalSpace)
        tile_size : tuple
            Size of the tile components in pixels
        i0 : dict
            Dictionary containing the prior model parameters of the signal (see prior_model)
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters (see gaussian_model), by default None
            -> multiply the tile components with a gaussian
        func : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        check_type(n_min, int)
        check_type(n_max, int)
 
        space = SignalSpace.build(**space)

        tile_space = SignalSpace.build(
            shape = tile_size,
            distances = space.distances,
            n_copies = max(n_max, 2)
        )
        i0, _ = prior_model('tg i0 ', tile_space, max(n_max, 2), **i0)

        centers = uniform_model(
            prefix = 'tg centers',
            shape = (max(n_max, 2), 2),
            u_min = space.limits[0,0] - tile_space.limits[0,0,0],
            u_max = space.limits[1,1] - tile_space.limits[1,1,1],
        )
        n_copies = integer_model(
            prefix = 'tg n copies',
            shape = (1,),
            i_min = n_min,
            i_max = n_max + 1,
        )
        if gaussian:
            gaussian, _ = prior_model('tg gm ', tile_space, max(n_max, 2), **gaussian)

        if func:
            func = getattr(jnp, func, None)

        return cls(space, i0, centers, n_copies, gaussian, func)

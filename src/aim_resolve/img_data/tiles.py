"""Tile component generator model for synthetic sky images."""

import jax.numpy as jnp
from nifty.re import Model, Vector, VModel
from typing import Callable

from ..model.grid import SignalGrid
from ..model.integer import integer_model
from ..model.map import map_array
from ..model.prior import prior_model
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init



class TileGenerator(Model):
    """Generative model for tile-based extended components.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    grid : SignalGrid
        Spatial grid of the output image.
    i0 : Model or VModel
        Prior model for tile intensities.
    centers : Model
        Model producing integer tile centre coordinates.
    n_copies : Model
        Model producing the number of active tiles.
    gaussian : Model, VModel or None
        Optional Gaussian envelope per tile.
    func : callable or None
        Point-wise activation function.
    """

    def __init__(self, grid, i0, centers, n_copies, gaussian=None, func=jnp.exp):
        check_type(grid, SignalGrid)
        check_type(i0, (Model, VModel))
        check_type(centers, Model)
        check_type(n_copies, Model)
        check_type(gaussian, (Model, VModel, type(None)))
        check_type(func, (Callable, type(None)))

        self.grid = grid
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

        n_copies = nc_mask.shape[0]
        in_shape = x_val.shape[-2:]
        out_shape = self.grid.shape
        out_start = self.centers(x)
        in_start = jnp.zeros_like(out_start, dtype='int32')

        x_val = map_array(x_val, n_copies, 1, in_shape, out_shape, in_start, out_start, 1)
        y_val = map_array(y_val, n_copies, 1, in_shape, out_shape, in_start, out_start, 1)
        
        return jnp.stack((x_val, jnp.zeros(self.grid.shape), y_val), axis=0)

    @classmethod
    def build(cls, *, n_min=0, n_max=0, grid, tile_size, i0, gaussian=None, func='exp'):
        """Build a tile generator from configuration dictionaries.

        Parameters
        ----------
        n_min : int, optional
            Minimum number of tiles. Default is 0.
        n_max : int, optional
            Maximum number of tiles. Default is 0.
        grid : dict
            Signal grid parameters (see ``SignalGrid``).
        tile_size : tuple of int
            ``(height, width)`` of each tile in pixels.
        i0 : dict
            Prior model parameters for tile intensities.
        gaussian : dict or None, optional
            Gaussian envelope parameters. Default is None.
        func : str or None, optional
            ``jax.numpy`` activation function name. Default is ``'exp'``.

        Returns
        -------
        TileGenerator
            The constructed tile generator.
        """
        check_type(n_min, int)
        check_type(n_max, int)
 
        grid = SignalGrid.build(**grid)

        tile_grid = SignalGrid.build(
            space = tile_size,
            distances = grid.distances,
            n_copies = max(n_max, 2)
        )
        i0, _ = prior_model('tg i0 ', tile_grid, max(n_max, 2), **i0)

        centers = integer_model(
            prefix = 'tg centers',
            shape = (max(n_max, 2), 2),
            i_min = 0,
            i_max = grid.shape[0] - tile_size[0],
        )
        n_copies = integer_model(
            prefix = 'tg n copies',
            shape = (1,),
            i_min = n_min,
            i_max = n_max + 1,
        )
        if gaussian:
            gaussian, _ = prior_model('tg gm ', tile_grid, max(n_max, 2), **gaussian)

        if func:
            func = getattr(jnp, func, None)

        return cls(grid, i0, centers, n_copies, gaussian, func)

"""Tile-based signal model for AIM-Resolve."""

import jax.numpy as jnp
import numpy as np
from nifty.re import Model, Vector, VModel

from ..optimize.samples import domain_tree, model_init
from .grid import SignalGrid
from .map import map_signal
from .prior import prior_model
from .signal import SignalModel
from .spectral import spectral_model
from .util import check_type, extend_shape, to_shape


class TileModel(Model):
    """Tile-based signal model composed of smaller tile sub-models.

    Generate a tile model that maps a signal defined on a coarse tile grid
    onto a finer output grid. Use the `build` class method to create an
    instance from configuration dictionaries.
    """

    def __init__(self, grid, freq, tiles, prefix="tm", gaussian=None):
        check_type(grid, SignalGrid)
        check_type(freq, np.ndarray)
        check_type(tiles, SignalModel)
        check_type(tiles.grid, SignalGrid)
        check_type(prefix, str)
        check_type(gaussian, (Model, VModel, type(None)))

        self.grid = grid
        self.freq = freq
        self.tiles = tiles
        self.prefix = prefix
        self.gaussian = gaussian
        self.set_out_grid(grid)
        super().__init__(
            domain=Vector(domain_tree((self.tiles, self.gaussian), error=False)),
            init=model_init((self.tiles, self.gaussian), error=False),
        )

    def __call__(self, x, *, map=True, nans=False):
        """Evaluate the tile model.

        Parameters
        ----------
        x : dict
            Latent parameters for the model.
        map : bool, optional
            Whether to map the tile signal onto the output grid,
            by default True.
        nans : bool, optional
            Whether to replace masked pixels with NaN,
            by default False.

        Returns
        -------
        jnp.ndarray
            Evaluated signal on the output grid.
        """
        res = self.tiles(x)
        if self.gaussian:
            gsm = self.gaussian(x)
            res *= gsm[:, None] if res.ndim == gsm.ndim + 1 else gsm
        if map:
            res = self.map_function(res)
        if nans:
            res = jnp.where(self.mask, res, jnp.nan)
        return res

    @classmethod
    def build(
        cls,
        *,
        grid,
        tile_grid,
        freq=None,
        params,
        prefix="tm",
        offset=0,
        nonlinearity="exp",
        gaussian=None,
    ):
        """Build a TileModel from the given parameters.

        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid).
        tile_grid : dict
            Dictionary containing the tile grid parameters
            (see SignalGrid, n_copies > 1).
        freq : list or np.ndarray or Observation
            Frequencies of the signal model. If an Observation is given,
            the frequencies are extracted from it, by default ``[1.]``.
        params : dict
            Dictionary containing the spectral model parameters of the
            signal (see spectral_model).
        prefix : str, optional
            Prefix for the model, by default ``'tm'``.
        offset : float or list of floats, optional
            Offsets for the individual tile signals, by default ``0``.
        nonlinearity : str, optional
            Function to apply to the signal, by default ``'exp'``.
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters
            (see gaussian_model).
        """
        if freq is None:
            freq = [1.0]
        from ..resolve.observation import Observation

        tile_grid = SignalGrid.build(**tile_grid)

        grid = SignalGrid.build(**{"factor": tile_grid.factor} | grid)

        if isinstance(freq, Observation):
            freq = freq.freq
        freq = to_shape(freq, (len(freq),), "float64")

        if nonlinearity:
            nonlinearity = getattr(jnp, nonlinearity, None)

        model_grid = SignalGrid.build(space=tile_grid.shape)
        model = spectral_model(
            f"{prefix} ", model_grid, freq, nonlinearity, tile_grid.n_copies, **params
        )

        offset_shape = extend_shape(tile_grid.n_copies, freq, (1, 1), offset=True)
        offset = to_shape(offset, offset_shape, "float64")

        if gaussian is not None and isinstance(grid, SignalGrid):
            gaussian, _ = prior_model(
                f"{prefix} gm ", tile_grid, tile_grid.n_copies, **gaussian
            )

        tiles = SignalModel(tile_grid, freq, model, prefix, offset, nonlinearity)

        return cls(grid, freq, tiles, prefix, gaussian)

    def set_out_grid(self, out_grid):
        """Set the output grid and update the mapping function.

        Parameters
        ----------
        out_grid : SignalGrid
            The output grid onto which the tile signal is mapped.
        """
        check_type(out_grid, SignalGrid)
        self.map_function = map_signal(self.tiles.grid, out_grid)
        return

    @property
    def shape(self):
        """Full output shape including copies and frequencies."""
        return extend_shape(self.tiles.grid.n_copies, self.freq, self.tiles.grid.shape)

    @property
    def n_copies(self):
        """Number of tile copies."""
        return self.tiles.grid.n_copies

    @property
    def mask(self):
        """Boolean mask indicating valid (non-zero) pixels."""
        res = self.map_function(np.ones(self.tiles.target.shape))
        return res > 0

    def set_offset(self, offset):
        """Set the offset for the tile model.

        Parameters
        ----------
        offset : float or list of floats
            Offsets for the individual tile signals.
        """
        offset_shape = extend_shape(
            self.tiles.grid.n_copies, self.freq, (1, 1), offset=True
        )
        self.tiles.offset = to_shape(offset, offset_shape, "float64")
        return

    def copy(self):
        """Return a shallow copy of the tile model."""
        return TileModel(self.grid, self.freq, self.tiles, self.prefix, self.gaussian)

    @property
    def ref_freq_model(self):
        """Return the reference frequency model."""
        tile = TileModel(
            self.grid,
            np.ones((1,)),
            self.tiles.ref_freq_model,
            self.prefix,
            self.gaussian,
        )
        return tile

    @property
    def spectral_index(self):
        """Return the spectral index model."""
        return TileModel(
            self.grid, np.ones((1,)), self.tiles.spectral_index, self.prefix, None
        )

    @property
    def spectral_deviations(self):
        """Return the spectral deviations model."""
        return TileModel(
            self.grid, self.freq, self.tiles.spectral_deviations, self.prefix, None
        )

    @property
    def spectral_model(self):
        """Return the spectral model."""
        return TileModel(
            self.grid, self.freq, self.tiles.spectral_model, self.prefix, None
        )

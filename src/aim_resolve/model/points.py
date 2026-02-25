"""Point source model for AIM-Resolve."""

import jax.numpy as jnp
import numpy as np
from nifty.re import Model

from .grid import PointGrid, SignalGrid
from .map import map_signal
from .signal import SignalModel
from .spectral import spectral_model
from .util import check_type, extend_shape, to_shape


class PointModel(Model):
    """Generate a point model.

    Use the ``build`` class method to create the model.
    """

    def __init__(self, grid, freq, points, prefix="pm"):
        check_type(grid, SignalGrid)
        check_type(freq, np.ndarray)
        check_type(points, SignalModel)
        check_type(points.grid, PointGrid)
        check_type(prefix, str)

        self.grid = grid
        self.freq = freq
        self.points = points
        self.prefix = prefix
        self.set_out_grid(grid)
        super().__init__(domain=self.points.domain, init=self.points.init)

    def __call__(self, x, *, map=True, nans=False):
        """Evaluate the point model.

        Parameters
        ----------
        x : dict
            Input latent parameters.
        map : bool, optional
            If ``True``, map the result to the output grid. Default is
            ``True``.
        nans : bool, optional
            If ``True``, set masked pixels to NaN. Default is ``False``.

        Returns
        -------
        jnp.ndarray
            The evaluated point model.
        """
        res = self.points(x)
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
        point_grid,
        freq=[1.0],
        params,
        prefix="pm",
        offset=0,
        nonlinearity="exp",
    ):
        """Build a PointModel from the given parameters.

        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see
            ``SignalGrid``).
        point_grid : dict
            Dictionary containing the point grid parameters (see
            ``PointGrid``).
        freq : list or np.ndarray or Observation
            Frequencies of the signal model. If an ``Observation`` is given,
            the frequencies are extracted from it. Default is ``[1.]``.
        params : dict
            Dictionary containing the spectral model parameters of the
            signal (see ``spectral_model``).
        prefix : str, optional
            Prefix for the model. Default is ``'pm'``.
        offset : float or list of floats, optional
            Offsets for the individual point signals. Default is ``0``.
        nonlinearity : str, optional
            Function to apply to the signal. Default is ``'exp'``.
        """
        from ..resolve.observation import Observation

        point_grid = PointGrid.build(**point_grid)

        grid = SignalGrid.build(**{"factor": point_grid.factor} | grid)

        if isinstance(freq, Observation):
            freq = freq.freq
        freq = to_shape(freq, (len(freq),), "float64")

        if nonlinearity:
            nonlinearity = getattr(jnp, nonlinearity, None)

        model_grid = SignalGrid.build(space=point_grid.shape)
        model = spectral_model(
            f"{prefix} ", model_grid, freq, nonlinearity, point_grid.n_copies, **params
        )

        offset_shape = extend_shape(point_grid.n_copies, freq, (1, 1), offset=True)
        offset = to_shape(offset, offset_shape, "float64")

        points = SignalModel(point_grid, freq, model, prefix, offset, nonlinearity)

        return cls(grid, freq, points, prefix)

    def set_out_grid(self, out_grid):
        """Set the output grid and update the map function.

        Parameters
        ----------
        out_grid : SignalGrid
            The output signal grid.
        """
        check_type(out_grid, SignalGrid)
        self.map_function = map_signal(self.points.grid, out_grid)
        return

    @property
    def shape(self):
        """Shape of the point model output."""
        return extend_shape(
            self.points.grid.n_copies, self.freq, self.points.grid.shape
        )

    @property
    def n_copies(self):
        """Number of copies of the point model."""
        return self.points.grid.n_copies

    @property
    def mask(self):
        """Boolean mask indicating valid pixels."""
        res = self.map_function(np.ones(self.points.target.shape))
        return res > 0

    def set_offset(self, offset):
        """Set the offset for the point model.

        Parameters
        ----------
        offset : float or list of floats
            Offsets for the individual point signals.
        """
        offset_shape = extend_shape(
            self.points.grid.n_copies, self.freq, (1, 1), offset=True
        )
        self.points.offset = to_shape(offset, offset_shape, "float64")
        return

    def copy(self):
        """Return a shallow copy of the point model."""
        return PointModel(self.grid, self.freq, self.points, self.prefix)

    @property
    def ref_freq_model(self):
        """Return the reference frequency model."""
        return PointModel(
            self.grid, np.ones((1,)), self.points.ref_freq_model, self.prefix
        )

    @property
    def spectral_index(self):
        """Return the spectral index model."""
        return PointModel(
            self.grid, np.ones((1,)), self.points.spectral_index, self.prefix
        )

    @property
    def spectral_deviations(self):
        """Return the spectral deviations model."""
        return PointModel(
            self.grid, self.freq, self.points.spectral_deviations, self.prefix
        )

    @property
    def spectral_model(self):
        """Return the spectral model."""
        return PointModel(self.grid, self.freq, self.points.spectral_model, self.prefix)

"""Signal model for AIM-Resolve astronomical reconstruction."""

import dataclasses
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from nifty.re import Model, Vector, VModel

from ..optimize.samples import domain_tree, model_init
from .grid import PointGrid, SignalGrid
from .map import map_signal
from .prior import prior_model
from .spectral import MultiFrequencyModel, spectral_model
from .util import check_type, extend_shape, to_shape


class SignalModel(Model):
    """Signal model combining spatial and spectral components.

    Wraps a spectral model with optional nonlinearity, offset, and
    Gaussian modulation. Use the ``build`` class method to construct
    instances from configuration dictionaries.
    """

    mask: Any = dataclasses.field(default=None, metadata=dict(static=False))

    def __init__(
        self,
        grid,
        freq,
        model,
        prefix="sm",
        offset=0,
        nonlinearity=jnp.exp,
        gaussian=None,
    ):
        check_type(grid, (SignalGrid, PointGrid))
        check_type(freq, np.ndarray)
        check_type(model, (Model, VModel))
        check_type(prefix, str)
        check_type(offset, ArrayLike)
        check_type(nonlinearity, (Callable, type(None)))
        check_type(gaussian, (Model, type(None)))

        self.grid = grid
        self.freq = freq
        self.model = model
        self.prefix = prefix
        self.offset = offset
        self.nonlinearity = nonlinearity
        self.gaussian = gaussian
        self.map_function = lambda x: x
        super().__init__(
            domain=Vector(domain_tree((self.model, self.gaussian), error=False)),
            init=model_init((self.model, self.gaussian), error=False),
        )

    def __call__(self, x, *, map=False):
        """Evaluate the signal model.

        Parameters
        ----------
        x : dict
            Latent parameter dictionary.
        map : bool, optional
            If True, apply the map function to the result. Default is False.

        Returns
        -------
        res : jnp.ndarray
            The evaluated signal.
        """
        res = self.model(x)
        if self.nonlinearity:
            res *= self.nonlinearity(self.offset)
        else:
            res += self.offset
        if self.gaussian:
            gsm = self.gaussian(x)
            res *= gsm[None] if res.ndim == gsm.ndim + 1 else gsm
        if isinstance(self.mask, ArrayLike):
            res = jnp.where(self.mask, res, 0.0)
        if map:
            res = self.map_function(res)
        return res

    @classmethod
    def build(
        cls,
        *,
        grid,
        freq=None,
        params,
        prefix="sm",
        offset=0,
        nonlinearity="exp",
        gaussian=None,
    ):
        """
        Build a SignalModel from the given parameters.

        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid).
        freq : list or np.ndarray or Observation
            Frequencies of the signal model. If an Observation is given,
            the frequencies are extracted from it. Default is ``[1.]``.
        params : dict
            Dictionary containing the spectral model parameters of the
            signal (see spectral_model).
        prefix : str, optional
            Prefix for the model. Default is ``'sm'``.
        offset : float, optional
            Offset to add to the signal. Default is 0.
        nonlinearity : str, optional
            Function to apply to the signal. Default is ``'exp'``.
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters
            (see gaussian_model). Multiplies the signal with a gaussian.
            Default is None.
        """
        if freq is None:
            freq = [1.0]
        from ..resolve.observation import Observation

        if "coordinates" in grid:
            grid = PointGrid.build(**grid)
        else:
            grid = SignalGrid.build(**grid)

        if isinstance(freq, Observation):
            freq = freq.freq
        freq = to_shape(freq, (len(freq),), "float64")

        if nonlinearity:
            nonlinearity = getattr(jnp, nonlinearity, None)

        model = spectral_model(f"{prefix} ", grid, freq, nonlinearity, **params)

        offset = to_shape(offset, (), "float64")

        if gaussian is not None and isinstance(grid, SignalGrid):
            gaussian, _ = prior_model(f"{prefix} gm ", grid, **gaussian)

        return cls(grid, freq, model, prefix, offset, nonlinearity, gaussian)

    def set_out_grid(self, out_grid):
        """Set the output grid and configure the map function.

        Parameters
        ----------
        out_grid : SignalGrid
            The output signal grid to map onto.
        """
        check_type(out_grid, SignalGrid)
        self.map_function = map_signal(self.grid, out_grid)
        return

    @property
    def shape(self):
        """Shape of the signal model output array."""
        return extend_shape(1, self.freq, self.grid.shape)

    def set_offset(self, offset):
        """
        Set the offset for the signal model.

        Parameters
        ----------
        offset : float
            Offset to add to the signal model.
        """
        self.offset = to_shape(offset, (), "float64")
        return

    def copy(self):
        """Return a shallow copy of this SignalModel."""
        return SignalModel(
            self.grid,
            self.freq,
            self.model,
            self.prefix,
            self.offset,
            self.nonlinearity,
            self.gaussian,
        )

    @property
    def ref_freq_model(self):
        """Return the reference frequency model."""
        offset = self.offset[:, 0] if self.offset.ndim == 4 else self.offset
        return self._spectral_property(
            "i0",
            "reference_frequency_distribution",
            offset,
            self.nonlinearity,
            self.gaussian,
        )

    @property
    def spectral_index(self):
        """Return the spectral index model."""
        return self._spectral_property(
            "alpha", "spectral_index_distribution", 0, None, None
        )

    @property
    def spectral_deviations(self):
        """Return the spectral deviations model."""
        return self._spectral_property(
            "deviations", "spectral_deviations_distribution", 0, None, None
        )

    @property
    def spectral_model(self):
        """Return the spectral model."""
        return self._spectral_property("", "spectral_distribution", 0, None, None)

    def _spectral_property(
        self, mfm_attr, ubik_attr, offset=0, nonlinearity=None, gaussian=None
    ):
        """Create a SignalModel for a specific spectral property.

        Parameters
        ----------
        mfm_attr : str
            Attribute name on the MultiFrequencyModel (e.g. ``'i0'``,
            ``'alpha'``, ``'deviations'``, or ``''`` for the full model).
        ubik_attr : str
            Attribute name for the ubik-style spectral model fallback.
        offset : float or jnp.ndarray, optional
            Offset applied to the resulting sub-model. Default is 0.
        nonlinearity : callable or None, optional
            Nonlinearity applied to the sub-model. Default is None.
        gaussian : Model or None, optional
            Optional Gaussian modulation model. Default is None.

        Returns
        -------
        sig : SignalModel
            A SignalModel wrapping the requested spectral property.
        """
        if self.freq.size > 1:
            model = self.model
            n_copies = 1
            if isinstance(model, VModel):
                n_copies = model.target.shape[0]
                model = model.model
            if isinstance(model, MultiFrequencyModel):
                if mfm_attr in ("alpha", "deviations"):
                    prop = getattr(model, mfm_attr)
                elif mfm_attr == "i0":
                    prop = MultiFrequencyModel(
                        model.i0, None, None, None, self.nonlinearity
                    )
                else:
                    prop = MultiFrequencyModel(
                        None, model.log_freq, model.alpha, model.deviations, None
                    )
            else:
                prop = Model(
                    lambda x: getattr(model, ubik_attr)(x),
                    domain=model.domain,
                    init=model.init,
                )
            if n_copies > 1:
                prop = VModel(prop, n_copies)
            freq = np.ones((1,)) if mfm_attr in ("i0", "alpha") else self.freq
            sig = SignalModel(
                self.grid, freq, prop, self.prefix, offset, nonlinearity, gaussian
            )
            return sig
        else:
            raise ValueError(
                "spectral properties are only defined for multi-frequency models."
            )

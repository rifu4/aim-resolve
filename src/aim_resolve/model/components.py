"""Component model combining multiple signal, point, and tile models."""

from itertools import product

import jax.numpy as jnp
import numpy as np
from nifty.re import Model, Vector

from ..optimize.samples import domain_keys, domain_tree, model_init
from .grid import SignalGrid
from .points import PointModel
from .signal import SignalModel
from .tiles import TileModel
from .util import check_type, extend_shape


class ComponentModel(Model):
    """Generate a component model. Use `build` function to create the model."""

    def __init__(self, grid, background, prefix="cm", *components):
        models = (background,) + components
        check_type(grid, SignalGrid)
        check_type(prefix, str)
        [check_type(m, (SignalModel, PointModel, TileModel)) for m in models]
        [check_type(m.grid, SignalGrid) for m in models]

        self.grid = grid
        self.freq = models[0].freq
        self.models = models
        self.prefix = prefix
        self.set_out_grid(grid)
        super().__init__(
            domain=Vector(domain_tree(self.models)),
            init=model_init(self.models),
        )

    def __call__(self, x, nans=False):
        """Evaluate the component model.

        Parameters
        ----------
        x : Mapping
            Input latent parameters.
        nans : bool, optional
            If True, set unmasked pixels to NaN. Default is False.

        Returns
        -------
        jnp.ndarray
            Combined model output on the output grid.
        """
        res = jnp.zeros(self.out_shape)
        for m in self.models:
            res += m(x, map=True)
        if nans:
            res = jnp.where(self.mask, res, jnp.nan)
        return res

    @classmethod
    def build(cls, *, background, prefix="cm", **components):
        """
        Build a ComponentModel from the given parameters.

        Parameters
        ----------
        background : SignalModel
            Model for the background signal.
        prefix : str, optional
            Prefix for the model, by default 'cm'.
        components : keyword arguments
            Key/Value pairs containing the component models ({'key': model}).
        """
        models = (background,) + tuple(components.values())
        check_type(background, SignalModel)
        check_type(prefix, str)
        [check_type(m, (SignalModel, PointModel, TileModel)) for m in models]
        [check_type(m.grid, SignalGrid) for m in models]

        for (i, mi), (j, mj) in product(enumerate(models), enumerate(models)):
            if i != j and domain_keys(mi) == domain_keys(mj):
                raise ValueError(f"Two models have the same prefix `{mi.prefix}`.")
            if i != j and np.any(mi.freq != mj.freq):
                raise ValueError(
                    f"Two models have different frequencies: `{mi.prefix}` and `{mj.prefix}`."
                )
            # TODO: ensure that ref_freq_indices are the same

        if len(models) == 1:
            grid = background.grid
        else:
            factor = max([m.grid.factor for m in models])
            grid = background.grid.refine(factor // background.grid.factor)

        return cls(grid, background, prefix, *models[1:])

    def set_out_grid(self, out_grid):
        """Set the output grid for the model and all sub-models.

        Parameters
        ----------
        out_grid : SignalGrid
            The output grid to project onto.
        """
        check_type(out_grid, SignalGrid)
        self.out_grid = out_grid
        self.out_shape = extend_shape(1, self.freq, self.out_grid.shape)
        for m in self.models:
            m.set_out_grid(out_grid)
        return

    @property
    def shape(self):
        """Shape of the output array including frequency and spatial dimensions."""
        return extend_shape(1, self.freq, self.out_grid.shape)

    @property
    def mask(self):
        """Boolean mask indicating pixels covered by at least one sub-model."""
        res = np.zeros(self.out_shape)
        for m in self.models:
            res += m.map_function(np.ones(m.shape))
        return res > 0

    def copy(self):
        """Return a copy of the component model."""
        return ComponentModel(self.grid, self.background, self.prefix, *self.components)

    @property
    def background(self):
        """Background signal model (first model in the list)."""
        return self.models[0]

    @property
    def components(self):
        """Tuple of all non-background component models."""
        return self.models[1:]

    @property
    def objects(self):
        """Tuple of SignalModel components (diffuse objects)."""
        return tuple(c for c in self.components if isinstance(c, SignalModel))

    @property
    def points(self):
        """Tuple of PointModel components."""
        return tuple(c for c in self.components if isinstance(c, PointModel))

    @property
    def tiles(self):
        """Tuple of TileModel components."""
        return tuple(c for c in self.components if isinstance(c, TileModel))

    @property
    def signals(self):
        """Tuple of all SignalModel instances (background + objects)."""
        return (self.background,) + self.objects

    @property
    def diffuse(self):
        """ComponentModel containing only the background and diffuse object models."""
        return ComponentModel(self.grid, self.background, self.prefix, *self.objects)

    @property
    def separate(self):
        """Tuple of the diffuse component model and individual point models."""
        return (self.diffuse,) + self.points

    @property
    def points_and_objects(self):
        """ComponentModel containing only point and object components (no background)."""
        return ComponentModel(
            self.grid, self.components[0], self.prefix, *self.components[1:]
        )

    @property
    def ref_freq_model(self):
        """Return the reference frequency model."""
        return self._spectral_property("ref_freq_model")

    @property
    def spectral_index(self):
        """Return the spectral index model."""
        return self._spectral_property("spectral_index")

    @property
    def spectral_deviations(self):
        """Return the spectral deviations model."""
        return self._spectral_property("spectral_deviations")

    @property
    def spectral_model(self):
        """Return the spectral model."""
        return self._spectral_property("spectral_model")

    def _spectral_property(self, attr):
        """Helper function to create spectral properties.

        Parameters
        ----------
        attr : str
            Name of the spectral attribute to extract from each sub-model.

        Returns
        -------
        ComponentModel
            A new ComponentModel built from the spectral sub-properties.
        """
        models = []
        for m in self.models:
            models += [getattr(m, attr)]
        return ComponentModel(self.grid, models[0], self.prefix, *models[1:])

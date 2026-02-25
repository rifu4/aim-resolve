"""Point-source generator model for synthetic sky images."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import random, vmap
from jax.typing import ArrayLike
from nifty.re import Model, Vector

from ..model.grid import SignalGrid
from ..model.integer import integer_model
from ..model.map import map_array
from ..model.normal import normal_model
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init
from .jax_fun import gaussian_filter2d


class PointGenerator(Model):
    """Generative model for point sources with optional Gaussian blur.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    grid : SignalGrid
        Spatial grid of the output image.
    i0 : Model
        Prior model for point-source intensities.
    coordinates : Model
        Model producing integer pixel coordinates.
    n_copies : Model
        Model producing the number of active sources.
    blur : array_like or None
        Array of blur sigma values for Gaussian smoothing.
    func : callable or None
        Point-wise activation function.
    """

    def __init__(self, grid, i0, coordinates, n_copies, blur=None, func=jnp.exp):
        check_type(grid, SignalGrid)
        check_type(i0, Model)
        check_type(coordinates, Model)
        check_type(n_copies, Model)
        check_type(blur, (ArrayLike, type(None)))
        check_type(func, (Callable, type(None)))

        self.grid = grid
        self.i0 = i0
        self.coordinates = coordinates
        self.n_copies = n_copies
        self.blur = (
            blur if isinstance(blur, ArrayLike) else jnp.zeros(self.i0.target.shape[0])
        )
        self.func = func
        super().__init__(
            domain=Vector(
                domain_tree((self.i0, self.coordinates, self.n_copies), error=False)
            ),
            init=model_init((self.i0, self.coordinates, self.n_copies), error=False),
        )

    def __call__(self, x, *, key=None):
        if key is None:
            key = random.PRNGKey(0)
        i0_val = self.i0(x)
        nc_mask = (jnp.arange(i0_val.shape[0]) < self.n_copies(x)[0]).reshape(-1, 1, 1)

        if self.func:
            i0_val = self.func(i0_val)

        n_copies = nc_mask.shape[0]
        in_shape = (1, 1)
        out_shape = self.grid.shape
        out_start = self.coordinates(x)
        in_start = jnp.zeros_like(out_start, dtype="int32")

        x_val = map_array(
            i0_val * nc_mask,
            n_copies,
            n_copies,
            in_shape,
            out_shape,
            in_start,
            out_start,
            1,
        )
        y_val = map_array(
            nc_mask, n_copies, n_copies, in_shape, out_shape, in_start, out_start, 1
        )

        bl_val = random.permutation(key, self.blur, axis=0)[: i0_val.shape[0]]
        vmap_filter = vmap(gaussian_filter2d, in_axes=(0, 0, None, None))
        x_val = vmap_filter(x_val, bl_val, 2, False)

        x_val = jnp.sum(x_val, axis=0)
        y_val = jnp.sum(y_val, axis=0)

        return jnp.stack((x_val, y_val, jnp.zeros(x_val.shape)), axis=0)

    @classmethod
    def build(cls, *, n_min=0, n_max=0, grid, i0, blur=None, func="exp"):
        """Build a point-source generator from configuration.

        Parameters
        ----------
        n_min : int, optional
            Minimum number of sources. Default is 0.
        n_max : int, optional
            Maximum number of sources. Default is 0.
        grid : dict
            Signal grid parameters (see ``SignalGrid``).
        i0 : dict
            Prior model parameters for source intensities.
        blur : dict or None, optional
            Blur array parameters (see ``get_blur``). Default is None.
        func : str or None, optional
            ``jax.numpy`` activation function name. Default is ``'exp'``.

        Returns
        -------
        PointGenerator
            The constructed point-source generator.
        """
        check_type(n_min, int)
        check_type(n_max, int)

        grid = SignalGrid.build(**grid)

        i0 = normal_model(
            prefix="pg i0 ",
            shape=(n_max, 1, 1),
            **i0,
        )
        coordinates = integer_model(
            prefix="pg coordinates",
            shape=(n_max, 2),
            i_min=0,
            i_max=grid.shape[0],
        )
        n_copies = integer_model(
            prefix="pg n copies",
            shape=(1,),
            i_min=n_min,
            i_max=n_max + 1,
        )
        if blur:
            blur = get_blur(n_max, **blur)

        if func:
            func = getattr(jnp, func, None)

        return cls(grid, i0, coordinates, n_copies, blur, func)


def get_blur(n_max, *, b_min=0, b_max=0, steps=10):
    """Generate linearly spaced blur sigma values.

    Parameters
    ----------
    n_max : int
        Number of point sources.
    b_min : float, optional
        Minimum blur sigma. Default is 0.
    b_max : float, optional
        Maximum blur sigma. Default is 0.
    steps : int, optional
        Minimum number of values to generate. Default is 10.

    Returns
    -------
    jnp.ndarray
        Array of blur sigma values.
    """
    return jnp.linspace(b_min, b_max, max(n_max, steps))

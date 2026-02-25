"""Extended-object generator model for synthetic sky images."""

import os
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import random
from jax.typing import ArrayLike
from nifty.re import Model, Vector

from ..model.grid import SignalGrid
from ..model.map import map_array
from ..model.normal import normal_model
from ..model.prior import uniform_model
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init
from .jax_fun import flip_data, rotate_data


class ObjectGenerator(Model):
    """Generative model for extended objects using randomly rotated masks.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    grid : SignalGrid
        Spatial grid of the output image.
    i0 : Model
        Prior model for the object intensity.
    masks : array_like
        Array of 2-D binary masks.
    zoom : Model or None
        Optional zoom-factor model.
    func : callable or None
        Point-wise activation function.
    """

    def __init__(self, grid, i0, masks, zoom=None, func=jnp.exp):
        check_type(grid, SignalGrid)
        check_type(i0, Model)
        check_type(masks, ArrayLike)
        check_type(zoom, (Model, type(None)))
        check_type(func, (Callable, type(None)))

        self.grid = grid
        self.i0 = i0
        self.masks = masks
        self.zoom = zoom
        self.func = func
        super().__init__(
            domain=Vector(domain_tree((self.i0, self.zoom), error=False)),
            init=model_init((self.i0, self.zoom), error=False),
        )

    def __call__(self, x, *, key=None):
        if key is None:
            key = random.PRNGKey(0)
        mk_val = random.permutation(key, self.masks, axis=0)[0]

        mk_val = rotate_data(mk_val, random.randint(key, (), 0, 4))
        mk_val = flip_data(mk_val, random.randint(key, (), 0, 4))

        zoom = self.grid.shape[0] / mk_val.shape[0]
        if self.zoom:
            raise NotImplementedError("Zoom not implemented yet.")

        in_shape = out_shape = (
            tuple(int(v * zoom) for v in mk_val.shape) if zoom < 1 else mk_val.shape
        )
        in_start = out_start = jnp.array([0, 0])

        mk_val = map_array(mk_val, 1, 1, in_shape, out_shape, in_start, out_start, zoom)

        i0_val = self.i0(x)
        if self.func:
            i0_val = self.func(i0_val)

        x_val = mk_val * i0_val
        y_val = jnp.ceil(mk_val)

        return jnp.stack((x_val, jnp.zeros(x_val.shape), y_val), axis=0)

    @classmethod
    def build(cls, *, grid, i0, masks, zoom=None, func="exp"):
        """Build an object generator from configuration dictionaries.

        Parameters
        ----------
        grid : dict
            Signal grid parameters (see ``SignalGrid``).
        i0 : dict
            Prior model parameters for the intensity.
        masks : dict
            Parameters for ``get_masks`` (``m_min``, ``m_max``).
        zoom : dict or None, optional
            Zoom-factor model parameters. Default is None.
        func : str or None, optional
            ``jax.numpy`` activation function name. Default is ``'exp'``.

        Returns
        -------
        ObjectGenerator
            The constructed object generator model.
        """
        grid = SignalGrid.build(**grid)

        i0 = normal_model(
            prefix="og i0",
            shape=(1,),
            **i0,
        )
        masks = get_masks(**masks)

        if zoom:
            zoom = uniform_model(
                prefix="og zoom",
                shape=(1,),
                **zoom,
            )
        if func:
            func = getattr(jnp, func, None)

        return cls(grid, i0, masks, zoom, func)


def get_masks(
    *,
    m_min=0,
    m_max=100,
):
    """Load 2-D binary masks from the bundled ``masks.npz`` file.

    Parameters
    ----------
    m_min : int, optional
        Start index of the mask slice. Default is 0.
    m_max : int, optional
        End index (inclusive). Zero-valued masks are appended when
        *m_max* > 90. Default is 100.

    Returns
    -------
    masks : np.ndarray
        Array of shape ``(m_max - m_min + 1, 256, 256)``.
    """
    dpath = os.path.dirname(__file__)
    fname = os.path.join(dpath, "masks.npz")
    masks = np.load(fname)["val"]

    if m_max > 90:
        masks = np.concatenate((masks, np.zeros((m_max - 90, 256, 256))), axis=0)

    return masks[m_min : m_max + 1]

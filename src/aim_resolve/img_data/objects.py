import os
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.typing import ArrayLike
from nifty8.re import Model, Vector
from typing import Callable

from .jax_fun import rotate_data, flip_data
from ..model.map import map_tiles
from ..model.prior import uniform_model
from ..model.normal import normal_model
from ..model.space import SignalSpace
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init




class ObjectGenerator(Model):
    '''Generate a object model. Use `build` function to create the model.'''

    def __init__(self, space, i0, masks, zoom=None, func=jnp.exp):
        check_type(space, SignalSpace)
        check_type(i0, Model)
        check_type(masks, ArrayLike)
        check_type(zoom, (Model, type(None)))
        check_type(func, (Callable, type(None)))

        self.space = space
        self.i0 = i0
        self.masks = masks
        self.zoom = zoom
        self.func = func
        super().__init__(
            domain=Vector(domain_tree((self.i0, self.zoom), error=False)),
            init=model_init((self.i0, self.zoom), error=False),
        )

    def __call__(self, x, *, key=random.PRNGKey(0)):
        mk_val = random.permutation(key, self.masks, axis=0)[0]

        mk_val = rotate_data(mk_val, random.randint(key, (), 0, 4))
        mk_val = flip_data(mk_val, random.randint(key, (), 0, 4))

        mk_dis = self.space.fov / mk_val.shape
        if self.zoom:
            mk_dis *= self.zoom(x)
        mk_val = map_tiles(mk_val, mk_dis, jnp.zeros((2,)), jnp.zeros(()), self.space)

        i0_val = self.i0(x)
        if self.func:
            i0_val = self.func(i0_val)

        x_val = mk_val * i0_val
        y_val = jnp.ceil(mk_val)

        return jnp.stack((x_val, jnp.zeros(x_val.shape), y_val), axis=0)

    @classmethod
    def build(cls, *, space, i0, masks, zoom=None, func='exp'):
        '''
        Build a object generator model.
        
        Parameters
        ----------
        space : dict
            Dictionary containing the signal space parameters (see SignalSpace)
        i0 : dict
            Dictionary containing the prior model parameters of the signal (see prior_model)
        masks : dict
            Dictionary containing the parameters to build the mask array (see get_masks)
        zoom : dict, optional
            Dictionary containing the zoom model parameters (see uniform_model), by default None
            -> multiply the signal with a zoom factor
        func : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        space = SignalSpace.build(**space)

        i0 = normal_model(
            prefix = 'og i0',
            shape = (1,),
            **i0,
        )
        masks = get_masks(**masks)

        if zoom:
            zoom = uniform_model(
                prefix = 'og zoom',
                shape = (1,),
                **zoom,
            )
        if func:
            func = getattr(jnp, func, None)

        return cls(space, i0, masks, zoom, func)



def get_masks(*,
        m_min = 0,
        m_max = 100, 
):
    '''
    Get the array containing 90 different 2D masks. Uses the `masks.npz` file.

    Parameters
    ----------
    m_min : int
        Minimum index of the mask array to use
    m_max : int
        Maximum index of the mask array to use. If m_max > 90, zero-valued masks are added to the array.
    '''
    dpath = os.path.dirname(__file__)
    fname = os.path.join(dpath, 'masks.npz')
    masks = np.load(fname)['val']

    masks = np.concatenate((masks, np.zeros((m_max-90, 256, 256))), axis=0)

    return masks[m_min : m_max + 1]

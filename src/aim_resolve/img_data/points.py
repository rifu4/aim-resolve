import jax.numpy as jnp
from jax import random, vmap
from jax.typing import ArrayLike
from nifty8.re import Model, Vector
from typing import Callable

from .jax_fun import gaussian_filter2d
from ..model.integer import integer_model
from ..model.map import map_points
from ..model.normal import normal_model
from ..model.prior import uniform_model
from ..model.space import SignalSpace
from ..model.util import check_type
from ..optimize.samples import domain_tree, model_init



class PointGenerator(Model):
    '''Generate a point model. Use `build` function to create the model.'''

    def __init__(self, space, i0, coordinates, n_copies, blur=None, func=jnp.exp):
        check_type(space, SignalSpace)
        check_type(i0, Model)
        check_type(coordinates, Model)
        check_type(n_copies, Model)
        check_type(blur, (ArrayLike, type(None)))
        check_type(func, (Callable, type(None)))

        self.space = space
        self.i0 = i0
        self.coordinates = coordinates
        self.n_copies = n_copies
        self.blur = blur if isinstance(blur, ArrayLike) else jnp.zeros(self.i0.target.shape[0])
        self.func = func
        super().__init__(
            domain = Vector(domain_tree((self.i0, self.coordinates, self.n_copies), error=False)), 
            init = model_init((self.i0, self.coordinates, self.n_copies), error=False),
        )

    def __call__(self, x, *, key=random.PRNGKey(0)):
        i0_val = self.i0(x)
        nc_msk = (jnp.arange(i0_val.shape[0]) < self.n_copies(x)[0]).reshape(-1, 1, 1)

        if self.func:
            i0_val = self.func(i0_val)

        x_val = map_points(i0_val * nc_msk, self.coordinates(x), self.space, vmap_sum=False)
        y_val = map_points(nc_msk, self.coordinates(x), self.space, vmap_sum=False)

        bl_val = random.permutation(key, self.blur, axis=0)[:i0_val.shape[0]]
        vmap_filter = vmap(gaussian_filter2d, in_axes=(0, 0, None, None))
        x_val = vmap_filter(x_val, bl_val, 2, False)

        x_val = jnp.sum(x_val, axis=0)
        y_val = jnp.sum(y_val, axis=0)
        
        return jnp.stack((x_val, y_val, jnp.zeros(x_val.shape)), axis=0)

    @classmethod
    def build(cls, *, n_min=0, n_max=0, space, i0, blur=None, func='exp'):
        '''
        Build a point generator model.

        Parameters
        ----------
        n_min : int
            Minimum number of points, by default 0
        n_max : int
            Maximum number of points, by default 0
        space : dict
            Dictionary containing the signal space parameters (see SignalSpace)
        i0 : dict
            Dictionary containing the prior model parameters of the signal (see prior_model)
        blur : dict, optional
            Dictionary containing the parameters to generate the blur array, by default None
            -> apply different gaussian filters to the point sources
        func : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        check_type(n_min, int)
        check_type(n_max, int)

        space = SignalSpace.build(**space)

        i0 = normal_model(
            prefix = 'pg i0 ',
            shape = (n_max, 1, 1),
            **i0,
        )
        coordinates = uniform_model(
            prefix = 'pg coordinates',
            shape = (n_max, 2),
            u_min = space.limits[0,0],
            u_max = space.limits[0,1],
        )
        n_copies = integer_model(
            prefix = 'pg n copies',
            shape = (1,),
            i_min = n_min,
            i_max = n_max + 1,
        )
        if blur:
            blur = get_blur(n_max, **blur)

        if func:
            func = getattr(jnp, func, None)

        return cls(space, i0, coordinates, n_copies, blur, func)



def get_blur(
        n_max,
        *,
        b_min = 0,
        b_max = 0,
        steps = 10
):
    '''
    Generate an array containing different blur values.

    Parameters
    ----------
    n_max : int
        Maximum number of points to generate
    b_min : float, optional
        Minimum blur value, by default 0
    b_max : float, optional
        Maximum blur value, by default 0
    steps : int, optional
        Number of blur values to generate, by default 10
    '''
    return jnp.linspace(b_min, b_max, max(n_max, steps))

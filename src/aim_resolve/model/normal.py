from functools import partial
from jax import vmap
from typing import Union
from nifty8.re import Model, NormalPrior, WrappedCall, random_like

from .util import to_shape



def normal_model(*,
        prefix: str,
        shape: tuple,
        mean: Union[tuple, float, int],
        std: Union[tuple, float, int],
        n_copies: int = 1,
) -> Model:
    '''
    Define a normal model with the given parameters.

    Parameters:
    -----------
    prefix : str
        The prefix for the model.
    shape : tuple
        The shape of the model.
    mean : tuple or float
        The mean of the model.
    std : tuple or float
        The standard deviation of the model.
    n_copies : int
        The number of copies of the model. The copies can have diffeerent means and stds.
        If 0: every entry of the model gets its own mean and std.
        If 1: The NormalPrior of nifty8.re is used.    
    '''
    if n_copies == 0:
        mean = to_shape(mean, shape, 'float64')
        std = to_shape(std, shape, 'float64')
    elif n_copies == 1:
        mean = to_shape(mean, (), 'float64')
        std = to_shape(std, (), 'float64')
        return NormalPrior(mean, std, shape=shape, name=prefix)
    else:
        mean = to_shape(mean, (n_copies,), 'float64')
        std = to_shape(std, (n_copies,), 'float64')
        shape = (n_copies, ) + shape

    ptree = {}
    call = WrappedCall(lambda x: x, shape=shape, name=prefix)
    ptree.update(call.domain)

    def standard_to_normal(xi, mean, std):
        return mean + std * xi

    def multi_normal(primals, mean, std):
        return vmap(standard_to_normal, in_axes=(0,0,0))(call(primals), mean, std)

    init = {
        k: partial(random_like, primals=v) for k, v in ptree.items()
    }
    return Model(partial(multi_normal, mean=mean, std=std), domain=ptree.copy(), init=init)

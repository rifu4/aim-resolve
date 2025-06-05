import jax.numpy as jnp
from jax.typing import ArrayLike
from nifty8.re import Model, VModel, Vector
from typing import Callable

from .map import map_signal
from .prior import prior_model
from .space import SignalSpace, PointSpace
from .util import check_type, to_shape
from ..optimize.samples import domain_tree, model_init



class SignalModel(Model):
    '''Generate a signal model. Use `build` function to create the model.'''

    factor = None

    def __init__(self, space, i0, offset=0, prefix='sm', func=jnp.exp, zero_pad=None, gaussian=None, pspec=None):
        check_type(space, (SignalSpace, PointSpace))
        check_type(i0, (Model, VModel))
        check_type(offset, ArrayLike)
        check_type(prefix, str)
        check_type(func, (Callable, type(None)))
        check_type(zero_pad, (Callable, type(None)))
        check_type(gaussian, (Model, type(None)))
        check_type(pspec, (Model, VModel, type(None)))

        self.space = space
        self.i0 = i0
        self.offset = offset
        self.prefix = prefix
        self.func = func
        self.zero_pad = zero_pad
        self.gaussian = gaussian
        self.pspec = pspec
        super().__init__(
            domain = Vector(domain_tree((self.i0, self.space.coos, self.gaussian), error=False)), 
            init = model_init((self.i0, self.space.coos, self.gaussian), error=False),
        )

    def __call__(self, x, *, out_space=None):
        res = self.i0(x)
        res += self.offset
        if self.zero_pad:
            res = self.zero_pad(res)
        if self.func:
            res = self.func(res)
        if self.gaussian:
            res *= self.gaussian(x)
        if isinstance(self.factor, ArrayLike):
            res *= self.factor
        if out_space:
            return map_signal(res, self.space, out_space)
        else:
            return res

    @classmethod
    def build(cls, *, space, i0, offset=0, prefix='sm', func='exp', zero_pad=1.0, gaussian=None):
        '''
        Build a SignalModel from the given parameters.

        Parameters
        ----------
        space : dict
            Dictionary containing the signal space parameters (see SignalSpace)
        i0 : dict
            Dictionary containing the prior model parameters of the signal (see prior_model)
        offset : float, optional
            Offset to add to the signal, by default 0
        prefix : str, optional
            Prefix for the model, by default 'sig'
        func : str, optional
            Function to apply to the signal, by default 'exp'
        zero_pad : float, optional
            Zero padding factor, by default 1.0
            -> pad the signal with zeros, 1.0 means no padding
        gaussian : dict, optional
            Dictionary containing the gaussian model parameters (see gaussian_model), by default None
            -> multiply the signal with a gaussian
        '''
        if 'coordinates' in space:
            space = PointSpace.build(**space)
        else:
            space = SignalSpace.build(**space)
        
        offset = to_shape(offset, (), 'float64')

        check_type(prefix, str)

        check_type(zero_pad, (int, float))
        pad_space, pad_func = space, None
        if zero_pad != 1.0 and isinstance(space, SignalSpace):
            pad_func = zero_pad_func(space, zero_pad)
            pad_space = zero_pad * space
        
        i0, pspec = prior_model(f'{prefix} i0 ', pad_space, **i0)

        if func:
            func = getattr(jnp, func, None)

        if gaussian != None and isinstance(space, SignalSpace):
            gaussian, _ = prior_model(f'{prefix} gm ', space, **gaussian)

        return cls(space, i0, offset, prefix, func, pad_func, gaussian, pspec)
    
    @property
    def shape(self):
        return self.space.shape
    
    def set_offset(self, offset):
        '''
        Set the offset for the signal model.

        Parameters
        ----------
        offset : float
            Offset to add to the signal model
        '''
        self.offset = to_shape(offset, (), 'float64')
        return
    
    def copy(self):
        return SignalModel(self.space, self.i0, self.offset, self.prefix, self.func, self.zero_pad, self.gaussian, self.pspec)



def zero_pad_func(space, zero_pad=1):
    '''Zero pad the signal with the given factor.'''
    if zero_pad == 1:
        return None
    elif not 1 < zero_pad <= 2:
        raise ValueError('zero_pad must be between 1 and 2')
    
    pad_space = zero_pad * space
    pad_slice = tuple(slice((os-ss)//2, ss+(os-ss)//2) for os,ss in zip(pad_space.shape, space.shape))
    return lambda x: x[pad_slice]

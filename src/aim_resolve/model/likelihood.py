from nifty8.re import Model, Gaussian, VariableCovarianceGaussian, logger
import logging
import jax.numpy as jnp
import numpy as np
import inspect

from .noise import NoiseModel
from .util import check_type
from ..img_data.data import ImageData


class ImageLikelihood:
    def __init__(self, data, model, noise_cov_inv=None, noise_std_inv=None, noise_model=None):
        check_type(data, ImageData)
        check_type(model, Model)

        self.data = data
        self.model = model
        self.noise_cov_inv = noise_cov_inv
        self.noise_std_inv = noise_std_inv
        self.noise_model = noise_model
    
    def __call__(self, x):
        if self.noise_cov_inv:
            noise_std_inv = get_at_nit(self.noise_cov_inv, 1)**0.5
        else:
            noise_std_inv = get_at_nit(self.noise_std_inv, 1)

        logger.setLevel(logging.ERROR)
        if self.scaling:
            res = lambda x: self.noise_model(x) * noise_std_inv * (self.data - self.model(x))
            lh = Gaussian(jnp.broadcast_to(0.0, self.data.shape)).amend(res)
        elif self.varcov:
            res = lambda x: (noise_std_inv * (self.data - self.model(x)), self.noise_model(x))
            lh = VariableCovarianceGaussian(jnp.broadcast_to(0.0, self.data.shape)).amend(res)
        else:
            res = lambda x: noise_std_inv * (self.data - self.model(x))
            lh = Gaussian(jnp.broadcast_to(0.0, self.data.shape)).amend(res)
        logger.setLevel(logging.DEBUG)

        return lh

    @classmethod
    def build(cls, data, model, noise, func='exp'):
        '''
        Build a LikelihoodModel from the given parameters.

        Parameters
        ----------

        '''
        model = Model(lambda x: model(x, out_space=data.space), domain=model.domain, init=model.init)

        max_std = noise['max_std'] if 'max_std' in noise else 0.001
        noise_std_inv = (max_std * np.max(data.val))**-1

        noise_model = NoiseModel.build(shape=data.space.shape, **noise)

        return cls(data.val, model, None, noise_std_inv, noise_model)



def get_at_nit(c, nit):
    if callable(c) and len(inspect.getfullargspec(c).args) == 1:
        c = c(nit)
    return c

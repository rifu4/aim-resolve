from nifty8.re import Initializer, Model

from .prior import inverse_gamma_model
from .util import check_type



class NoiseModel(Model):
    '''Noise model for the signal. Use `build` function to create the model.'''

    def __init__(self, model, prefix='nm', scaling=False, varcov=False):
        check_type(model, Model)
        check_type(prefix, str)
        check_type(scaling, bool)
        check_type(varcov, bool)
        if scaling and varcov:
            raise ValueError('`scaling` and `varcov` cannot both be `True`')

        self.model = model
        self.prefix = prefix
        self.scaling = scaling
        self.varcov = varcov
        super().__init__(
            domain = {prefix: self.model.domain}, 
            init = Initializer({prefix: self.model.init}),
        )

    def __call__(self, x):
        return 1 / self.model(x[self.prefix])

    @classmethod
    def build(cls, *, shape, parameters={}, prefix='nm', scaling=False, varcov=False, **kwargs):
        '''
        Build an inverse noise model from the given parameters.

        Parameters
        ----------
        shape : tuple
            Shape of the model (usually the shape of the data)
        parameters : dict
            Parameters for model (see inverse gamma model). Default is `{}`
        prefix : str
            Prefix for the model. Default is `nm`
        scaling : bool
            If true, mulitplies the noise with the scaling function in the likelihood. Default is `False`
        varcov : bool
            If true, uses the `VariableCovarianceLikelihood`. Default is `False`
        kwargs : keyword arguments
            Additional parameters for the noise (max_std, wgt_fac, ...) that do not belong to the noise model
        '''
        if parameters and (scaling or varcov):
            model = inverse_gamma_model(
                prefix=None,
                shape=shape,
                **parameters
            )
            return cls(model, prefix, scaling, varcov)
        else:
            return LazyNoise()
    


class LazyNoise():
    '''LazyNoise, assume constant noise without any model'''
    def __init__(self):
        self.model = None
        self.prefix = None
        self.scaling = False
        self.varcov = False

    def __call__(self, x):
        return 1

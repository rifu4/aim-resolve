import jax.numpy as jnp
from nifty8.re import Model

from .map import map_points
from .prior import prior_model, normal_model
from .signal import SignalModel
from .space import SignalSpace, PointSpace
from .util import check_type, to_shape
    


class PointModel(Model):
    '''Generate a point model. Use `build` function to create the model.'''

    def __init__(self, space, prefix='pm', points=None, n_copies=1):
        check_type(space, SignalSpace)
        check_type(prefix, str)
        check_type(points, SignalModel)
        check_type(points.space, PointSpace)
        check_type(n_copies, int)

        self.space = space
        self.prefix = prefix
        self.points = points
        self.n_copies = n_copies
        super().__init__(domain=self.points.domain, init=self.points.init)

    def __call__(self, x, *, out_space=None):
        out_space = out_space if out_space else self.space
        return map_points(self.points(x), self.points.space(x), out_space)

    @classmethod
    def build(cls, *, space, coordinates, i0, offset=0, n_copies=1, prefix='pm', func='exp'):
        '''
        Build a PointModel from the given parameters.
        
        Parameters
        ----------
        space : dict
            Dictionary containing the signal space parameters (see SignalSpace)
        coordinates : dict
            Dictionary containing the point source coordinates (see PointSpace)
        i0 : dict
            Dictionary containing the prior model parameters (see prior_model)
        offset : float or list of floats, optional
            Offsets for the individual point signals, by default '0'
        n_copies : int, optional
            Number of point sources, by default 1
        prefix : str, optional
            Prefix for the model, by default 'pm'
        func : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        space = SignalSpace.build(**space)

        point_space = PointSpace.build(coordinates=coordinates, n_copies=n_copies, prefix=f'{prefix} cm')

        i0_space = SignalSpace.build(shape=point_space.shape)
        i0, _ = prior_model(f'{prefix} i0', i0_space, n_copies, **i0)

        offset_shape = (n_copies, 1, 1) if n_copies > 1 else (1, 1)
        offset = to_shape(offset, offset_shape, 'float64')

        check_type(prefix, str)

        if func:
            func = getattr(jnp, func, None)

        points = SignalModel(point_space, i0, offset, prefix, func)

        return cls(space, prefix, points, n_copies)
    
    @property
    def shape(self):
        return (self.n_copies, ) + self.points.space.shape
    
    def set_offset(self, offset):
        '''
        Set the offset for the point model.
        
        Parameters
        ----------
        offset : float or list of floats
            Offsets for the individual point signals
        '''
        offset_shape = (self.n_copies, 1, 1) if self.n_copies > 1 else (1, 1)
        self.points.offset = to_shape(offset, offset_shape, 'float64')
        return



class CoordinateModel(Model):
    '''Generate a coordinate model. Use `build` function to create the model.'''

    def __init__(self, coordinates, n_copies=1):
        check_type(coordinates, Model)
        check_type(n_copies, int)

        self.coordinates = coordinates
        self.n_copies = n_copies
        super().__init__(domain=coordinates.domain, init=coordinates.init)

    def __call__(self, x):
        return self.coordinates(x)
    
    def __len__(self):
        return self.n_copies

    @classmethod
    def build(cls, *, coordinates, n_copies=1, prefix='cm'):
        '''
        Build a CoordinateModel from the given parameters.
        
        Parameters
        ----------
        coordinates : dict
            Dictionary containing the point source coordinates (mean and std)
        n_copies : int, optional
            Number of point sources, by default 1
        prefix : str, optional
            Prefix for the model, by default 'cm'
        '''
        coos = normal_model(
            prefix=prefix,
            shape=(n_copies, 2),
            n_copies=0,
            **coordinates
        )
        return cls(coos, n_copies)
    
    @property
    def shape(self):
        return (self.n_copies, 2)

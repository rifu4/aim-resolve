import jax.numpy as jnp
from nifty8.re import Model

from .map import map_points
from .prior import prior_model, normal_model
from .signal import SignalModel
from .grid import SignalGrid, PointGrid
from .util import check_type, to_shape
    


class PointModel(Model):
    '''Generate a point model. Use `build` function to create the model.'''

    def __init__(self, grid, prefix='pm', points=None):
        check_type(grid, SignalGrid)
        check_type(prefix, str)
        check_type(points, SignalModel)
        check_type(points.grid, PointGrid)

        self.grid = grid
        self.prefix = prefix
        self.points = points
        super().__init__(domain=self.points.domain, init=self.points.init)

    def __call__(self, x, *, out_grid=None):
        out_grid = out_grid if out_grid else self.grid
        return map_points(self.points.grid, out_grid)(self.points(x))

    @classmethod
    def build(cls, *, grid, point_grid, i0, offset=0, prefix='pm', func='exp'):
        '''
        Build a PointModel from the given parameters.
        
        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid)
        point_grid : dict
            Dictionary containing the point grid parameters (see PointGrid)
        i0 : dict
            Dictionary containing the prior model parameters (see prior_model)
        offset : float or list of floats, optional
            Offsets for the individual point signals, by default '0'
        prefix : str, optional
            Prefix for the model, by default 'pm'
        func : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        point_grid = PointGrid.build(**point_grid)
        
        grid = SignalGrid.build(**grid, factor=point_grid.factor)

        i0_grid = SignalGrid.build(space=point_grid.shape)
        i0, _ = prior_model(f'{prefix} i0', i0_grid, point_grid.n_copies, **i0)

        offset_shape = (point_grid.n_copies, 1, 1) if point_grid.n_copies > 1 else (1, 1)
        offset = to_shape(offset, offset_shape, 'float64')

        check_type(prefix, str)

        if func:
            func = getattr(jnp, func, None)

        points = SignalModel(point_grid, i0, offset, prefix, func)

        return cls(grid, prefix, points)
    
    @property
    def shape(self):
        return (self.points.grid.n_copies, ) + self.points.grid.shape
    
    def set_offset(self, offset):
        '''
        Set the offset for the point model.
        
        Parameters
        ----------
        offset : float or list of floats
            Offsets for the individual point signals
        '''
        offset_shape = (self.points.grid.n_copies, 1, 1) if self.points.grid.n_copies > 1 else (1, 1)
        self.points.offset = to_shape(offset, offset_shape, 'float64')
        return

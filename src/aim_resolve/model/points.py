import jax.numpy as jnp
import numpy as np
from nifty.re import Model, VModel

from .grid import SignalGrid, PointGrid
from .map import map_signal
from .signal import SignalModel
from .spectral import spectral_model
from .util import check_type, to_shape, extend_shape



class PointModel(Model):
    '''Generate a point model. Use `build` function to create the model.'''

    def __init__(self, grid, freq, points, prefix='pm'):
        check_type(grid, SignalGrid)
        check_type(freq, np.ndarray)
        check_type(points, SignalModel)
        check_type(points.grid, PointGrid)
        check_type(prefix, str)

        self.grid = grid
        self.freq = freq
        self.points = points
        self.prefix = prefix
        self.set_out_grid(grid)
        super().__init__(domain=self.points.domain, init=self.points.init)

    def __call__(self, x, *, map=True):
        res = self.points(x)
        if map:
            return self.map_function(res)
        return res

    @classmethod
    def build(cls, *, grid, point_grid, freq=[1.], params, prefix='pm', offset=0, nonlinearity='exp'):
        '''
        Build a PointModel from the given parameters.
        
        Parameters
        ----------
        grid : dict
            Dictionary containing the signal grid parameters (see SignalGrid)
        point_grid : dict
            Dictionary containing the point grid parameters (see PointGrid)
        freq : list or np.ndarray or Observation
            Frequencies of the signal model. If an Observation is given, the frequencies are extracted from it, by default '[1.]'
        params : dict
            Dictionary containing the spectral model parameters of the signal (see spectral_model)
        prefix : str, optional
            Prefix for the model, by default 'pm'
        offset : float or list of floats, optional
            Offsets for the individual point signals, by default '0'
        nonlinearity : str, optional
            Function to apply to the signal, by default 'exp'
        '''
        from ..resolve.observation import Observation

        point_grid = PointGrid.build(**point_grid)

        grid = SignalGrid.build(**{'factor': point_grid.factor} | grid)

        if isinstance(freq, Observation):
            freq = freq.freq
        freq = to_shape(freq, (len(freq),), 'float64')

        if nonlinearity:
            nonlinearity = getattr(jnp, nonlinearity, None)

        model_grid = SignalGrid.build(space=point_grid.shape)
        model = spectral_model(f'{prefix} ', model_grid, freq, nonlinearity, point_grid.n_copies, **params)

        offset_shape = extend_shape(point_grid.n_copies, freq, (1, 1), offset=True)
        offset = to_shape(offset, offset_shape, 'float64')

        points = SignalModel(point_grid, freq, model, prefix, offset, nonlinearity)

        return cls(grid, freq, points, prefix)
    
    def set_out_grid(self, out_grid):
        check_type(out_grid, SignalGrid)
        self.map_function = map_signal(self.points.grid, out_grid)
        return
    
    @property
    def shape(self):
        return extend_shape(self.points.grid.n_copies, self.freq, self.points.grid.shape)
    
    @property
    def n_copies(self):
        return self.points.grid.n_copies
    
    def set_offset(self, offset):
        '''
        Set the offset for the point model.
        
        Parameters
        ----------
        offset : float or list of floats
            Offsets for the individual point signals
        '''
        offset_shape = extend_shape(self.points.grid.n_copies, self.freq, (1, 1), offset=True)
        self.points.offset = to_shape(offset, offset_shape, 'float64')
        return
    
    def copy(self):
        return PointModel(self.grid, self.freq, self.points, self.prefix)

    @property
    def spectral_index(self):
        '''Return the spectral index model.'''
        return PointModel(self.grid, self.freq, self.points.spectral_index, self.prefix)

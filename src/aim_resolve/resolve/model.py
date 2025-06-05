import jax.numpy as jnp
from nifty8.re import Model

from .observation import Observation
from .response import point_response, signal_response
from ..model.components import ComponentModel
from ..model.points import PointModel
from ..model.signal import SignalModel
from ..model.tiles import TileModel
from ..model.util import check_type



class SignalResponse(Model):
    '''Generate a signal response model. Applies the radio response to a signal or component model.'''

    def __init__(self, model, observation, wgridding=False):
        '''
        Initialize the signal response model.

        Parameters
        ----------
        model : SignalModel
            The signal model the signal response function is applied to.
        observation : Observation
            Observation data.
        wgridding : bool
            Whether to use wgridding or not.
        '''
        check_type(model, (SignalModel, ComponentModel))
        check_type(observation, Observation)
        check_type(wgridding, bool)

        self.model = model
        self.observation = observation
        self.wgridding = wgridding
        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        return signal_response(self.model.space, self.observation, self.wgridding)(self.model(x))
    


class PointResponse(Model):
    '''Generate a point response model. Applies the radio response to a point model.'''

    def __init__(self, model, observation):
        '''
        Initialize the point response model.

        Parameters
        ----------
        model : PointModel
            The point model the point response function is applied to.
        observation : Observation
            Observation data.
        '''
        check_type(model, PointModel)
        check_type(observation, Observation)

        self.model = model
        self.points = model.points
        self.observation = observation
        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        return point_response(self.points(x), self.points.space(x), self.model.space, self.observation)

        

class TileResponse(Model):
    '''Generate a tile response model. Applies the radio response to a tile model.'''

    def __init__(self, model, observation, wgridding=False):
        '''
        Initialize the tile response model.

        Parameters
        ----------
        model : TileModel
            The tile model the tile response function is applied to.
        observation : Observation
            Observation data.
        wgridding : bool
            Whether to use wgridding or not.
        '''
        check_type(model, TileModel)
        check_type(observation, Observation)
        if wgridding:
            raise ValueError('ducc response cannot vmap over multiple signals')

        self.model = model
        self.tiles = model.tiles
        self.observation = observation
        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        # return signal_response(self.tiles(x), self.tiles.space, self.observation)
        raise NotImplementedError('TileResponse not implemented yet')



class ComponentResponse(Model):
    '''Generate a component response model. Applies the radio response to a component model.'''

    def __init__(self, model, observation, split=False, wgridding=False):
        '''
        Initialize the component response model.

        Parameters
        ----------
        model : ComponentModel
            The component model the component response function is applied to
        observation : Observation
            Observation data
        split : bool
            Whether to split the model into separate components
        wgridding : bool
            Whether to use wgridding or not
        '''
        check_type(model, ComponentModel)
        check_type(observation, Observation)
        check_type(split, bool)
        check_type(wgridding, bool)

        #TODO: check which way is faster (split or separate)
        if split:
            self.models = model.models
        else:
            self.models = model.separate
        self.observation = observation
        self.wgridding = wgridding
        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        res = jnp.zeros(self.observation.vis.shape)
        #TODO: speed up the for loop with jax
        for m in self.models:
            if isinstance(m, PointModel):
                res += point_response(m.points(x), m.points.space(x), m.space, self.observation)
            elif isinstance(m, TileModel):
                res += signal_response(m.tiles.space, self.observation, self.wgridding)(m.tiles(x))
            else:
                res += signal_response(m.space, self.observation, self.wgridding)(m(x))
        return res

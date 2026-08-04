"""Sky model construction for radio interferometric imaging."""

from nifty.re import Model

from ..model.components import ComponentModel
from ..model.points import PointModel
from ..model.signal import SignalModel
from ..model.tiles import TileModel
from ..model.util import check_type
from .observation import Observation
from .response import point_response, signal_response


class SignalResponse(Model):
    """Generate a signal response model.

    Applies the (finufft or, with ``wgridding``, ducc) interferometric response
    to a diffuse signal or to a full component sky. A lone ``SignalModel`` is
    evaluated on its own grid, while a ``ComponentModel`` is first summed onto
    its single output grid ("apply the response once on the full sky"). Works
    for single- and multi-frequency models: the response maps each model
    frequency channel onto exactly one data frequency.
    """

    def __init__(self, model, observation, wgridding=False):
        """Initialize the signal response model.

        Parameters
        ----------
        model : SignalModel or ComponentModel
            The signal (or full component) model the response is applied to.
        observation : Observation
            Observation data.
        wgridding : bool
            Whether to use wgridding (ducc response) or the finufft response.
        """
        check_type(model, (SignalModel, ComponentModel))
        check_type(observation, Observation)
        check_type(wgridding, bool)

        self.model = model
        self.observation = observation
        self.wgridding = wgridding
        # A ComponentModel evaluates onto its (single) output grid; a lone
        # SignalModel lives on its own grid.
        self.grid = getattr(model, "out_grid", model.grid)
        self.response = signal_response(self.grid, observation, wgridding)
        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        return self.response(self.model(x))


class PointResponse(Model):
    """Generate a point response model.

    Applies the interferometric response to a point model directly in uv-space
    (no gridding): each source contributes a phase-shifted amplitude at its
    fixed sky coordinate. Works for single- and multi-frequency point models
    and for one or several point sources.
    """

    def __init__(self, model, observation):
        """Initialize the point response model.

        Parameters
        ----------
        model : PointModel
            The point model the point response function is applied to.
        observation : Observation
            Observation data.
        """
        check_type(model, PointModel)
        check_type(observation, Observation)

        self.model = model
        self.points = model.points
        # Fixed point-source coordinates: (2,) for one source, (n, 2) for more.
        self.coos = model.points.grid.coos
        self.observation = observation
        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        return point_response(
            self.points(x), self.coos, self.model.grid, self.observation
        )


class TileResponse(Model):
    """Generate a tile response model.

    Applies the finufft response to a tile model. Each of the ``n_copies``
    extended components is a small image on the shared tile grid; the response
    vmaps over the copies axis (one phase center per tile) and sums their
    visibility contributions. Works for single- and multi-frequency models.
    """

    def __init__(self, model, observation, wgridding=False):
        """Initialize the tile response model.

        Parameters
        ----------
        model : TileModel
            The tile model the tile response function is applied to.
        observation : Observation
            Observation data.
        wgridding : bool
            Must be ``False``; ducc/wgridding cannot vmap over the tile copies.
        """
        check_type(model, TileModel)
        check_type(observation, Observation)
        if wgridding:
            raise ValueError("ducc response cannot vmap over multiple signals")

        self.model = model
        self.observation = observation
        # The tiles live on their own (small) grid with n_copies > 1.
        self.response = signal_response(model.tiles.grid, observation, wgridding=False)
        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        # map=False keeps the (gaussian-modulated) tiles on the tile grid; the
        # response phase-shifts each copy to its center instead of gridding.
        return self.response(self.model(x, map=False))


class ComponentResponse(Model):
    """Generate a component response model.

    Applies the interferometric response to a full component model, either

    * **split** (``split=True``, default): the response is applied to each
      component on its own native grid -- diffuse ``SignalModel``\\ s via the
      signal response, ``PointModel``\\ s via the point response, and
      ``TileModel``\\ s via the tile response -- and the resulting visibilities
      are summed; or
    * **once** (``split=False``): the full sky is first built on a single grid
      (``ComponentModel.__call__``) and a single signal response is applied.

    Both paths support single- and multi-frequency models.
    """

    def __init__(self, model, observation, split=False, wgridding=False):
        """Initialize the component response model.

        Parameters
        ----------
        model : ComponentModel
            The component model the component response function is applied to.
        observation : Observation
            Observation data.
        split : bool
            If ``True`` apply the response per component and sum the
            visibilities; if ``False`` build the full sky once and apply a
            single response. Default is ``True``.
        wgridding : bool
            Whether to use wgridding (ducc) for the diffuse/full-sky signal
            response. Tile components always use the finufft response.
        """
        check_type(model, ComponentModel)
        check_type(observation, Observation)
        check_type(split, bool)
        check_type(wgridding, bool)

        self.model = model
        self.observation = observation
        self.split = split
        self.wgridding = wgridding

        if split:
            self.responses = tuple(
                _sub_response(m, observation, wgridding) for m in model.models
            )
        else:
            # Build the full sky on a single grid, then apply one response.
            self.responses = (SignalResponse(model, observation, wgridding),)

        super().__init__(domain=model.domain, init=model.init)

    def __call__(self, x):
        res = self.responses[0](x)
        for response in self.responses[1:]:
            res = res + response(x)
        return res


def _sub_response(model, observation, wgridding):
    """Build the response matching a single component model.

    Parameters
    ----------
    model : SignalModel, PointModel or TileModel
        The component model to wrap.
    observation : Observation
        Observation data.
    wgridding : bool
        Whether to use wgridding for signal components.

    Returns
    -------
    Model
        The matching response model.
    """
    if isinstance(model, PointModel):
        return PointResponse(model, observation)
    if isinstance(model, TileModel):
        return TileResponse(model, observation)
    return SignalResponse(model, observation, wgridding)

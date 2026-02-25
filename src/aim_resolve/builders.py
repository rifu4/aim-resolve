"""Builder utilities for constructing pipeline sections from configuration."""

from .data import data_func
from .likelihood import likelihood_func
from .model.components import ComponentModel
from .model.points import PointModel
from .model.signal import SignalModel
from .model.tiles import TileModel
from .transition import transition_func



def get_builders(sections : dict):
    """Create a builders dictionary mapping section names to their builder functions.

    Automatically selects the correct builder function for each section
    based on the section key prefix.

    For sky sections the dictionary values indicate the sky model type.
    For other sections the ``mode`` key selects the concrete implementation:

    - **data** : ``image`` or ``radio``
    - **lh** : ``image``, ``fast``, ``radio`` or ``sum``
    - **trans** : ``anew``, ``freq``, ``addt`` or ``zoom``

    Parameters
    ----------
    sections : dict
        Dictionary of section names to their configuration values.

    Returns
    -------
    builders : dict
        Dictionary mapping section names to callable builder functions.

    Raises
    ------
    ValueError
        If a sky section type cannot be determined from its values.
    """
    builders = {}
    for sec,val in sections.items():
        sec = str(sec)

        if sec.startswith('data') or sec.startswith('obs'):
            builders[sec] = data_func

        elif sec.startswith('lh') or sec.startswith('likelihood'):
            builders[sec] = likelihood_func

        elif sec.startswith('sky') or sec.startswith('sig') or sec.startswith('model'):
            match val:
                case v if 'background' in v:
                    builders[sec] = ComponentModel.build
                case v if 'point_grid' in v:
                    builders[sec] = PointModel.build
                case v if 'tile_grid' in v:
                    builders[sec] = TileModel.build
                case v if 'params' in v:
                    builders[sec] = SignalModel.build
                case _:
                    raise ValueError(f'Cannot determine the type of the sky model `{sec}`')

        elif sec.startswith('trans') or sec.startswith('transition'):
            builders[sec] = transition_func

    return builders

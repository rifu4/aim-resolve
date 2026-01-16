from .data import data_func
from .likelihood import likelihood_func
from .model.components import ComponentModel
from .model.points import PointModel
from .model.signal import SignalModel
from .model.tiles import TileModel
from .transition import transition_func



def get_builders(sections : dict):
    '''
    Create the builders dictionary if it isn`t specified.
    
    Parameters
    ----------
    sections : dict
        Dictionary containing the sections of the model.
        -> automatically selects the correct function to use depending on the section key.

    For the sky sections the used keys indicate the type of the sky model.
    For the other section it is necessary to specify the `mode`:
    - data: `image` or `radio`
    - lh: `image`, `fast`, `radio` or `sum`
    - trans: `anew`, `freq`, `addt` or `zoom`
    '''
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

from .data import image_data, radio_data
from .likelihood import image_likelihood, radio_likelihood, fast_likelihood, likelihood_sum
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

    Use correct keys to indicate the type of model:
    - data or obs: data model (exp, radio)
    - lh: likelihood function (exp, radio, fast_radio, sum)
    - sky or sig or model: sky model (component, point, tile, signal)
    - trans: transition function
    '''
    builders = {}
    for sec,val in sections.items():
        sec = str(sec)

        if sec.startswith('data') or sec.startswith('obs'):
            match val['fun']:
                case 'exp':
                    builders[sec] = image_data
                case f if 'radio' in f:
                    builders[sec] = radio_data
                case _:
                    raise ValueError('`fun` has to be either `exp`, `radio`, or `fast_radio`')

        elif sec.startswith('lh'):
            match val['fun']:
                case 'exp':
                    builders[sec] = image_likelihood
                case f if 'fast' in f and 'radio' in f:
                    builders[sec] = fast_likelihood
                case 'radio':
                    builders[sec] = radio_likelihood
                case 'sum':
                    builders[sec] = likelihood_sum
                case _:
                    raise ValueError('`fun` has to be either `exp`, `radio`, `fast_radio`, or `sum`')

        elif sec.startswith('sky') or sec.startswith('sig') or sec.startswith('model'):
            match val:
                case v if 'background' in v:
                    builders[sec] = ComponentModel.build
                case v if 'coordinates' in v:
                    builders[sec] = PointModel.build
                case v if 'tile_spaces' in v:
                    builders[sec] = TileModel.build
                case v if 'i0' in v:
                    builders[sec] = SignalModel.build
                case _:
                    raise ValueError(f'Cannot determine the type of the sky model `{sec}`')

        elif sec.startswith('trans'):
            builders[sec] = transition_func

    return builders

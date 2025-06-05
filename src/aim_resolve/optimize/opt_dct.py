from .opt_config import get_reps
from .util import get_it, clean_reps, is_or_contains_type, check_dict



OPT_NEEDED = {'n_total_iterations', 'n_samples'}
OPT_OPTION = {'draw_linear_kwargs', 'nonlinearly_update_kwargs', 'kl_kwargs', 'sample_mode'}



def callable_optimize_dict(opt_dct):

    opt_dct = check_dict(opt_dct, OPT_NEEDED, OPT_OPTION)

    opt_dct = get_reps(opt_dct, opt_dct['n_total_iterations'])
    opt_dct = clean_reps(opt_dct)
    for ok,ov in opt_dct.items():
        opt_dct[ok] = make_callable(ov)

    return opt_dct



def make_callable(val):
    '''Make a list of values callable.'''

    def fun(it):
        return get_it(val, it)
    
    if is_or_contains_type(val, list):
        return fun
    else:
        return fun(0)

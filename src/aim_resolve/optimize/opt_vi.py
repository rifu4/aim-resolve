import jax
from functools import partial
from nifty8.re import OptimizeVI
from nifty8.re.optimize_kl import _kl_vg, _kl_met, draw_linear_residual, nonlinearly_update_residual, get_status_message

from .samples import MySamples



class MyOptimizeVI(OptimizeVI):
    '''
    Extension of the OptimizeVI class to handle callable likelihood functions.
    
    Parameters
    ----------
    lh_fun : callable
        Function to generate the likelihood given a sky model and data (see `my_lh` function in `opt_kl.py` file).
    kwargs : dict
        Additional arguments to pass to the OptimizeVI class.
    '''
    def __init__(self, lh_fun, **kwargs):
        super().__init__(
            likelihood = None,
            n_total_iterations = None,
            **kwargs,
            _kl_value_and_grad = partial(my_kl_vg, lh_fun=lh_fun),
            _kl_metric = partial(my_kl_metric, lh_fun=lh_fun),
            _draw_linear_residual = partial(my_draw_linear_residual, lh_fun=lh_fun),
            _nonlinearly_update_residual = partial(my_nonlinearly_update_residual, lh_fun=lh_fun),
            _get_status_message = partial(my_stat_mes, lh_fun=lh_fun),
        )

    def my_update(self, samples, opt_vi_st, lh_dict):
        '''Update the samples and state with the likelihood function and return the output as a MySamples object.'''
        samples, opt_vi_st = self.update(samples, opt_vi_st, lh_dict=lh_dict)
        samples = MySamples(pos=samples._pos, samples=samples._samples, keys=samples._keys)
        return samples, opt_vi_st
    

jax.jit
def my_kl_vg(primals, primals_samples, *, lh_fun, lh_dict, **kwargs):
    lh = lh_fun(**lh_dict)
    return _kl_vg(lh, primals, primals_samples, **kwargs)

jax.jit
def my_kl_metric(primals, tangents, primals_samples, *, lh_fun, lh_dict, **kwargs):
    lh = lh_fun(**lh_dict)
    return _kl_met(lh, primals, tangents, primals_samples, **kwargs)

jax.jit
def my_draw_linear_residual(pos, key, *, lh_fun, lh_dict, **kwargs):
    lh = lh_fun(**lh_dict)
    return draw_linear_residual(lh, pos, key, **kwargs)

def my_nonlinearly_update_residual(pos, residual_sample, metric_sample_key, metric_sample_sign, *, lh_fun, lh_dict, **kwargs):
    lh = lh_fun(**lh_dict)
    return nonlinearly_update_residual(lh, pos, residual_sample, metric_sample_key, metric_sample_sign, **kwargs)

def my_stat_mes(samples, state, *, lh_fun, lh_dict, **kwargs):
    lh = lh_fun(**lh_dict)
    return get_status_message(samples, state, lh.normalized_residual, **kwargs)

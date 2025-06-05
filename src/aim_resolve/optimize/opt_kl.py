import os
import inspect
import logging
import pickle
from functools import partial
from os import makedirs
from typing import Callable, Literal, Optional, Union

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from nifty8.re import Gaussian, OptimizeVIState, VariableCovarianceGaussian, logger

from .opt_vi import MyOptimizeVI
from .samples import MySamples, get_samples



def get_at_nit(c, nit):
    '''Get the value of `c` at the iteration `nit`.'''
    if callable(c) and len(inspect.getfullargspec(c).args) == 1:
        c = c(nit)
    return c


def my_lh(*, model, data, noise_cov_inv=None, noise_std_inv=None, noise_model=None):
    '''Likelihood function that is passed to the OptimizeVI class. Builds a likelihood at each iteration.'''

    if noise_cov_inv:
        noise_std_inv = get_at_nit(noise_cov_inv, 1)**0.5
    else:
        noise_std_inv = get_at_nit(noise_std_inv, 1)

    logger.setLevel(logging.ERROR)
    if noise_model and noise_model.scaling:
        res = lambda x: noise_model(x) * noise_std_inv * (data - model(x))
        lh = Gaussian(jnp.broadcast_to(0.0, data.shape)).amend(res)
    elif noise_model and noise_model.varcov:
        res = lambda x: (noise_std_inv * (data - model(x)), noise_model(x))
        lh = VariableCovarianceGaussian(jnp.broadcast_to(0.0, data.shape)).amend(res)
    else:
        res = lambda x: noise_std_inv * (data - model(x))
        lh = Gaussian(jnp.broadcast_to(0.0, data.shape)).amend(res)
    logger.setLevel(logging.DEBUG)
    
    return lh


_reduce = partial(tree_map, partial(jnp.mean, axis=0))


SMPL_MODE_TYP = Literal[
    "linear_sample",
    "linear_resample",
    "nonlinear_sample",
    "nonlinear_resample",
    "nonlinear_update",
]
SMPL_MODE_GENERIC_TYP = Union[SMPL_MODE_TYP, Callable[[int], SMPL_MODE_TYP]]



def optimize_kl(
    likelihood: Union[dict, Callable[[int], dict]],
    *,
    key: Union[ArrayLike, int],
    n_total_iterations: int,
    n_samples: Union[int, Callable[[int], int]],
    position_or_samples=None,
    transitions: Union[Callable, None] = None,
    constants=(),
    point_estimates=(),
    kl_jit=True,
    residual_jit=True,
    kl_map=jax.vmap,
    residual_map="lmap",
    kl_reduce=_reduce,
    mirror_samples=True,
    draw_linear_kwargs=dict(cg_name="SL", cg_kwargs=dict()),
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(name="SN", cg_kwargs=dict(name="SNCG"))
    ),
    kl_kwargs=dict(minimize_kwargs=dict(name="M", cg_kwargs=dict(name="MCG"))),
    sample_mode: SMPL_MODE_GENERIC_TYP = "nonlinear_resample",
    resume: Union[str, bool] = False,
    callback: Optional[Callable[[MySamples, OptimizeVIState], None]] = None,
    odir: Optional[str] = None,
) -> tuple[MySamples, OptimizeVIState]:
    '''
    One-stop-shop for MGVI/geoVI style VI approximation. Can be used with the `OptimizeKLConfig` class.

    Parameters
    ----------
    likelihood: dict or callable
        Dictionary containing the inputs for the likelihood function as items (see `my_lh` function):
        - model: Model
        - data: array-like
        - noise_cov_inv: callable or array-like
        - noise_std_inv: callable or array-like
        - noise_model: Model or None
    key : int or array-like
        Random key. If an integer is passed, it is used to seed a random key.
    n_total_iterations : int
        Total number of iterations.
    n_samples : int, callable or None
        Number of samples. 
    position_or_samples: Samples or tree-like
        Initial position for minimization. If `None`, draw new samples randomly. Default is None.
    transitions : callable or None
        Transition function that can be used if parts of the likelihood are 
        replaced and thereby have a different domain.
    constants: tree-like structure, tuple of str or callable
        Pytree of same structure as likelihood input but with boolean
        leaves indicating whether to keep these values constant during the
        KL minimization. As a convenience method, for dict-like inputs, a
        tuple of strings is also valid. From these the boolean indicator
        pytree is automatically constructed.
    point_estimates: tree-like structure, tuple of str or callable
        Pytree of same structure as likelihood input but with boolean
        leaves indicating whether to sample the value in the input or use
        it as a point estimate. As a convenience method, for dict-like
        inputs, a tuple of strings is also valid. From these the boolean
        indicator pytree is automatically constructed.
    kl_jit: bool or callable
        Whether to jit the KL minimization.
    residual_jit: bool or callable
        Whether to jit the residual sampling functions.
    kl_map: callable or str
        Map function used for the KL minimization.
    residual_map: callable or str
        Map function used for the residual sampling functions.
    kl_reduce: callable
        Reduce function used for the KL minimization.
    mirror_samples: bool
        Whether to mirror the samples or not.
    draw_linear_kwargs : dict or callable
        Configuration for drawing linear samples
    nonlinearly_update_kwargs : dict or callable
        Configuration for nonlinearly updating samples
    kl_kwargs : dict or callable
        Keyword arguments for the KL minimizer.
    sample_mode : str or callable
        One in {"linear_sample", "linear_resample", "nonlinear_sample",
        "nonlinear_resample", "nonlinear_update"}. The mode denotes the way
        samples are drawn and/or updates, "linear" draws MGVI samples,
        "nonlinear" draws MGVI samples which are then nonlinearly updated
        with geoVI, the "_sample" versus "_resample" suffix denotes whether
        the same stochasticity or new stochasticity is used for the drawing
        of the samples, and "nonlinear_update" nonlinearly updates existing
        samples using geoVI.
    resume : str or bool
        Resume partially run optimization. If `True`, the optimization is
        resumed from the previos state in `odir` otherwise it is resumed from
        the location toward which `resume` points
        (stating the folder containing the `last.pkl` file is sufficient).
    callback : callable or None
        Function called after every global iteration taking the samples and the
        optimization state.
    odir : str or None
        Path at which all output files are saved.

    Returns
    -------
    samples : MySamples
        Posterior samples.
    opt_vi_st : OptimizeVIState
        State of the optimization.
    '''
    LAST_FILENAME = "last.pkl"
    MINISANITY_FILENAME = "minisanity.txt"
    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None
    sanity_fn = os.path.join(odir, MINISANITY_FILENAME) if odir is not None else None

    samples, opt_vi_st, msg = {}, None, ''
    if resume:
        rdir = resume if os.path.isdir(resume) else odir

        old_last_fn = os.path.join(rdir, LAST_FILENAME)
        with open(old_last_fn, "rb") as f:
            samples, opt_vi_st = pickle.load(f)
        
        old_sanity_fn = os.path.join(rdir, MINISANITY_FILENAME)
        with open(old_sanity_fn, "r") as f:
            msg = f.read()

    if odir:
        makedirs(odir, exist_ok=True)
        with open(sanity_fn, "w") as f:
            f.write(msg)
    
    key = random.PRNGKey(key) if isinstance(key, int) else key

    opt_vi = MyOptimizeVI(
        lh_fun=my_lh,
        kl_jit=kl_jit,
        residual_jit=residual_jit,
        kl_map=kl_map,
        residual_map=residual_map,
        kl_reduce=kl_reduce,
        mirror_samples=mirror_samples,
    )

    if opt_vi_st is None or len(opt_vi_st.config) == 0:
        key, k_o = random.split(key)
        opt_vi_st_init = opt_vi.init_state(
            k_o,
            n_samples=n_samples,
            draw_linear_kwargs=draw_linear_kwargs,
            nonlinearly_update_kwargs=nonlinearly_update_kwargs,
            kl_kwargs=kl_kwargs,
            sample_mode=sample_mode,
            point_estimates=point_estimates,
            constants=constants,
        )
        opt_vi_st = opt_vi_st_init if opt_vi_st is None else opt_vi_st
        if len(opt_vi_st.config) == 0:  # resume or _optimize_vi_state has empty config
            opt_vi_st = opt_vi_st._replace(config=opt_vi_st_init.config)

    nm = "OPTIMIZE_KL"
    for i in range(opt_vi_st.nit, n_total_iterations):
        logger.info(f"{nm}: Starting {i+1:04d}")
        lh_i = get_at_nit(likelihood, i)
        tr_i = get_at_nit(transitions, i)
        key, samples = get_samples(key, samples, position_or_samples, lh_i, tr_i, opt_vi_st.nit)
        samples, opt_vi_st = opt_vi.my_update(samples, opt_vi_st, lh_dict=lh_i)
        msg = opt_vi.get_status_message(samples, opt_vi_st, lh_dict=lh_i, name=nm)
        logger.info(msg)
        if odir:
            with open(last_fn, "wb") as f:
                pickle.dump((samples, opt_vi_st._replace(config={})), f)
            with open(sanity_fn, "a") as f:
                f.write('\n' + msg)
        if callback is not None:
            callback(samples, opt_vi_st)

    return samples, opt_vi_st

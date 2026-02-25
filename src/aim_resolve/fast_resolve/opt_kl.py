import os
import logging
import pickle
import dataclasses
from os import makedirs
from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike
from nifty.re import Gaussian, Model, OptimizeVIState, Samples, logger
from nifty.re.optimize import _newton_cg
from nifty.re.conjugate_gradient import cg as _cg

from .response import apply_exact_response
from ..optimize.opt_kl import get_at_nit, _reduce, SMPL_MODE_GENERIC_TYP
from ..optimize.opt_vi import MyOptimizeVI
from ..optimize.samples import MySamples, get_samples



class SkyResidualModel(Model):
    sky_response: Callable = dataclasses.field(metadata=dict(static=True))
    old_reconstruction: ArrayLike = dataclasses.field(metadata=dict(static=True))
    residual_data: ArrayLike = dataclasses.field(metadata=dict(static=True))

    def __init__(self, sky_response, old_reconstruction, residual_data):
        self.sky_response = sky_response
        self.old_reconstruction = old_reconstruction
        self.residual_data = residual_data
        super().__init__(domain=sky_response.domain, init=sky_response.init)

    def __call__(self, x):
        return self.sky_response(x, self.old_reconstruction, self.residual_data)



def my_lh(*, sky_response, old_reconstruction, residual_data, **kwargs):
    '''fast-resolve likelihood function. It builds a likelihood at each iteration.'''

    model = SkyResidualModel(sky_response, old_reconstruction, residual_data)

    logger.setLevel(logging.ERROR)
    lh = Gaussian(jnp.broadcast_to(0.0, residual_data.shape)).amend(model)
    logger.setLevel(logging.DEBUG)
    return lh



def fast_optimize_kl(
    likelihood: Union[dict, Callable[[int], dict]],
    *,
    key: Union[ArrayLike, int],
    n_major_iterations: int,
    n_minor_iterations: Union[int, Callable[[int], int]],
    n_samples: Union[int, Callable[[int], int]],
    position_or_samples=None,
    transitions: Union[Callable, None] = None,
    constants=(),
    point_estimates=(),
    kl_jit=True,
    residual_jit=True,
    kl_map=jax.vmap,
    residual_map='lmap',
    kl_reduce=_reduce,
    mirror_samples=True,
    draw_linear_kwargs=dict(minimize=_cg, cg_name='SL', cg_kwargs=dict()),
    # draw_linear_kwargs=dict(cg_name='SL', cg_kwargs=dict()),
    nonlinearly_update_kwargs=dict(minimize_kwargs=dict(name='SN', cg_kwargs=dict(name=None))),
    kl_kwargs=dict(minimize=_newton_cg, minimize_kwargs=dict(name='M', cg_kwargs=dict(name=None))),
    # kl_kwargs=dict(minimize_kwargs=dict(name='M', cg_kwargs=dict(name=None))),
    sample_mode: SMPL_MODE_GENERIC_TYP = 'nonlinear_resample',
    resume: Union[str, bool] = False,
    callback: Optional[Callable[[Samples, OptimizeVIState], None]] = None,
    odir: Optional[str] = None,
    devices: Optional[list] = None,
    kl_device_map='shard_map',
    residual_device_map='shard_map',
) -> tuple[MySamples, OptimizeVIState, int]:
    '''
    One-stop-shop for fast-resolve approximation with major and minor cycles.

    Parameters
    ----------
    likelihood: dict or callable
        Dictionary containing the inputs for the likelihood function as items (see `my_lh` function):
        - data: array-like
        - sky_model: Model
        - sky_response: Model
        - old_reconstruction: array-like
        - residual_data: array-like
        - RNR: callable
    key : int or array-like
        Random key. If an integer is passed, it is used to seed a random key.
    n_major_iterations : int
        Number of major iterations.
    n_minor_iterations : int or callable
        Number of minor iterations. Can be different for each major iteration.
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
    devices : list of devices or None
        Devices over which the samples are mapped. If `None` only the
        default device is used. To use all detected devices pass
        jax.devices(). Generally the samples needs to be evenly
        distributable over the devices. For details see the descriptions of
        the `kl_device_map` and `residual_device_map` arguments.
    kl_device_map : str
        Map function used for mapping KL minimization over the devices
        listed in `devices`. `kl_device_map` can be 'shard_map', 'pmap', or
        'jit'. If set to 'pmap', 2*n_samples need to be equal to the number
        of devices. For all other maps the samples needs to be equally
        distributable over the devices.
    residual_device_map : str
        Map function used for mapping sampling over the devices listed in
        `devices`. `residual_device_map` can be 'shard_map', 'pmap', or
        'jit'. If set to 'pmap', 2*n_samples need to be equal to the number
        of devices. If only linear samples are drawn ,'pmap' also works if
        n_samples equals the number of devices. For the other maps it is
        sufficient if 2*n_samples equals the number of devices, or
        n_samples can be evenly divided by the number of devices.

    Returns
    -------
    samples : MySamples
        Posterior samples.
    opt_vi_st : OptimizeVIState
        State of the optimization.
    n_major_iterations : int
        Total number of major iterations performed.
    '''
    LAST_FILENAME = 'last.pkl'
    MINISANITY_FILENAME = 'minisanity.txt'
    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None
    sanity_fn = os.path.join(odir, MINISANITY_FILENAME) if odir is not None else None

    samples, opt_vi_st, msg, last_mj = {}, None, '', 0
    if resume:
        rdir = resume if os.path.isdir(resume) else odir

        old_last_fn = os.path.join(rdir, LAST_FILENAME)
        with open(old_last_fn, 'rb') as f:
            samples, opt_vi_st, last_mj = pickle.load(f)
        
        old_sanity_fn = os.path.join(rdir, MINISANITY_FILENAME)
        with open(old_sanity_fn, 'r') as f:
            msg = f.read()

        if opt_vi_st.nit - sum(get_at_nit(n_minor_iterations, mj) for mj in range(last_mj)) < 0:
            last_mj -= 1

    if odir:
        makedirs(odir, exist_ok=True)
        with open(sanity_fn, 'w') as f:
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
        devices=devices,
        kl_device_map=kl_device_map,
        residual_device_map=residual_device_map,
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

    if not resume:
        data = get_at_nit(likelihood, 0)['data']
        residual_data = data
        sub_val = jnp.zeros(data.shape, dtype=data.dtype)
    
    for i_mj in range(last_mj, n_major_iterations):
        mj_msg = f'\nMAJOR: Iteration {i_mj+1:02d}\n'
        logger.info('\n' + mj_msg.replace('Iteration', 'Starting'))

        lh_i = get_at_nit(likelihood, i_mj)
        tr_i = get_at_nit(transitions, i_mj)
        key, samples = get_samples(key, samples, position_or_samples, lh_i, tr_i, opt_vi_st.nit)

        if opt_vi_st.nit > 0:
            sub_val = samples.mean(lh_i['sky_model'])
            residual_data = lh_i['data'] - apply_exact_response(lh_i['RNR'], sub_val)

        lh_i['old_reconstruction'] = sub_val
        lh_i['residual_data'] = residual_data

        jax.clear_caches()

        last_mn = opt_vi_st.nit - sum(get_at_nit(n_minor_iterations, mj) for mj in range(i_mj))
    
        kl_nm = 'OPTIMIZE_KL'
        for i in range(last_mn, get_at_nit(n_minor_iterations, i_mj)):
            logger.info(f'{kl_nm}: Starting {opt_vi_st.nit+1:04d}')
            samples, opt_vi_st = opt_vi.my_update(samples, opt_vi_st, lh_dict=lh_i)
            kl_msg = opt_vi.get_status_message(samples, opt_vi_st, lh_dict=lh_i, name=kl_nm)
            logger.info(mj_msg + kl_msg)
            if odir:
                with open(last_fn, 'wb') as f:
                    pickle.dump((samples, opt_vi_st._replace(config={}), i_mj+1), f)
                with open(sanity_fn, 'a') as f:
                    f.write(mj_msg + kl_msg)
            if callback is not None:
                callback(samples, opt_vi_st, i_mj+1)

    return samples, opt_vi_st, n_major_iterations

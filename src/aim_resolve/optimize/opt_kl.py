"""KL divergence optimization routines for variational inference."""

import inspect
import logging
import os
import pickle
from collections.abc import Callable
from functools import partial
from os import makedirs
from typing import Literal

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from nifty.re import (
    Gaussian,
    OptimizeVI,
    OptimizeVIState,
    VariableCovarianceGaussian,
    logger,
    static_cg,
    static_newton_cg,
)

from .samples import MySamples, get_samples


def get_at_nit(c, nit):
    """Get the value of `c` at the iteration `nit`."""
    if callable(c) and len(inspect.getfullargspec(c).args) == 1:
        return c(nit)
    elif isinstance(c, dict) and nit > 0:
        return None
    else:
        return c


_reduce = partial(tree_map, partial(jnp.mean, axis=0))


SMPL_MODE_TYP = Literal[
    "linear_sample",
    "linear_resample",
    "nonlinear_sample",
    "nonlinear_resample",
    "nonlinear_update",
]
SMPL_MODE_GENERIC_TYP = SMPL_MODE_TYP | Callable[[int], SMPL_MODE_TYP]


def build_lh(
    *,
    data,
    sky_response,
    noise_cov_inv=None,
    noise_std_inv=None,
    noise_model=None,
    **kwargs,
):
    """Likelihood function that is passed to the OptimizeVI class. Builds a likelihood at each iteration.

    Parameters
    ----------
    data : np.ndarray
        The to be reconstructed data array.
    sky_response : Model
        The response model that maps the sky to the data space.
    noise_cov_inv : np.ndarray, optional
        The inverse noise covariance matrix. Default is None.
    noise_std_inv : np.ndarray, optional
        The inverse noise standard deviation array. Default is None.
    noise_model : Model, optional
        Adds a noise model with learnable parameters. Default is None.
    **kwargs
        Additional parametrs of the likelihood dictionary.

    Returns
    -------
    lh : Model
        The likelihood model to be passed to the OptimizeVI class.
    """
    if noise_cov_inv:
        noise_std_inv = get_at_nit(noise_cov_inv, 1) ** 0.5
    else:
        noise_std_inv = get_at_nit(noise_std_inv, 1)

    logger.setLevel(logging.ERROR)
    if noise_model and noise_model.scaling:

        def res(x):
            return noise_model(x) * noise_std_inv * (data - sky_response(x))

        lh = Gaussian(jnp.broadcast_to(0.0, data.shape)).amend(res)
    elif noise_model and noise_model.varcov:

        def res(x):
            return (noise_std_inv * (data - sky_response(x)), noise_model(x))

        lh = VariableCovarianceGaussian(jnp.broadcast_to(0.0, data.shape)).amend(res)
    else:

        def res(x):
            return noise_std_inv * (data - sky_response(x))

        lh = Gaussian(jnp.broadcast_to(0.0, data.shape)).amend(res)
    logger.setLevel(logging.DEBUG)

    return lh


def optimize_kl(
    likelihood: dict | Callable[[int], dict],
    *,
    key: ArrayLike | int,
    n_total_iterations: int,
    n_samples: int | Callable[[int], int],
    position_or_samples=None,
    transitions: Callable | None = None,
    constants=(),
    point_estimates=(),
    jit=True,
    linear_minimizer_jit=False,
    nonlinear_minimizer_jit=False,
    kl_map=jax.vmap,
    residual_map="lmap",
    kl_reduce=_reduce,
    mirror_samples=True,
    draw_linear_kwargs=None,
    nonlinearly_update_kwargs=None,
    kl_kwargs=None,
    sample_mode: SMPL_MODE_GENERIC_TYP = "nonlinear_resample",
    resume: str | bool = False,
    callback: Callable[[MySamples, OptimizeVIState], None] | None = None,
    odir: str | None = None,
    devices: list | None = None,
) -> tuple[MySamples, OptimizeVIState]:
    """Run the optimization of the KL divergence with NIFTy.

    Parameters
    ----------
    likelihood : dict or callable
        Dictionary containing the values needed to build the likelihood:
        - data: array-like
        - sky_response: Model
        - noise_cov_inv: array-like or None
        - noise_std_inv: array-like or None
        - noise_model: Model or None
        Can contain additional paramters not needed for the likelihood-build.
    key : int or array_like
        JAX random key (or integer seed).
    n_total_iterations : int
        Total number of iterations.
    n_samples : int, callable or None
        Number of posterior samples.
    position_or_samples : Samples or tree-like or None, optional
        Initial position. Drawn randomly when None.
    transitions : callable or None, optional
        Transition function applied between iterations.
    constants : tuple or tree-like, optional
        Parameters held constant during KL minimisation.
    point_estimates : tuple or tree-like, optional
        Parameters treated as point estimates (not sampled).
    jit : bool, optional
        Whether to JIT-compile the KL value and gradient function. Default is True.
    linear_minimizer_jit : bool, optional
        Whether to JIT-compile the linear minimizer. Default is False.
    nonlinear_minimizer_jit : bool, optional
        Whether to JIT-compile the nonlinear minimizer. Default is False.
    kl_map : callable or str, optional
        Map function for the KL minimisation.
    residual_map : callable or str, optional
        Map function for the residual computation.
    kl_reduce : callable, optional
        Reduce function for the KL minimisation.
    mirror_samples : bool, optional
        Whether to mirror samples. Default is True.
    draw_linear_kwargs : dict, optional
        Configuration for drawing linear samples.
    nonlinearly_update_kwargs : dict, optional
        Configuration for nonlinear sample updates.
    kl_kwargs : dict, optional
        Keyword arguments for the KL minimiser.
    sample_mode : str or callable, optional
        Sampling strategy. Default is ``'nonlinear_resample'``.
    resume : str or bool, optional
        Path or flag to resume a previous run. Default is False.
    callback : callable or None, optional
        Called after every minor iteration with ``(samples, state, mj)``.
    odir : str or None, optional
        Output directory for checkpoints and logs.
    devices : list or None, optional
        JAX devices for sample distribution.

    Returns
    -------
    samples : MySamples
        Posterior samples.
    opt_vi_st : OptimizeVIState
        Final optimisation state.
    """
    if draw_linear_kwargs is None:
        draw_linear_kwargs = dict(cg_name="SL", cg_kwargs=dict())
    if nonlinearly_update_kwargs is None:
        nonlinearly_update_kwargs = dict(
            minimize_kwargs=dict(name="SN", cg_kwargs=dict(name=None))
        )
    if kl_kwargs is None:
        kl_kwargs = dict(
            minimize_kwargs=dict(name="M", cg_kwargs=dict(name=None)),
        )
    if devices is not None:

        def add_static_update(fn_or_dict, update):
            if fn_or_dict is None:
                return update
            if callable(fn_or_dict):
                return lambda i: dict(fn_or_dict(i)) | update
            return dict(fn_or_dict) | update

        draw_linear_kwargs = add_static_update(draw_linear_kwargs, dict(cg=static_cg))
        nonlinearly_update_kwargs = add_static_update(
            nonlinearly_update_kwargs, dict(minimize=static_newton_cg)
        )
        if residual_map == "lmap":
            residual_map = "vmap"

    LAST_FILENAME = "last.pkl"
    MINISANITY_FILENAME = "minisanity.txt"
    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None
    sanity_fn = os.path.join(odir, MINISANITY_FILENAME) if odir is not None else None

    samples, opt_vi_st, msg = {}, None, ""
    if resume:
        rdir = resume if os.path.isdir(resume) else odir

        old_last_fn = os.path.join(rdir, LAST_FILENAME)
        with open(old_last_fn, "rb") as f:
            samples, opt_vi_st = pickle.load(f)

        old_sanity_fn = os.path.join(rdir, MINISANITY_FILENAME)
        with open(old_sanity_fn) as f:
            msg = f.read()

    if odir:
        makedirs(odir, exist_ok=True)
        with open(sanity_fn, "w") as f:
            f.write(msg)

    key = random.PRNGKey(key) if isinstance(key, int) else key

    if opt_vi_st is None or len(opt_vi_st.config) == 0:
        key, k_o = random.split(key)
        opt_vi_st_init = OptimizeVIState(
            nit=0,
            key=k_o,
            config=dict(
                n_samples=n_samples,
                draw_linear_kwargs=draw_linear_kwargs,
                nonlinearly_update_kwargs=nonlinearly_update_kwargs,
                kl_kwargs=kl_kwargs,
                sample_mode=sample_mode,
                point_estimates=point_estimates,
                constants=constants,
            ),
        )
        opt_vi_st = opt_vi_st_init if opt_vi_st is None else opt_vi_st
        if len(opt_vi_st.config) == 0:  # resume or _optimize_vi_state has empty config
            opt_vi_st = opt_vi_st._replace(config=opt_vi_st_init.config)

    nm = "OPTIMIZE_KL"
    lh_i = None
    for i in range(opt_vi_st.nit, n_total_iterations):
        logger.info(f"{nm}: Starting {i + 1:04d}")

        if get_at_nit(likelihood, i) is not None or lh_i is None:
            if get_at_nit(likelihood, i) is not None:
                logger.info("-> Building new likelihood")
                lh_i = get_at_nit(likelihood, i)
                tr_i = get_at_nit(transitions, i)
                key, samples = get_samples(
                    key, samples, position_or_samples, lh_i, tr_i, opt_vi_st.nit
                )
            elif lh_i is None:
                for prev_i in range(i - 1, -1, -1):
                    lh_i = get_at_nit(likelihood, prev_i)
                    if lh_i is not None:
                        break
                if lh_i is None:
                    raise ValueError("No valid likelihood found for iteration.")

            opt_vi = OptimizeVI(
                likelihood=build_lh(**lh_i),
                n_total_iterations=None,
                jit=jit,
                linear_minimizer_jit=linear_minimizer_jit,
                nonlinear_minimizer_jit=nonlinear_minimizer_jit,
                kl_map=kl_map,
                residual_map=residual_map,
                kl_reduce=kl_reduce,
                mirror_samples=mirror_samples,
                devices=devices,
            )

        samples, opt_vi_st = opt_vi.update(samples, opt_vi_st)
        msg = opt_vi.get_status_message(samples, opt_vi_st, name=nm)
        logger.info(msg)
        if odir:
            with open(last_fn, "wb") as f:
                pickle.dump(
                    (MySamples.from_samples(samples), opt_vi_st._replace(config={})), f
                )
            with open(sanity_fn, "a") as f:
                f.write("\n" + msg)
        if callback is not None:
            callback(MySamples.from_samples(samples), opt_vi_st)

    return MySamples.from_samples(samples), opt_vi_st

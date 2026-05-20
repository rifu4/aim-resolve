"""Optimise-KL loop with major/minor cycles for fast-resolve."""

import dataclasses
import logging
import os
import pickle
from collections.abc import Callable
from os import makedirs

import jax  # type: ignore
from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike
from nifty.re import Gaussian, Model, OptimizeVI, OptimizeVIState, Samples, logger

from ..optimize.opt_kl import SMPL_MODE_GENERIC_TYP, _reduce, get_at_nit
from ..optimize.samples import MySamples, get_samples
from .response import apply_exact_response


class SkyResidualModel(Model):
    """Sky model that subtracts a previous reconstruction and residual.

    Wraps a ``sky_response`` model so that the forward evaluation computes
    the response to the difference between the current sky and an old
    reconstruction, minus residual data.

    Parameters
    ----------
    sky_response : Model
        Approximate response model.
    old_reconstruction : array_like
        Previous sky reconstruction subtracted before convolution.
    residual_data : array_like
        Residual data subtracted after convolution.
    """

    old_reconstruction: ArrayLike = dataclasses.field(metadata=dict(static=False))
    residual_data: ArrayLike = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky_response, old_reconstruction, residual_data):
        self.sky_response = sky_response
        self.old_reconstruction = old_reconstruction
        self.residual_data = residual_data
        super().__init__(domain=sky_response.domain, init=sky_response.init)

    def __call__(self, x):
        return self.sky_response(x, self.old_reconstruction, self.residual_data)


def build_likelihood(sky_response, old_reconstruction, residual_data):
    """Build the fast-resolve Gaussian likelihood for one minor cycle.

    Parameters
    ----------
    sky_response : Model
        Approximate response model.
    old_reconstruction : array_like
        Previous sky reconstruction.
    residual_data : array_like
        Residual (dirty-image minus predicted) data.

    Returns
    -------
    lh : Gaussian
        Gaussian likelihood amended with the residual response model.
    """

    model = SkyResidualModel(sky_response, old_reconstruction, residual_data)

    logger.setLevel(logging.ERROR)
    lh = Gaussian(jnp.broadcast_to(0.0, residual_data.shape)).amend(model)
    logger.setLevel(logging.DEBUG)
    return lh


def fast_optimize_kl(
    likelihood: dict | Callable[[int], dict],
    *,
    key: ArrayLike | int,
    n_major_iterations: int,
    n_minor_iterations: int | Callable[[int], int],
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
    callback: Callable[[Samples, OptimizeVIState], None] | None = None,
    odir: str | None = None,
    devices: list | None = None,
    plot_residuals=True,
) -> tuple[MySamples, OptimizeVIState, int]:
    """Run the fast-resolve optimise-KL loop with major/minor cycles.

    Parameters
    ----------
    likelihood : dict or callable
        Dictionary containing the values needed for the fast-resolve inference:
        - data: array-like (dirty image)
        - sky_model: Model (prior model for the sky)
        - sky_response: Model (approximate response model for the minor cycles)
        - noise_model: Model or None
        - RNR: Model or None (true response for the major cycles)
    key : int or array_like
        JAX random key (or integer seed).
    n_major_iterations : int
        Number of major (residual-update) iterations.
    n_minor_iterations : int or callable
        Number of minor (KL) iterations per major cycle.
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
    plot_residuals : bool, optional
        Whether to plot the residuals at each major iteration. Default is True.

    Returns
    -------
    samples : MySamples
        Posterior samples.
    opt_vi_st : OptimizeVIState
        Final optimisation state.
    n_major_iterations : int
        Total number of major iterations performed.
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

    LAST_FILENAME = "last.pkl"
    MINISANITY_FILENAME = "minisanity.txt"
    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None
    sanity_fn = os.path.join(odir, MINISANITY_FILENAME) if odir is not None else None

    samples, opt_vi_st, msg, last_mj = {}, None, "", 0
    if resume:
        rdir = resume if os.path.isdir(resume) else odir

        old_last_fn = os.path.join(rdir, LAST_FILENAME)
        with open(old_last_fn, "rb") as f:
            samples, opt_vi_st, last_mj = pickle.load(f)

        old_sanity_fn = os.path.join(rdir, MINISANITY_FILENAME)
        with open(old_sanity_fn) as f:
            msg = f.read()

        if (
            opt_vi_st.nit
            - sum(get_at_nit(n_minor_iterations, mj) for mj in range(last_mj))
            < 0
        ):
            last_mj -= 1

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
        if len(opt_vi_st.config) == 0:
            opt_vi_st = opt_vi_st._replace(config=opt_vi_st_init.config)

    fr_nm = "FAST-RESOLVE MAJOR"
    for i_mj in range(last_mj, n_major_iterations):
        mj_msg = f"\n{fr_nm}: Iteration {i_mj + 1:02d}\n"
        logger.info("\n" + mj_msg)

        if get_at_nit(likelihood, i_mj) is not None:
            logger.info("-> Building new likelihood")
            lh_i = get_at_nit(likelihood, i_mj)
            tr_i = get_at_nit(transitions, i_mj)
            key, samples = get_samples(
                key, samples, position_or_samples, lh_i, tr_i, opt_vi_st.nit
            )

        if opt_vi_st.nit > 0:
            old_rec = MySamples.from_samples(samples).mean(lh_i["sky_model"])
            residual_data = lh_i["data"] - apply_exact_response(lh_i["RNR"], old_rec)
        else:
            residual_data = lh_i["data"]
            old_rec = jnp.zeros(residual_data.shape, dtype=residual_data.dtype)

        if plot_residuals and odir:
            from ..plot import plot_arrays

            plot_arrays(
                array=[old_rec, residual_data],
                rows=1,
                norm="log",
                odir=f"{odir}/residuals",
                name=f"mj_{i_mj:02d}",
            )

        opt_vi = OptimizeVI(
            likelihood=build_likelihood(lh_i["sky_response"], old_rec, residual_data),
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

        last_mn = opt_vi_st.nit - sum(
            get_at_nit(n_minor_iterations, mj) for mj in range(i_mj)
        )

        kl_nm = "OPTIMIZE_KL"
        for _i in range(last_mn, get_at_nit(n_minor_iterations, i_mj)):
            logger.info(f"{kl_nm}: Starting {opt_vi_st.nit + 1:04d}")
            samples, opt_vi_st = opt_vi.update(samples, opt_vi_st)
            kl_msg = opt_vi.get_status_message(samples, opt_vi_st, name=kl_nm)
            logger.info(mj_msg + kl_msg)
            if odir:
                with open(last_fn, "wb") as f:
                    pickle.dump(
                        (
                            MySamples.from_samples(samples),
                            opt_vi_st._replace(config={}),
                            i_mj + 1,
                        ),
                        f,
                    )
                with open(sanity_fn, "a") as f:
                    f.write(mj_msg + kl_msg)
            if callback is not None:
                callback(MySamples.from_samples(samples), opt_vi_st, i_mj + 1)

    return MySamples.from_samples(samples), opt_vi_st, n_major_iterations

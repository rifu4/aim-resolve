"""Transition utilities for multi-iteration optimisation workflows.

Provides functions to transfer model parameters between successive
optimisation iterations, including random initialisation, frequency
extension, component addition and zoom refinement.
"""

import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from jax import random
from nifty.re import Model, Vector, random_like

from .mask import masks_from_model, masks_to_boxes, remove_freq_axis
from .model.components import ComponentModel
from .model.map import map_signal, upsample
from .model.noise import NoiseModel
from .model.points import PointModel
from .model.signal import SignalModel
from .model.tiles import TileModel
from .model.util import check_type
from .modeling import get_offset
from .optimize.opt_dct import callable_optimize_dict
from .optimize.opt_kl import optimize_kl
from .optimize.samples import MySamples, domain_keys, domain_tree, random_init
from .plot import dynamic_limits, plot_arrays, plot_panels, plot_style, split_style
from .reuse import box_regions, copy_plan, kept_pairs, renamed_latents, tile_slices


def transition_func(
    mode,
    **kwargs,
):
    """Return a partially applied transition function for the given mode.

    Parameters
    ----------
    mode : {'anew', 'freq', 'addt', 'zoom'}
        Transition mode.
    **kwargs
        Additional keyword arguments forwarded to the selected transition
        function.

    Returns
    -------
    func : callable
        Partially applied transition function.

    Raises
    ------
    TypeError
        If *mode* is not recognised.
    """
    if mode == "anew":
        return partial(transition_anew, **kwargs)
    elif mode == "freq":
        return partial(transition_freq, **kwargs)
    elif mode == "addt":
        return partial(transition_util, func=transition_addt, **kwargs)
    elif mode == "zoom":
        return partial(transition_util, func=transition_zoom, **kwargs)
    else:
        raise TypeError(
            f"Unknown transition mode. Available modes are `anew`, `freq`, `addt` and `zoom`, but got mode `{mode}`."
        )


def transition_anew(
    key,
    *args,
    lh_new,
    **kwargs,
):
    """Generate a new random position for all model parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    lh_new : dict
        Likelihood dictionary for the new optimisation iteration.

    Returns
    -------
    pos_new : MySamples
        Randomly initialised position.

    Raises
    ------
    ValueError
        If no model domains are found in *lh_new*.
    """
    models = [v for v in lh_new.values() if isinstance(v, Model)]
    if domain_keys(models) == set():
        raise ValueError(
            "Check that sky and noise models in the `lh_dict` are of type `nifty.re.Model`"
        )

    key, k_p = random.split(key)
    pos_new = random_init(k_p, models, factor=0.01)

    print("New model parameters:")
    for k, v in domain_tree(pos_new).items():
        print(f"  {k}:", v.shape)

    return pos_new


def transition_freq(
    key,
    samples,
    *args,
    lh_old,
    lh_new,
    reset_noise=False,
    **kwargs,
):
    """Transfer parameters from a single-frequency to a multi-frequency model.

    Copies matching model components and noise scaling from the previous
    iteration and randomly initialises any remaining parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    samples : MySamples
        Samples from the previous iteration.
    lh_old : dict
        Likelihood dictionary for the previous iteration.
    lh_new : dict
        Likelihood dictionary for the new iteration.
    reset_noise : bool, optional
        Do not take over the learned noise scaling of the previous iteration:
        the new noise model starts at its prior. Default is ``False``.

    Returns
    -------
    samples_new : MySamples
        Samples initialised for the new multi-frequency iteration.
    """
    sky_old = lh_old["sky_model"]
    sky_new = lh_new["sky_model"]
    check_type(samples, MySamples)
    check_type(sky_old, ComponentModel)
    check_type(sky_new, ComponentModel)

    # initialize an empty position tree
    ptree = {}

    # copy over matching model components
    for sky_oi, sky_ni in zip(sky_old.models, sky_new.models, strict=False):
        if sky_ni.grid != sky_oi.grid:
            raise ValueError("Old and new sky model components have to match.")
        for key_oi in domain_keys(sky_oi):
            key_ni = key_oi.replace(sky_oi.prefix, sky_ni.prefix)
            ptree[key_ni] = domain_tree(samples)[key_oi]

    # load learned noise scaling of the previous iteration if available
    nm_old = lh_old["noise_model"]
    nm_new = lh_new["noise_model"]
    if isinstance(nm_old, NoiseModel) and isinstance(nm_new, NoiseModel) and not reset_noise:
        ptree[nm_new.prefix] = np.broadcast_to(
            domain_tree(samples)[nm_old.prefix], nm_new.target.shape
        )

    key, k_p = random.split(key)
    pos_new = random_init(
        k_p, [v for v in lh_new.values() if isinstance(v, Model)], ptree, factor=0.01
    )

    print("New model parameters:")
    for k, v in domain_tree(pos_new).items():
        print(f"  {k}:", v.shape)

    samples_new = MySamples(pos=pos_new, samples=None, keys=None)

    return samples_new


def transition_util(
    key,
    samples,
    it,
    *,
    func,
    lh_new,
    odir=None,
    **kwargs,
):
    """Cache-aware wrapper around a transition function.

    Loads a previously saved transition result if available.
    Otherwise, calls *func* and persists the result for future reuse.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    samples : MySamples
        Samples from the previous iteration.
    it : int
        Current optimisation iteration number.
    func : callable
        Transition function to call when no cache exists.
    lh_new : dict
        Likelihood dictionary for the new iteration.
    odir : str or None, optional
        Output directory for the cached transition file. Default is
        None.

    Returns
    -------
    samples : MySamples
        Samples initialised for the new iteration.
    """
    pos_fn = f"{odir}/{it}_trans.pkl" if odir else ""

    if odir:
        os.makedirs(odir, exist_ok=True)

    if os.path.isfile(pos_fn):
        with open(pos_fn, "rb") as f:
            samples = pickle.load(f)
        models = [v for v in lh_new.values() if isinstance(v, Model)]
        if domain_keys(samples) == domain_keys(models):
            return MySamples.from_samples(samples)

    samples, *_ = func(key, samples, it, lh_new=lh_new, odir=odir, **kwargs)

    if pos_fn:
        with open(pos_fn, "wb") as f:
            pickle.dump(samples, f)

    return samples


def transition_addt(
    key,
    samples,
    it,
    *,
    lh_old,
    lh_new,
    opt_dct,
    offsets=False,
    odir=None,
    mask=None,
    noise=None,
    plot_dct=None,
    style_dct=None,
    reset_noise=False,
    reuse=None,
    reuse_log=None,
    **kwargs,
):
    """Add components by fitting them to the previous reconstruction.

    Optimises the new background, point, object and tile models on the
    corresponding masked regions of the old reconstruction and assembles
    the initial parameter tree for the next optimisation iteration.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    samples : MySamples
        Samples from the previous iteration.
    it : int
        Current optimisation iteration number.
    lh_old : dict
        Likelihood dictionary for the previous iteration.
    lh_new : dict
        Likelihood dictionary for the new iteration.
    opt_dct : dict
        Optimisation parameters for the component fits.
    offsets : bool, optional
        If True, compute and return per-component offsets. Default is
        False.
    odir : str or None, optional
        Output directory for diagnostic plots. Default is None.
    mask : str or None, optional
        Path to a ``.npz`` mask file. A new mask is created when None.
        Default is None.
    noise : dict, optional
        Noise configuration for the sub-optimisations. Default uses
        ``max_std=1e-5``.
    plot_dct : dict, optional
        Plotting parameters. Default is ``{'norm': 'log'}``.
    style_dct : dict, optional
        Style of the pipeline plots (the ``base_style`` section, see ``split_style``).
        If given, the masks and the fit of every stage are plotted with ``plot_panels``
        (target and fit side by side, shared colorbar, step label), otherwise with
        ``plot_arrays``. Default is None.
    reset_noise : bool, optional
        Do not take over the learned noise scaling of the previous iteration: the new
        noise model starts at its prior. Default is ``False``.
    reuse : dict, optional
        Kept components of the model step (section ``reuse.{it}``: ``objects`` {new: old prefix},
        ``tiles`` [[new, old copy index], ...]). Kept components whose overlap group is unchanged
        and whose grid did not change are copied (latent parameters and offsets of the previous
        posterior) instead of fitted; the background is then fitted on the old reconstruction minus
        the copied components. Default is None.
    reuse_log : str, optional
        File for the copy decisions (JSON). Default is None.

    Returns
    -------
    samples_new : MySamples
        Samples initialised for the new iteration.
    ofs_dct : dict
        Per-component offset dictionary (empty when *offsets* is False).
    """
    if noise is None:
        noise = dict(max_std=1e-5, parameters=dict())
    if plot_dct is None:
        plot_dct = dict(norm="log")

    sky_old = lh_old["sky_model"]
    sky_new = lh_new["sky_model"]
    check_type(samples, MySamples)
    check_type(sky_old, (ComponentModel, SignalModel, PointModel, TileModel))
    check_type(sky_new, ComponentModel)
    plot_dct = plot_dct.copy() | dict(label=None, grid=None, odir=odir)

    # get reconstruction of the previous iteration
    rec_old = np.asarray(samples.mean(sky_old))

    # load or create masks for an efficient separation of the components
    mask_fn = mask if mask else ""
    if os.path.isfile(mask_fn):
        mask_dct = dict(np.load(mask))
    else:
        mask_dct = masks_from_model(sky_new)
    mask_box = masks_to_boxes(sky_new, mask_dct)
    if odir:
        mask_val = [remove_freq_axis(v, sky_new.freq) for v in mask_dct.values()]
        mask_val = [np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_val]
        if style_dct is not None:
            theme, _, pnl_dct = split_style(style_dct)
            group = dict(arrays=mask_val, labels=list(mask_dct), cmap=plot_dct.get("cmap", "inferno"), norm="linear", vmin=0, vmax=1)
            with plot_style(**theme):
                plot_panels([group], name=f"{it}_masks.png", odir=odir, step=f"pre-fit · it {it} · masks", **pnl_dct)
        else:
            p_dct = plot_dct | dict(name=f"{it}_masks.png", norm="linear", vmin=0, vmax=1)
            plot_arrays(mask_val, **p_dct)

    # initialize an empty position tree
    ptree = {}
    keys = list(random.split(key, 2 + len(sky_new.models)))

    # reuse: copy kept components with unchanged overlap groups, subtract them from the target
    copy = _copy_components(sky_old, sky_new, reuse, reuse_log) if reuse else {}
    copied_obj = {n: o for n, (o, a) in copy.items() if a == "copy" and "[" not in n}
    copied_tiles = {int(n.split("[")[1][:-1]): int(o.split("[")[1][:-1]) for n, (o, a) in copy.items() if a == "copy" and "[" in n}
    old_tree = domain_tree(samples) if copy else {}
    if (copied_obj or copied_tiles) and sky_old.grid != sky_new.grid:
        print("reuse in the pre-fit: the sky grid changed, nothing is copied")
        copied_obj, copied_tiles = {}, {}
    if copied_obj or copied_tiles:
        rec_old = np.asarray(rec_old) - _copied_image(samples, sky_old, sky_new, copy, mask_dct)

    # optimize the new background model on the old reconstruction (mask regions around point sources and the object boxes)
    sky_bg = sky_new.background.copy()
    rec_bg = map_signal(sky_old.grid, sky_bg.grid)(rec_old)
    msk_bg = mask_box[sky_bg.prefix]
    if copied_obj or copied_tiles:
        # the regions of copied components hold only the old background now: fit it there as well
        msk_bg = _background_mask(sky_new, sky_bg, mask_dct, copied_obj, copied_tiles)
    sky_bg.mask = msk_bg
    pos_bg = optimize_and_plot(
        key=keys.pop(),
        sky=sky_bg,
        data=np.where(msk_bg, rec_bg, 0.0),
        noise=noise.copy(),
        opt_dct=opt_dct,
        plot_dct=plot_dct | dict(name=f"{it}_{sky_bg.prefix}.png"),
        style_dct=style_dct,
        step=f"pre-fit · it {it} · {sky_bg.prefix}",
    )
    ptree |= pos_bg.tree

    sky_bg.mask = None
    rec_sub = map_signal(sky_old.grid, sky_new.grid)(rec_old) - map_signal(
        sky_bg.grid, sky_new.grid
    )(np.asarray(sky_bg(pos_bg)))
    rec_sub = rec_sub.clip(0, None)
    ofs_dct = {}

    # optimize the new object and tile models on the corresponding regions of the old reconstruction
    for sky_ci in sky_new.points + sky_new.objects + sky_new.tiles:
        if sky_ci.prefix in copied_obj:
            # copied: latent parameters of the old component, its offset is already in the model section
            ptree |= renamed_latents(old_tree, copied_obj[sky_ci.prefix], sky_ci.prefix)
            continue
        is_tiles = isinstance(sky_ci, TileModel)
        if is_tiles and copied_tiles and len(copied_tiles) == sky_ci.tiles.grid.n_copies:
            old_t = next(m for m in sky_old.models if isinstance(m, TileModel))
            init = {k: np.zeros(v.shape) for k, v in domain_tree(sky_ci).items()}
            ptree |= tile_slices(old_tree, old_t.prefix, sky_ci.prefix, init, copied_tiles)
            continue
        sub_ci = map_signal(sky_new.grid, sky_ci.grid)(rec_sub)
        msk_ci = mask_box[sky_ci.prefix]
        if offsets:
            ofs_dct[sky_ci.prefix] = get_offset(sky_ci, sub_ci, msk_ci, sky_ci.freq)
            if is_tiles and copied_tiles:
                # copied tiles keep the offsets of their model section
                ofs = [float(v) for v in np.atleast_1d(ofs_dct[sky_ci.prefix])]
                cur = np.atleast_1d(np.squeeze(np.asarray(sky_ci.tiles.offset)))
                for kn in copied_tiles:
                    ofs[kn] = float(cur[kn])
                ofs_dct[sky_ci.prefix] = ofs
            sky_ci.set_offset(ofs_dct[sky_ci.prefix])
        if isinstance(sky_ci, PointModel | TileModel):
            msk_ci = msk_ci.sum(axis=0).clip(0, 1)
        pos_ci = optimize_and_plot(
            key=keys.pop(),
            sky=sky_ci,
            data=np.where(msk_ci, sub_ci, 0.0),
            noise=noise.copy(),
            opt_dct=opt_dct,
            plot_dct=plot_dct | dict(name=f"{it}_{sky_ci.prefix}.png"),
            style_dct=style_dct,
            step=f"pre-fit · it {it} · {sky_ci.prefix}",
        )
        ptree |= pos_ci.tree
        if is_tiles and copied_tiles:
            old_t = next(m for m in sky_old.models if isinstance(m, TileModel))
            ptree |= tile_slices(old_tree, old_t.prefix, sky_ci.prefix, pos_ci.tree, copied_tiles)

    rec_sky = map_signal(sky_old.grid, sky_new.grid)(rec_old)
    pos_sky = optimize_and_plot(
        key=keys.pop(),
        sky=sky_new,
        data=rec_sky,
        noise=noise.copy(),
        pos=Vector(ptree),
        opt_dct=None,
        plot_dct=plot_dct | dict(name=f"{it}_{sky_new.prefix}.png"),
        style_dct=style_dct,
        step=f"pre-fit · it {it} · {sky_new.prefix}",
    )
    ptree = pos_sky.tree

    # load learned noise scaling of the previous iteration if available
    nm_old = lh_old["noise_model"]
    nm_new = lh_new["noise_model"]
    if isinstance(nm_old, NoiseModel) and isinstance(nm_new, NoiseModel) and not reset_noise:
        nm_val = domain_tree(samples)[nm_old.prefix]
        if np.shape(nm_val) == tuple(nm_new.model.target.shape):
            ptree[nm_new.prefix] = nm_val
        else:
            # the noise scaling of the fast likelihood lives on the Fourier grid of the sky: a zoom changes it, and
            # the values cannot be mapped like a sky image; the new noise model starts at its prior
            print(f"noise model {nm_new.prefix}: new shape (zoom), starts at its prior")

    key, k_p = random.split(key)
    pos_new = random_init(
        k_p, [v for v in lh_new.values() if isinstance(v, Model)], ptree, factor=0.01
    )

    print("New model parameters:")
    for k, v in domain_tree(pos_new).items():
        print(f"  {k}:", v.shape)

    samples_new = MySamples(pos=pos_new, samples=None, keys=None)

    return samples_new, ofs_dct



def _copy_components(sky_old, sky_new, reuse, reuse_log):
    """Copy decisions of the kept components: ``{new name: (old name, 'copy' | 'refit')}``."""
    old_t = next((m for m in sky_old.models if isinstance(m, TileModel)), None)
    new_t = next((m for m in sky_new.models if isinstance(m, TileModel)), None)
    pairs = kept_pairs(reuse, new_t.prefix if new_t else "", old_t.prefix if old_t else "")
    new_reg, old_reg = box_regions(sky_new, sky_new.grid), box_regions(sky_old, sky_new.grid)
    by_prefix_old = {m.prefix: m for m in sky_old.models}
    by_prefix_new = {m.prefix: m for m in sky_new.models}

    def grid_of(model):
        return model.tiles.grid if isinstance(model, TileModel) else model.grid

    same_grid = {}
    for n, o in pairs.items():
        gn, go = grid_of(by_prefix_new[n.split("[")[0]]), grid_of(by_prefix_old[o.split("[")[0]])
        same_grid[n] = bool(np.all(gn.fac == go.fac) and np.all(np.asarray(gn.shape) == np.asarray(go.shape)))
    plan = copy_plan(new_reg, old_reg, pairs, same_grid)
    decisions = {n: dict(old=pairs.get(n)) | d for n, d in plan.items()}
    n_copy = sum(d["action"] == "copy" for d in plan.values())
    print(f"reuse in the pre-fit: {n_copy} of {len(plan)} components copied")
    for n, d in decisions.items():
        print(f"  {n}: {d['action']}" + (f" = {d['old']}" if d["action"] == "copy" else f" ({d.get('reason', 'not kept')})"))
    if reuse_log:
        import json

        os.makedirs(os.path.dirname(reuse_log) or ".", exist_ok=True)
        with open(reuse_log, "w") as f:
            json.dump(decisions, f, indent=1)
    return {n: (pairs[n], d["action"]) for n, d in plan.items() if n in pairs}

def _copied_image(samples, sky_old, sky_new, copy, mask_dct):
    """Old reconstruction of the copied components on the new sky grid (tiles: only the copied ones)."""
    to_new = map_signal(sky_old.grid, sky_new.grid) if sky_old.grid != sky_new.grid else (lambda a: a)
    by_prefix = {m.prefix: m for m in sky_old.models}
    out = 0.0
    regions = None
    tiles = {}
    for o, action in copy.values():
        if action != "copy":
            continue
        if "[" not in o:
            out = out + np.asarray(to_new(np.asarray(samples.mean(lambda x, m=by_prefix[o]: m(x, map=True)))))
        else:
            tiles.setdefault(o.split("[")[0], []).append(o)
    for prefix, names in tiles.items():
        img = np.asarray(to_new(np.asarray(samples.mean(lambda x, m=by_prefix[prefix]: m(x, map=True)))))
        regions = box_regions(sky_old, sky_new.grid) if regions is None else regions
        sel = np.any([regions[o] for o in names], axis=0)
        out = out + np.where(sel, img, 0.0)
    return out


def _background_mask(sky_new, sky_bg, mask_dct, copied_obj, copied_tiles):
    """Background mask: everywhere except the masks of the refitted components (on the background grid)."""
    refit = []
    for k, v in mask_dct.items():
        if k in ("sum",) or k.startswith("bg"):
            continue
        v = remove_freq_axis(np.asarray(v), sky_new.freq)
        if v.ndim == 3:  # tiles / points: one mask per copy
            refit += [m for i, m in enumerate(v) if not (k.startswith("t") and i in copied_tiles)]
        elif k not in copied_obj:
            refit.append(v)
    msk = np.ones(sky_new.grid.shape) if not refit else 1.0 - np.clip(np.sum(refit, axis=0), 0, 1)
    msk = np.floor(np.asarray(map_signal(sky_new.grid, sky_bg.grid)(msk))) if sky_bg.grid != sky_new.grid else np.floor(msk)
    return np.asarray(msk.astype(bool))

def transition_zoom(
    key,
    samples,
    it,
    *,
    lh_old,
    lh_new,
    opt_dct,
    odir=None,
    noise=None,
    plot_dct=None,
    fit="kl",
    log_map=None,
    own_kl=None,
    flux_scale=1.0,
    reset_noise=False,
    style_dct=None,
    **kwargs,
):
    """Transfer parameters to a higher-resolution (zoomed) grid.

    Re-optimises each component on the zoomed grid using the
    reprojected reconstruction from the previous iteration.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    samples : MySamples
        Samples from the previous iteration.
    it : int
        Current optimisation iteration number.
    lh_old : dict
        Likelihood dictionary for the previous iteration.
    lh_new : dict
        Likelihood dictionary for the new iteration.
    opt_dct : dict
        Optimisation parameters for the component fits.
    odir : str or None, optional
        Output directory for diagnostic plots. Default is None.
    noise : dict, optional
        Noise configuration for the sub-optimisations. Default uses
        ``max_std=1e-5``.
    plot_dct : dict, optional
        Plotting parameters. Default is ``{'norm': 'log'}``.
    fit : {'kl', 'own_kl', 'log_map', 'log_kl'}, optional
        How the components on a new grid are fitted to the old reconstruction:
        ``'kl'`` (default): KL fit (``opt_dct``) of the new component, mapped onto the
        full sky grid, to the old one; ``'own_kl'``: as ``'kl'``, but tiles and point
        sources in their own space, per copy (see ``own_kl_fit``); ``'log_map'``: MAP fit in log space of every
        component in its own space, per copy (each tile and each point source in its
        own array, no superposition; see ``log_map_fit``); ``'log_kl'``: the
        same log-space fit of every component in its own space, but optimised with the
        KL (``opt_dct``, with samples) instead of MAP (see ``log_kl_fit``). Components on
        an unchanged grid with matching parameters are copied in all cases.
    log_map : dict, optional
        Keyword arguments of ``log_map_fit`` for ``fit='log_map'`` (``sigma``,
        ``rel``, ``maxiter``, ``cg_maxiter``), of ``log_kl_fit`` for ``fit='log_kl'``
        (``sigma``, ``rel``).
    own_kl : dict, optional
        Start help of ``own_kl_fit`` for ``fit='own_kl'``: ``level`` (default True) and
        ``taper`` (default False) as in ``own_kl_fit``, ``taper_rel`` for the tiles (default
        1, the brightest pixel), and ``signals``: prefixes of signal components (e.g.
        ``[o0, o1]``) that are also fitted with ``own_kl_fit`` (default none), with
        ``signal_taper_rel`` (default 0.01).
    flux_scale : float, optional
        Factor applied to the old reconstruction before the new components are fitted to
        it, for new data on a different flux scale. The offsets of the new sky components
        are expected to be shifted by ln(flux_scale) (``zoom_extension``), so that copied
        components are scaled as well. Default is 1.0.
    reset_noise : bool, optional
        If True, the learned noise scaling of the previous iteration is not copied and
        the new noise model starts at its prior (e.g. for new data or weights). Default is
        ``False``.
    style_dct : dict, optional
        Style of the pipeline plots (the ``base_style`` section, see ``split_style``).
        If given, target and fit of every component are plotted with ``plot_panels``
        (tiles as sub-grids, shared colorbar, step label), otherwise with
        ``plot_arrays``. Default is None.

    Returns
    -------
    samples_new : MySamples
        Samples initialised for the zoomed iteration.
    None
        Placeholder for consistency with ``transition_addt``.
    """
    if noise is None:
        noise = dict(max_std=1e-5, parameters=dict())
    if plot_dct is None:
        plot_dct = dict(norm="log")

    sky_old = lh_old["sky_model"]
    sky_new = lh_new["sky_model"]
    check_type(samples, MySamples)
    check_type(sky_old, ComponentModel)
    check_type(sky_new, ComponentModel)
    plot_dct = plot_dct.copy() | dict(label=None, grid=None, odir=odir)

    # initialize an empty position tree
    ptree = {}
    keys = list(random.split(key, len(sky_new.models) + 1))

    if fit not in ("kl", "own_kl", "log_map", "log_kl"):
        raise ValueError(f"Unknown fit `{fit}` of the zoom transition, use `kl`, `own_kl`, `log_map` or `log_kl`.")
    log_map = {} if log_map is None else dict(log_map)
    own_kl = {} if own_kl is None else dict(own_kl)
    own_signals = {str(p) for p in own_kl.pop("signals", ())}
    taper_rel = dict(tiles=own_kl.pop("taper_rel", 1.0), signals=own_kl.pop("signal_taper_rel", 0.01))

    # copy over matching model components
    for sky_oi, sky_ni in zip(sky_old.models, sky_new.models, strict=False):
        if sky_ni.grid not in sky_oi.grid and sky_oi.grid in sky_ni.grid:
            raise ValueError("Old and new sky model components have to match.")
        rec_oi = flux_scale * map_signal(sky_oi.grid, sky_ni.grid)(samples.mean(sky_oi))
        copied = {
            k.replace(sky_oi.prefix, sky_ni.prefix)
            for k in domain_keys(samples)
            if k.startswith(sky_oi.prefix)
        }
        same_params = copied == set(domain_keys(sky_ni))
        own_signal = isinstance(sky_ni, SignalModel) and sky_ni.prefix.split(".")[0] in own_signals
        own = fit == "own_kl" and (isinstance(sky_ni, (TileModel, PointModel)) or own_signal)
        if own and not (sky_ni.grid == sky_oi.grid and same_params):
            data_ni = flux_scale * own_to_new(sky_oi, sky_ni, samples.mean(partial(component_own, sky_oi)))
            print(f"zoom transition: KL fit of {sky_ni.prefix} in its own space {data_ni.shape}")
            pos_ci = own_kl_fit(
                keys.pop(),
                sky_ni,
                data_ni,
                opt_dct,
                max_std=noise.get("max_std", 1e-5),
                taper_rel=taper_rel["signals" if own_signal else "tiles"],
                **own_kl,
            )
            pos_ci = optimize_and_plot(  # plot only (no opt_dct)
                key=key,
                sky=sky_ni,
                data=rec_oi,
                pos=pos_ci,
                plot_dct=plot_dct | dict(name=f"{it}_{sky_ni.prefix}.png"),
                style_dct=style_dct,
                step=f"zoom transition · it {it} · {sky_ni.prefix}",
            )
        elif fit in ("log_map", "log_kl") and not (sky_ni.grid == sky_oi.grid and same_params):
            data_ni = flux_scale * own_to_new(sky_oi, sky_ni, samples.mean(partial(component_own, sky_oi)))
            how = "MAP" if fit == "log_map" else "KL"
            print(f"zoom transition: log-space {how} fit of {sky_ni.prefix} in its own space {data_ni.shape}")
            if fit == "log_map":
                pos_ci = log_map_fit(keys.pop(), sky_ni, data_ni, **log_map)
            else:
                pos_ci = log_kl_fit(keys.pop(), sky_ni, data_ni, opt_dct, **log_map)
            pos_ci = optimize_and_plot(  # plot only (no opt_dct)
                key=key,
                sky=sky_ni,
                data=rec_oi,
                pos=pos_ci,
                plot_dct=plot_dct | dict(name=f"{it}_{sky_ni.prefix}.png"),
                style_dct=style_dct,
                step=f"zoom transition · it {it} · {sky_ni.prefix}",
            )
        elif sky_ni.grid == sky_oi.grid:
            pos_ci = {
                k.replace(sky_oi.prefix, sky_ni.prefix): v
                for k, v in domain_tree(samples).items()
                if k.startswith(sky_oi.prefix)
            }
            pos_ci = optimize_and_plot(
                key=keys.pop(),
                sky=sky_ni,
                data=rec_oi,
                pos=Vector(pos_ci),
                plot_dct=plot_dct | dict(name=f"{it}_{sky_ni.prefix}.png"),
                style_dct=style_dct,
                step=f"zoom transition · it {it} · {sky_ni.prefix}",
            )
        else:
            pos_ci = optimize_and_plot(
                key=keys.pop(),
                sky=sky_ni,
                data=rec_oi,
                noise=noise.copy(),
                opt_dct=opt_dct,
                plot_dct=plot_dct | dict(name=f"{it}_{sky_ni.prefix}.png"),
                style_dct=style_dct,
                step=f"zoom transition · it {it} · {sky_ni.prefix}",
            )
        ptree |= pos_ci.tree

    rec_sky = flux_scale * map_signal(sky_old.grid, sky_new.grid)(samples.mean(sky_old))
    pos_sky = optimize_and_plot(
        key=keys.pop(),
        sky=sky_new,
        data=rec_sky,
        noise=noise.copy(),
        pos=Vector(ptree),
        opt_dct=None,
        plot_dct=plot_dct | dict(name=f"{it}_{sky_new.prefix}.png"),
        style_dct=style_dct,
        step=f"zoom transition · it {it} · {sky_new.prefix}",
    )
    ptree = pos_sky.tree

    # load learned noise scaling of the previous iteration if available
    nm_old = lh_old["noise_model"]
    nm_new = lh_new["noise_model"]
    if isinstance(nm_old, NoiseModel) and isinstance(nm_new, NoiseModel) and not reset_noise:
        nm_val = domain_tree(samples)[nm_old.prefix]
        if np.shape(nm_val) == tuple(nm_new.model.target.shape):
            ptree[nm_new.prefix] = nm_val
        else:
            # the noise scaling of the fast likelihood lives on the Fourier grid of the sky: a zoom changes it, and
            # the values cannot be mapped like a sky image; the new noise model starts at its prior
            print(f"noise model {nm_new.prefix}: new shape (zoom), starts at its prior")

    key, k_p = random.split(key)
    pos_new = random_init(
        k_p, [v for v in lh_new.values() if isinstance(v, Model)], ptree, factor=0.01
    )

    print("New model parameters:")
    for k, v in domain_tree(pos_new).items():
        print(f"  {k}:", v.shape)

    samples_new = MySamples(pos=pos_new, samples=None, keys=None)

    return samples_new, None


def component_own(model, x):
    """A sky component in its own space: tiles and point sources per copy, ``(n_copies, ...)``,
    without the mapping onto the sky grid; other components on their own grid."""
    if isinstance(model, (TileModel, PointModel)):
        return model(x, map=False)
    return model(x)


def own_to_new(old, new, values):
    """Map a component, given in its own space (``component_own``), from the old to the
    new (zoomed) component's own space.

    Signal components are mapped with ``map_signal``; tiles are zoomed per tile (same
    tile centres, integer resolution ratio); every refined point source takes the value
    of the old point source in whose coarse pixel it lies.
    """
    values = np.asarray(values)
    if isinstance(new, TileModel):
        check_type(old, TileModel)
        go, gn = old.tiles.grid, new.tiles.grid
        if go.n_copies != gn.n_copies:
            raise ValueError("The tiles of the old and new model have to match.")
        ratio = gn.fac / go.fac
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError(f"Tiles can only be zoomed by an integer factor, got {ratio}.")
        res = upsample(values, int(round(ratio)))
        if res.shape[-2:] != tuple(gn.shape[-2:]):
            raise ValueError(f"Zoomed old tiles {res.shape} do not match the new tiles {gn.shape}.")
        return res
    if isinstance(new, PointModel):
        check_type(old, PointModel)
        co = np.asarray(old.points.grid.coos).reshape(-1, 2)
        cn = np.asarray(new.points.grid.coos).reshape(-1, 2)
        parent = {tuple(c): i for i, c in enumerate(np.floor(co).astype(int).tolist())}
        try:
            idx = np.array([parent[tuple(c)] for c in np.floor(cn).astype(int).tolist()])
        except KeyError as e:
            raise ValueError(f"A new point source has no old point source in its coarse pixel {e}.") from e
        return values[idx]
    return np.asarray(map_signal(old.grid, new.grid)(values))


def _log_target(model, data, rel, floor=False):
    """Log data and log model of ``log_map_fit`` / ``log_kl_fit``, with the threshold ``rel`` times
    the maximum (per copy for tiles and point sources). ``floor=False``: only pixels above the
    threshold are kept (mask). ``floor=True``: all pixels, with the threshold ``f`` added to data
    and model, ``log(S + f)`` against ``log(d + f)``: above ``f`` as before, below ``f`` a model
    brighter than ``f`` is penalised, a fainter one is free (no flux is added)."""
    data = np.asarray(data, dtype="float64")
    shape = np.asarray(jax.eval_shape(model.log, model.init(random.PRNGKey(0))).shape)
    if tuple(shape) != data.shape:
        raise ValueError(f"The data {data.shape} do not match the component's own space {tuple(shape)}.")
    if isinstance(model, (TileModel, PointModel)):
        dmax = data.reshape(data.shape[0], -1).max(axis=1).reshape((-1,) + (1,) * (data.ndim - 1))
    else:
        dmax = data.max()
    if floor:
        log_f = jnp.asarray(np.log(np.broadcast_to(rel * dmax, data.shape)))
        log_d = jnp.logaddexp(jnp.asarray(np.log(np.maximum(data, np.finfo(np.float64).tiny))), log_f)
        return log_d, Model(lambda x: jnp.logaddexp(model.log(x), log_f), domain=model.domain, init=model.init)
    mask = jnp.asarray(data > rel * dmax, dtype="float64")
    log_d = jnp.asarray(np.log(np.maximum(data, np.finfo(np.float64).tiny))) * mask
    return log_d, Model(lambda x: model.log(x) * mask, domain=model.domain, init=model.init)


def _per_copy(model, a):
    """``a`` as ``(n_copies, -1)``: per tile / point source, one copy for other components."""
    a = np.asarray(a, dtype="float64")
    return a.reshape(a.shape[0] if isinstance(model, (TileModel, PointModel)) else 1, -1)


def start_taper(model, data, pos, frac=0.5, rel=1.0, steps=None):
    """Widen the Gaussian taper at the start until it is at least ``frac`` of its maximum at every
    pixel of the target above ``rel`` times its maximum (default: the brightest pixel), per tile
    for tiles: an off-centre source is visible from the start. The excitations of ``cov_x`` and
    ``cov_y`` are raised together in ``steps``; the prior is unchanged. Returns ``pos``
    unchanged if the model has no taper.
    """
    if steps is None:
        steps = np.arange(31.0)
    gm = getattr(model, "gaussian", None)
    keys = [k for k in pos.tree if k.endswith("cov_x") or k.endswith("cov_y")]
    if gm is None or len(keys) != 2:
        return pos
    d = _per_copy(model, data)
    n = d.shape[0]
    src = d >= rel * d.max(axis=1, keepdims=True)
    tree = dict(pos.tree)
    base = {k: np.asarray(tree[k]) for k in keys}
    best = np.full(n, steps[-1])
    for e in steps[::-1]:  # smallest step that reaches frac, per copy
        g = _per_copy(model, gm(Vector(dict(tree, **{k: base[k] + e for k in keys}))))
        g = g / g.max(axis=1, keepdims=True)
        reach = np.where(src, g, np.inf).min(axis=1) >= frac
        best = np.where(reach, e, best)
    for k in keys:
        tree[k] = jnp.asarray(base[k] + best.reshape(base[k].shape))
    return Vector(tree)


def start_level(key, model, data, pos=None, factor=0.01, clip=10.0):
    """Start position for the fit of tiles or point sources to ``data`` (own space, per copy)
    with the flux of every copy at the level of its target.

    The zero mode of every copy (domain key ending in ``zeromode``) is shifted such that
    ``sum(model)`` equals ``sum(data)`` per copy; the change of ``log(sum(model))`` per unit
    zero mode is measured, not assumed. A copy that starts far below its target has a
    vanishing gradient in a linear-space fit (``d S / d x = S``) and is never fitted.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Key of the random start (``factor`` times a prior draw) if ``pos`` is None.
    model : TileModel or PointModel
        The component.
    data : array-like
        Target in the component's own space, ``(n_copies, ...)``.
    pos : Vector, optional
        Start position to shift. Default is None.
    factor : float, optional
        Scale of the random start. Default is 0.01.
    clip : float, optional
        Bound of the zero-mode excitations. Default is 10.

    Returns
    -------
    Vector
        The start position (unchanged if the model has no zero mode).
    """
    if pos is None:
        pos = Vector(jax.tree_util.tree_map(lambda a: factor * a, dict(model.init(key))))
    elif not isinstance(pos, Vector):
        pos = Vector(dict(pos))
    zm = [k for k in pos.tree if k.endswith("zeromode")]
    if len(zm) != 1:
        return pos
    zm = zm[0]

    def log_flux(x):
        return np.log(np.maximum(_per_copy(model, component_own(model, x)).sum(axis=1), 1e-300))

    tree = dict(pos.tree)
    step = dict(tree, **{zm: tree[zm] + 1.0})
    slope = log_flux(Vector(step)) - log_flux(pos)
    target = np.log(np.maximum(_per_copy(model, data).sum(axis=1), 1e-300))
    shift = np.where(slope > 0, (target - log_flux(pos)) / np.where(slope > 0, slope, 1.0), 0.0)
    tree[zm] = jnp.clip(tree[zm] + shift.reshape(np.shape(tree[zm])), -clip, clip)
    return Vector(tree)


def own_kl_fit(
    key, model, data, opt_dct, pos=None, max_std=1e-5, level=True, taper=False, taper_rel=1.0, callback=None
):
    """KL fit of tiles or point sources to ``data`` in their own space, per copy (also usable for a
    signal component on its own grid, one copy).

    The KL fit of the zoom transition (``fit='kl'``: Gaussian likelihood with noise std
    ``max_std`` times the maximum, one noise realisation added to the data, schedule
    ``opt_dct``), but on ``model(x, map=False)``, shape ``(n_copies, ...)``, instead of the
    full sky grid, and with the maximum of every copy: no superposition of tiles and the
    same relative weighting for faint and bright tiles.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Key of the noise realisation and the KL optimisation.
    model : TileModel or PointModel
        The component.
    data : array-like
        Target in the component's own space (``component_own``).
    opt_dct : dict
        Optimisation schedule of ``optimize_kl``.
    pos : Vector, optional
        Start position. Default is None (``optimize_kl`` default).
    max_std : float, optional
        Noise std relative to the maximum of every copy. Default is 1e-5.
    level : bool, optional
        Start with the flux of every copy at its target (``start_level``). Default is True.
    taper : bool, optional
        Start with the taper (of every tile) wide enough for the target (``start_taper``,
        before the flux level). Default is False.
    taper_rel : float, optional
        ``rel`` of ``start_taper``: the taper covers the pixels above ``taper_rel`` times the
        maximum. Default is 1 (the brightest pixel).
    callback : callable, optional
        Passed to ``optimize_kl`` (``callback(samples, opt_state)``). Default is None.

    Returns
    -------
    Vector
        The fitted position (latent mean).
    """
    if not isinstance(model, (TileModel, PointModel, SignalModel)):
        raise TypeError(f"own_kl_fit fits tiles, point sources and signal components, got {type(model).__name__}.")
    data = np.asarray(data, dtype="float64")
    shape = jax.eval_shape(partial(component_own, model), model.init(random.PRNGKey(0))).shape
    if tuple(shape) != data.shape:
        raise ValueError(f"The data {data.shape} do not match the component's own space {tuple(shape)}.")
    dmax = _per_copy(model, data).max(axis=1)
    noise_std = max_std * np.broadcast_to(dmax.reshape((-1,) + (1,) * (data.ndim - 1)) if dmax.size > 1 else dmax[0], data.shape)
    k_n, k_o, k_p = random.split(key, 3)
    if taper:
        if pos is None:
            pos = Vector(jax.tree_util.tree_map(lambda a: 0.01 * a, dict(model.init(k_p))))
        pos = start_taper(model, data, pos, rel=taper_rel)
    if level:
        pos = start_level(k_p, model, data, pos)
    data = data + noise_std * np.asarray(random.normal(k_n, data.shape))
    lh_dct = dict(
        data=data,
        sky_response=Model(partial(component_own, model), domain=model.domain, init=model.init),
        noise_cov_inv=None,
        noise_std_inv=jnp.asarray(1.0 / noise_std),
        noise_model=None,
    )
    samples, _ = optimize_kl(
        lh_dct, key=k_o, position_or_samples=pos, callback=callback, **callable_optimize_dict(opt_dct)
    )
    return samples.pos


def log_kl_fit(key, model, data, opt_dct, pos=None, sigma=0.1, rel=1e-4, floor=True):
    """KL fit of a sky component to ``data`` in log space, in the component's own space.

    The likelihood of ``log_map_fit`` (Gaussian in ``log S`` with std ``sigma``, threshold
    ``rel`` times the maximum, per copy for tiles and point sources), optimised with
    ``optimize_kl`` and the schedule ``opt_dct`` (with samples) instead of MAP. The samples
    keep the amplitudes of the correlated fields in check. By default all pixels enter with
    the threshold as floor (``floor``): with a mask, the samples and the amplitude are free
    outside it and the KL puts flux there.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Key of the KL optimisation.
    model : SignalModel, TileModel or PointModel
        The component; ``model.log(x)`` must have the shape of ``data``.
    data : array-like
        Target in the component's own space (``component_own``), positive.
    opt_dct : dict
        Optimisation schedule of ``optimize_kl``.
    pos : Vector, optional
        Start position. Default is None (``optimize_kl`` default).
    sigma : float, optional
        Standard deviation in log space (relative deviation). Default is 0.1.
    rel : float, optional
        Threshold relative to the maximum. Default is 1e-4.
    floor : bool, optional
        Fit ``log(S + f)`` to ``log(d + f)`` on all pixels, ``f = rel * max`` (default),
        instead of leaving out the pixels below ``f``.

    Returns
    -------
    Vector
        The fitted position (latent mean).
    """
    log_d, log_s = _log_target(model, data, rel, floor)
    lh_dct = dict(data=log_d, sky_response=log_s, noise_cov_inv=None, noise_std_inv=1.0 / sigma, noise_model=None)
    samples, _ = optimize_kl(lh_dct, key=key, position_or_samples=pos, **callable_optimize_dict(opt_dct))
    return samples.pos


HYPER_KEYS = ("fluctuations", "flexibility", "asperity", "sigma")


def log_map_fit(
    key,
    model,
    data,
    pos=None,
    sigma=0.1,
    rel=1e-4,
    maxiter=100,
    cg_maxiter=200,
    miniter=10,
    xtol=1e-8,
    hyper_std=0.3,
    hyper_keys=HYPER_KEYS,
):
    """MAP fit of a sky component to ``data`` in log space, in the component's own space.

    Minimises ``0.5 * sum_mask ((log S(x) - log data) / sigma)^2 + 0.5 * |x / s|^2`` with
    Newton-CG (Gauss-Newton metric), where ``log S`` is ``model.log`` (without the
    exponential, see ``SignalModel.log``). Relative deviations count the same for faint
    and bright structure. Only pixels above ``rel`` times the maximum enter the
    likelihood, per copy for tiles and point sources, otherwise of the whole component.

    The prior std ``s`` is 1 except for the amplitude hyperparameters (``hyper_keys``),
    where it is ``hyper_std``. A MAP fit trades excitations for amplitude (amplitude
    times xi is degenerate, a large amplitude with small xi is cheaper under the prior);
    the samples of the following KL then scale with that amplitude and the sky overflows.
    The tighter prior keeps the amplitudes near the prior median.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Key for the start position (``0.1`` times a prior draw) if ``pos`` is None.
    model : SignalModel, TileModel or PointModel
        The component; ``model.log(x)`` must have the shape of ``data``.
    data : array-like
        Target in the component's own space (``component_own``), positive.
    pos : Vector, optional
        Start position.
    sigma : float, optional
        Standard deviation in log space (relative deviation). Default is 0.1.
    rel : float, optional
        Pixels below ``rel`` times the maximum are left out. Default is 1e-4.
    maxiter : int, optional
        Maximum number of Newton steps. Default is 100.
    cg_maxiter : int, optional
        Maximum number of CG steps per Newton step. Default is 200.
    miniter : int, optional
        Minimum number of Newton steps. Default is 10.
    xtol : float, optional
        Convergence threshold of the Newton step per parameter (Newton-CG stops when
        the step norm is below ``xtol`` times the number of parameters). Default is
        1e-8; NIFTy's default of 1e-5 stops large components (e.g. a 3072^2
        background, ~1e7 parameters) after the first step.
    hyper_std : float, optional
        Prior std of the excitations of the amplitude hyperparameters. Default is 0.3;
        1 is the standard prior.
    hyper_keys : tuple of str, optional
        Domain keys whose last word is in ``hyper_keys`` get the prior std ``hyper_std``.
        Default: ``fluctuations``, ``flexibility``, ``asperity`` (correlated fields) and
        ``sigma`` (spectral deviations).

    Returns
    -------
    Vector
        The fitted position.
    """
    log_d, log_s = _log_target(model, data, rel)
    lh = jft.Gaussian(log_d, noise_std_inv=lambda r: r / sigma).amend(log_s)

    hyper = set(hyper_keys)
    prec = Vector({k: 1.0 / hyper_std**2 if k.split()[-1] in hyper else 1.0 for k in model.domain})

    def energy(x):
        return lh(x) + 0.5 * jft.vdot(x, prec * x)

    def hessp(x, v):  # Gauss-Newton metric of the likelihood + prior precision
        return lh.metric(x, v) + prec * v

    if pos is None:
        pos = Vector(jax.tree_util.tree_map(lambda a: 0.1 * a, dict(model.init(key))))
    return jft.newton_cg(
        fun_and_grad=jax.jit(jax.value_and_grad(energy)),
        x0=pos,
        hessp=jax.jit(hessp),
        maxiter=maxiter,
        miniter=min(miniter, maxiter),
        xtol=xtol,
        name=f"LOGMAP {getattr(model, 'prefix', '')}".strip(),
        cg_kwargs=dict(name=None, maxiter=cg_maxiter),
        energy_reduction_factor=1e-3,
    )


def optimize_and_plot(
    key,
    sky,
    data,
    pos=None,
    opt_dct=None,
    noise=None,
    plot_dct=None,
    style_dct=None,
    step=None,
):
    """Optimise a sky model on given data and optionally plot the result.

    When *opt_dct* is provided the model is fitted to *data* with added
    noise.  Diagnostic plots are written when ``plot_dct['odir']`` is
    set.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    sky : Model
        Sky model to optimise.
    data : np.ndarray
        Target data array.
    pos : Vector or None, optional
        Initial parameter position. Default is None.
    opt_dct : dict or None, optional
        Optimisation configuration. Skips fitting when None.
    noise : dict, optional
        Noise configuration for the fit. Default uses ``max_std=1e-5``.
    plot_dct : dict, optional
        Plotting parameters including ``odir`` and ``name``.
    style_dct : dict, optional
        Style of the pipeline plots (``base_style``); if given, target and fit are
        plotted with ``plot_panels``, otherwise with ``plot_arrays``. Default is None.
    step : str, optional
        Pipeline step shown above the ``plot_panels`` plots. Default is None.

    Returns
    -------
    pos : Vector
        Fitted (or initial) parameter position.
    """
    if noise is None:
        noise = dict(max_std=1e-5, parameters=dict())
    if plot_dct is None:
        plot_dct = dict(odir=None, name=None)

    if opt_dct:
        max_std = noise.get("max_std", 1e-5)
        noise_model = NoiseModel.build(shape=data.shape, **noise)

        k_n, k_o = random.split(key)
        noise_std = max_std * np.max(data)
        noise_init = np.asarray(noise_std * random_like(k_n, sky.target))
        data = data + noise_init

        lh_dct = dict(
            data=data,
            sky_response=sky,
            noise_cov_inv=None,
            noise_std_inv=noise_std**-1,
            noise_model=noise_model,
        )

        if "callback" in opt_dct and opt_dct["callback"]:

            def callback(samples, opt_state, *_):
                p_dct = plot_dct | dict(
                    odir=plot_dct["odir"] + "/callback",
                    name=f"{opt_state.nit}_{sky.prefix}",
                )
                if style_dct is not None:
                    _plot_target_and_fit(
                        None, samples.mean(sky), p_dct, style_dct, f"{step} · KL {opt_state.nit}" if step else None
                    )
                else:
                    plot_arrays(samples.mean(sky), **p_dct)
        else:
            callback = None

        opt_dct = callable_optimize_dict(opt_dct)

        samples, _ = optimize_kl(
            lh_dct, key=k_o, position_or_samples=pos, callback=callback, **opt_dct
        )
        pos = samples.pos

    if plot_dct["odir"] and style_dct is not None:
        if isinstance(sky, SignalModel):
            sky.mask = None
        if isinstance(sky, TileModel):
            _plot_tiles_target_and_fit(sky, data, pos, plot_dct, style_dct, step)
        else:
            _plot_target_and_fit(data, sky(pos), plot_dct, style_dct, step)
    elif plot_dct["odir"]:
        [plot_dct.pop(k) for k in ["vmin", "vmax"] if k in plot_dct]
        if isinstance(sky, SignalModel):
            sky.mask = None

        arrays = []
        for a in (data, sky(pos)):
            if a.ndim == 2:
                arrays += [
                    a,
                ]
            elif a.ndim == 3:
                arrays += [a[i] for i in range(a.shape[0])]

        plot_arrays(
            array=arrays,
            rows=2,
            vmin=max(sky(pos).min(), 1),
            vmax=sky(pos).max(),
            **plot_dct,
        )

    return pos


def _tile_cutouts(sky, image):
    """Cutouts of an image on the grid of a TileModel at the boxes of its tiles (central frequency)."""
    image = np.asarray(image, dtype="float64")
    image = image[image.shape[0] // 2] if image.ndim == 3 else image
    lims0 = np.asarray(sky.grid.lims, dtype="float64")[:, 0]
    fac = np.asarray(sky.grid.fac, dtype="float64")
    shape = np.asarray(sky.tiles.grid.shape)
    lims = np.asarray(sky.tiles.grid.lims, dtype="float64").reshape((-1, 2, 2))
    cuts = []
    for lm in lims:
        i0, j0 = np.round((lm[:, 0] - lims0) * fac).astype(int)
        cuts.append(image[i0 : i0 + shape[0], j0 : j0 + shape[1]])
    return cuts


def _plot_tiles_target_and_fit(sky, data, pos, plot_dct, style_dct, step, max_sub=100):
    """Target and fit of a TileModel as two sub-grid panels of its tiles (at most the ``max_sub`` brightest)."""
    from .plot.panels import _capped

    fits = np.asarray(sky(pos, map=False), dtype="float64")
    if sky.tiles.grid.n_copies == 1:
        fits = fits[None]
    if fits.ndim == 4:  # (tiles, freq, x, y): central frequency
        fits = fits[:, fits.shape[1] // 2]
    targets = _tile_cutouts(sky, data)
    (targets,), (label,), (sel,) = _capped([targets], ["target"], max_sub)
    fits = list(fits) if sel is None else [fits[i] for i in sel]
    theme, drange, pnl_dct = split_style(style_dct)
    vmin, vmax = dynamic_limits(np.array(targets), drange, plot_dct.get("vmin"), plot_dct.get("vmax"))
    group = dict(
        arrays=[targets, fits], labels=[label, label.replace("target", "fit")], cmap=plot_dct.get("cmap", "inferno"),
        vmin=vmin, vmax=vmax,
    )
    pnl_dct.pop("cols", None)
    with plot_style(**theme):
        plot_panels([group], name=plot_dct["name"], odir=plot_dct["odir"], step=step, cols=2, **pnl_dct)


def _plot_target_and_fit(data, fit, plot_dct, style_dct, step):
    """Target (if given) and fit of a transition stage side by side with one shared colorbar."""
    theme, drange, pnl_dct = split_style(style_dct)
    fit = np.asarray(fit, dtype="float64")
    fits = list(fit) if fit.ndim == 3 else [fit]
    if data is not None:
        data = np.asarray(data, dtype="float64")
        targets = list(data) if data.ndim == 3 else [data]
        arrays, labels = [], []
        for f, (d, m) in enumerate(zip(targets, fits, strict=True)):
            sfx = f" ch {f}" if len(fits) > 1 else ""
            arrays += [d, m]
            labels += [f"target{sfx}", f"fit{sfx}"]
        reference = np.array(targets)
    else:
        arrays, labels, reference = fits, ["fit"] * len(fits), np.array(fits)
    vmin, vmax = dynamic_limits(reference, drange, plot_dct.get("vmin"), plot_dct.get("vmax"))
    group = dict(arrays=arrays, labels=labels, cmap=plot_dct.get("cmap", "inferno"), vmin=vmin, vmax=vmax)
    cols = pnl_dct.pop("cols", None) or (2 if data is not None else None)
    with plot_style(**theme):
        plot_panels([group], name=plot_dct["name"], odir=plot_dct["odir"], step=step, cols=cols, **pnl_dct)

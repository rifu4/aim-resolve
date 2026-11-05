"""Script for building the sky model from detected sources between reconstruction iterations."""

import copy
import json
import os
import sys

import numpy as np

from aim_resolve import (
    ImageData,
    SetupKLConfig,
    SignalGrid,
    component_boxes,
    dynamic_limits,
    map_signal,
    masks_from_maps,
    model_background,
    model_objects,
    model_points,
    model_tiles,
    plot_panels,
    plot_style,
    remove_freq_axis,
    split_style,
    yaml_load,
)
from aim_resolve.mask import add_freq_axis
from aim_resolve.reuse import component_masks, match_components, upsample_mask


def zoom_of_it(zoom, it):
    """Zoom factor of iteration `it` from a single factor or a list with one factor per iteration.

    The grids can only be refined by whole factors: every factor of the list has to be a multiple
    of the one before (e.g. [1, 2, 4] or [1, 1, 3], not [1, 2, 3], where 2 -> 3 is a step of 1.5).
    """
    if isinstance(zoom, list | tuple):
        zoom = [int(z) for z in zoom]
        for z0, z1 in zip(zoom[:-1], zoom[1:], strict=True):
            if z1 < z0 or z1 % z0:
                raise ValueError(f"zoom steps have to be whole factors, got {z0} -> {z1} in base_model.zoom {zoom}")
        return zoom[min(it, len(zoom) - 1)]
    return int(zoom)


def reuse_components(mask_dct, prev_secs, mdl_dct, odir, it, zoom, zoom_it, freq):
    """Match the new components with those of the previous iteration (base_model.reuse).

    A matched object or tile gets the mask of the old one (so box and mask stay the same) and is
    later written with the old box and offset. The kept components go to the section reuse.{it} of
    the model file (see write_reuse), all decisions with their values to {odir}/reuse/{it}_reuse.json.

    Returns
    -------
    dict
        ``objects``: {new prefix: old prefix}, ``tiles``: {new copy index: old copy index},
        ``n``: number of kept components.
    """
    out = dict(objects={}, tiles={}, n=0)
    opts = dict(mdl_dct.get("reuse") or {})
    msk_file = f"{odir}/files/{it - 1}_msk.npz"
    if not opts.get("enabled", False) or it < 2 or not os.path.isfile(msk_file):
        return out

    def plain(v):
        return remove_freq_axis(np.asarray(v), freq)

    old = {k: upsample_mask(plain(v), zoom) for k, v in np.load(msk_file).items() if k.endswith(f".{it - 1}")}
    new = {k: plain(v) for k, v in mask_dct.items()}
    # the core of a new mask has to keep this margin inside the old box (base pixels, scaled with the zoom)
    margin = float(opts.get("fit_margin", 0.5)) * mdl_dct["masks"].get("margin_min", 2) * zoom_it
    match = match_components(
        component_masks(new, it), component_masks(old, it - 1), margin,
        iou_min=float(opts.get("iou_min", 0.7)), area_min=float(opts.get("area_min", 0.7)),
    )
    for name, res in match.items():
        if res["old"] is None:
            continue
        if name.startswith("o") and f"sky_{res['old']}" in prev_secs:
            out["objects"][name] = res["old"]
            mask_dct[name] = old[res["old"]] if len(freq) == 1 else add_freq_axis(old[res["old"]], freq)
        elif name.startswith("t") and f"sky_t0.{it - 1}" in prev_secs:
            k, j = int(name.split("[")[1][:-1]), int(res["old"].split("[")[1][:-1])
            out["tiles"][k] = j
            tile = old[f"t0.{it - 1}"][j]
            mask_dct[name.split("[")[0]][k] = tile if len(freq) == 1 else add_freq_axis(tile, freq)
        else:
            res["old"], res["reason"] = None, "kind not handled (points)"
    out["n"] = len(out["objects"]) + len(out["tiles"])
    if out["n"]:
        # the masks changed: update the background mask
        mask_dct["sum"] = np.sum(
            [np.sum(v, axis=0) if v.ndim == 3 else v for k, v in mask_dct.items() if k not in ("sum", f"bg.{it}")], axis=0
        )
        mask_dct[f"bg.{it}"] = np.floor(1 - mask_dct["sum"]).clip(0, 1)
    print(f"reuse: {out['n']} of {len(match)} components kept from iteration {it - 1}")
    for name, res in match.items():
        print(f"  {name}: " + (f"= {res['old']}" if res["old"] else "new") + f" ({ {k: v for k, v in res.items() if k != 'old'} })")
    os.makedirs(f"{odir}/reuse", exist_ok=True)
    with open(f"{odir}/reuse/{it}_reuse.json", "w") as f:
        json.dump(dict(objects=out["objects"], tiles={str(k): v for k, v in out["tiles"].items()}, decisions=match), f, indent=1)
    return out


def write_reuse(cfg, reuse, it):
    """Section reuse.{it} of the model file: kept objects {new prefix: old prefix} and kept tiles as
    [new copy index, old copy index] pairs (read by the pre-fit, never built as a model)."""
    if reuse["n"]:
        cfg.add_sec(
            f"reuse.{it}",
            objects=dict(reuse["objects"]),
            tiles=[[int(k), int(j)] for k, j in sorted(reuse["tiles"].items())],
        )


def reused_section(prev_secs, old_prefix, zoom_it):
    """Section of an object that keeps its old box: old grid (factor of the current zoom), prior, taper and offset."""
    dct = copy.deepcopy(prev_secs[f"sky_{old_prefix}"])
    dct.pop("prefix", None)
    dct["grid"] = dict(dct["grid"]) | dict(factor=zoom_it)
    return dct


def reuse_tiles(ti_dct, prev_secs, tiles):
    """Kept tiles get the center and offset of the old tile ({new copy index: old copy index})."""
    old = next(v for k, v in prev_secs.items() if k.startswith("sky_t0."))
    centers, offsets = old["tile_grid"]["center"], old["offset"]
    for k, j in tiles.items():
        ti_dct["tile_grid"]["center"][k] = copy.deepcopy(centers[j])
        ti_dct["offset"][k] = offsets[j]


def main():
    """Build sky model components from detection maps and update the configuration for the next iteration."""
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, opt_pkl, det_npz, base_yml, it = files

    # load the model yaml-file from the previous iteration to the SetupKLConfig class and add one iteration
    cfg = SetupKLConfig.from_file(mdl_yml)
    prev_secs = {k: copy.deepcopy(v) for k, v in cfg.sections.items() if k.startswith("sky_") and k.endswith(f".{cfg.it}")}
    cfg.sections["opt"]["resume"] = cfg.sections["opt"]["odir"]
    cfg.sections["opt"]["odir"] = cfg.sections["opt"]["odir"].replace(
        f"{cfg.it}_rec", f"{cfg.it + 1}_rec"
    )
    cfg.add_it(fix_keys=["data"], del_comp=True)

    # load the base yaml-file and get the base modelling settings, output directory and iteration number
    base_dct = yaml_load(base_yml)
    mdl_dct = base_dct.pop("base_model")
    plt_dct = base_dct["base_plot"]
    theme, drange, pnl_dct = split_style(base_dct.get("base_style"))
    sky_dct = dict(
        dynamic_range=drange, vmin=plt_dct.get("vmin"), vmax=plt_dct.get("vmax"), cmap=plt_dct.get("cmap", "inferno")
    )
    odir = base_dct["base_opt"]["odir"]
    cfg.modify_sec(f"opt.{it}", base="base_opt.n")

    # zoom factor of this iteration: base_model.zoom is one factor for all iterations or a list with one
    # factor per iteration (the last entry holds for later iterations)
    zoom_it = zoom_of_it(mdl_dct["zoom"], int(it))

    # update fast-resolve kernels depending on the new resolution
    if cfg.sections[f"lh.{it}"]["mode"] == "fast":
        pkdir = "_".join(cfg.sections[f"lh.{it}"]["psf_kernel_fn"].split("_")[:-1])
        nkdir = "_".join(cfg.sections[f"lh.{it}"]["n_inv_kernel_fn"].split("_")[:-1])
        ksize = zoom_it * base_dct["grid_bg"]["space"][0]
        cfg.modify_sec(
            f"lh.{it}",
            psf_kernel_fn=f"{pkdir}_{ksize}.pkl",
            n_inv_kernel_fn=f"{nkdir}_{ksize}.pkl",
        )

    # load the reconstructed image and setup the model grid
    rec = ImageData.load(opt_pkl, dtype="float32")

    # load the detected point sources and extended objects
    det_dct = dict(np.load(det_npz))
    ps_map = det_dct["ps_map"].astype(float)
    cl_map = det_dct["cl_map"].astype(float)

    # adjust the zoom level of the sky model and zoom the reconstructed image if necessary
    if zoom_it < rec.grid.factor or zoom_it % rec.grid.factor:
        raise ValueError(
            f"zoom {zoom_it} of iteration {it} is not a whole multiple of the zoom {rec.grid.factor} of the previous one"
        )
    zoom = zoom_it // rec.grid.factor
    grid = rec.grid.refine(zoom)
    freq = cfg.sections[f"sky_bg.{it}"]["freq"]
    if zoom > 1:
        rec.val = map_signal(rec.grid, grid)(rec.val)
        ps_map = map_signal(rec.grid, grid)(ps_map)
        cl_map = map_signal(rec.grid, grid)(cl_map)

    # create a mask for the detected point sources and each extended object
    if mdl_dct["tiles"] and mdl_dct["tiles"]["tile_size"]:
        mdl_dct["masks"] |= {"tile_size": mdl_dct["tiles"]["tile_size"]}
    mask_dct = masks_from_maps(
        ps_map, cl_map, it, freq, zoom_it, **mdl_dct["masks"]
    )

    # stable boxes: new components that are the same source as an old one keep the old box (and offset)
    reuse = reuse_components(mask_dct, prev_secs, mdl_dct, odir, int(it), zoom, zoom_it, freq)
    write_reuse(cfg, reuse, int(it))

    bg_dct = model_background(mask_dct[f"bg.{it}"], freq, rec.val)
    cfg.modify_sec(f"sky_bg.{it}", merge_base=True, **bg_dct)
    # optionally the background on the zoomed grid as well (otherwise only the components are refined)
    if mdl_dct.get("zoom_bg", False) and zoom_it > 1:
        bg_grid = SignalGrid.build(**base_dct["grid_bg"]).refine(zoom_it)
        cfg.modify_sec(f"sky_bg.{it}", grid=bg_grid.to_dict())
    rec_sub = rec.val - np.exp(bg_dct["offset"])
    rec_sub = rec_sub.clip(0, None)

    if mdl_dct["points"]:
        for pi in filter(lambda x: "p" in x, mask_dct):
            pi_dct = model_points(mask_dct[pi], ps_map, grid, freq, rec_sub)
            if pi_dct:
                cfg.add_sec(f"sky_{pi}", prefix=pi, **pi_dct)
                cfg.modify_sec(f"sky.{it}", **{pi: f"=sky_{pi}"})

    if mdl_dct["objects"]:
        for oi in filter(lambda x: "o" in x, mask_dct):
            if oi in reuse["objects"]:
                oi_dct = reused_section(prev_secs, reuse["objects"][oi], zoom_it)
            else:
                oi_dct = model_objects(
                    mask_dct[oi], grid, freq, rec_sub, **mdl_dct["objects"]
                )
            if oi_dct:
                cfg.add_sec(f"sky_{oi}", prefix=oi, **oi_dct)
                cfg.modify_sec(f"sky.{it}", **{oi: f"=sky_{oi}"})

    if mdl_dct["tiles"]:
        for ti in filter(lambda x: "t" in x, mask_dct):
            ti_dct = model_tiles(mask_dct[ti], grid, freq, rec_sub, **mdl_dct["tiles"])
            if ti_dct and reuse["tiles"]:
                reuse_tiles(ti_dct, prev_secs, reuse["tiles"])
            if ti_dct:
                cfg.add_sec(f"sky_{ti}", prefix=ti, **ti_dct)
                cfg.modify_sec(f"sky.{it}", **{ti: f"=sky_{ti}"})

    # get the positions of the detected point sources and the boxes of the extended objects and plot them as markers
    px, py = np.argwhere(ps_map == 1).T
    # plot the model components: circles around the point sources, thin boxes around tiles and objects
    radius = max(4.0, 0.006 * max(rec.val.shape))
    tiles, objects = component_boxes(cfg.sections, grid, it)
    vmin, vmax = dynamic_limits(rec.val, sky_dct["dynamic_range"], sky_dct["vmin"], sky_dct["vmax"])
    overlay = dict(circles=[(x, y, radius) for x, y in zip(px, py, strict=True)], boxes=tiles + objects)
    mask_val = [remove_freq_axis(v, freq) for v in mask_dct.values()]
    with plot_style(**theme):
        plot_panels(
            [
                dict(
                    arrays=[rec.val],
                    labels=[
                        f"{len(px)} points, {len(objects)} objects, {len(tiles)} tiles"
                        + (f" ({reuse['n']} kept)" if reuse["n"] else "")
                    ],
                    overlays=[overlay],
                    cmap=sky_dct["cmap"],
                    vmin=vmin,
                    vmax=vmax,
                )
            ],
            name=f"{it}_mdl.png",
            odir=f"{odir}/plots",
            step=f"model · it {it}",
            **pnl_dct,
        )

        # plot the point source and object masks with one shared colorbar
        plot_panels(
            [
                dict(
                    arrays=[np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_val],
                    labels=list(mask_dct),
                    cmap=sky_dct["cmap"],
                    norm="linear",
                    vmin=0,
                    vmax=1,
                )
            ],
            name=f"{it}_msk.png",
            odir=f"{odir}/plots",
            step=f"model masks · it {it}",
            **pnl_dct,
        )
    np.savez(f"{odir}/files/{it}_msk", **mask_dct)

    # save the new model yaml-file
    cfg.to_file(f"{odir}/files/{it}_mdl.yml")


if __name__ == "__main__":
    main()

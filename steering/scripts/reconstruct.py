"""Script for running variational inference reconstruction and plotting results."""

import os
import pickle
import sys

import jax

from aim_resolve import (
    ImageData,
    Observation,
    OptimizeKLConfig,
    dynamic_limits,
    get_builders,
    plot_components,
    plot_panels,
    plot_style,
    split_style,
    yaml_load,
)

jax.config.update("jax_enable_x64", True)


def main():
    """Run the KL optimization for sky model reconstruction and generate diagnostic plots."""
    print(jax.devices())
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, tra_pkl, base_yml, it = files

    # load model and base yaml-files and extract output directory
    base_dct = yaml_load(base_yml)
    plt_dct = base_dct["base_plot"]
    theme, drange, pnl_dct = split_style(base_dct.get("base_style"))
    sky_dct = dict(
        dynamic_range=drange, vmin=plt_dct.get("vmin"), vmax=plt_dct.get("vmax"), cmap=plt_dct.get("cmap", "inferno")
    )
    odir = base_dct["base_opt"]["odir"]

    # initialize the sky model using the OptimizeKLConfig class
    cfg = OptimizeKLConfig.from_file([base_yml, mdl_yml], get_builders)

    # instantiate the sky models of the current iteration
    sky_mdl = cfg.instantiate_sec(f"sky.{it}")
    # sky and all its components (also a single background component) for the panel plots
    panel_models = [sky_mdl] + list(sky_mdl.models)

    # print and plot the data
    if int(it) == 0:
        data = cfg.instantiate_sec(f"data.{it}")
        print(data, "\n")
        if isinstance(data, ImageData):
            images = [("data", "0_data.png", data.noisy_val.clip(0, None)), ("truth", "0_truth.png", data.val)]
        elif isinstance(data, Observation):
            d_val = data.dirty_image(sky_mdl.grid)
            images = [("data", "0_data.png", d_val[d_val.shape[0] // 2] if d_val.ndim == 3 else d_val)]
        else:
            images = []
        with plot_style(**theme):
            for step, nm, d_val in images:
                vmin, vmax = dynamic_limits(d_val, drange)
                plot_panels(
                    [dict(arrays=[d_val], cmap=sky_dct["cmap"], vmin=vmin, vmax=vmax)],
                    name=nm,
                    odir=f"{odir}/plots",
                    step=step,
                    **pnl_dct,
                )

    # define a callback function to plot the results of the optimization after each iteration
    def callback(samples, state, *args):
        nit = args[0] if len(args) > 0 else state.nit
        with plot_style(**theme):
            plot_components(
                panel_models,
                samples,
                name=f"{nit}_sky.png",
                odir=f"{odir}/callback",
                step=f"reconstruct · it {it} · KL {nit}",
                **sky_dct,
                **pnl_dct,
            )

    # optimize the sky model. Start from pre-fit positions if available
    rec_pkl = os.path.join(cfg.sections["opt"]["odir"], "last.pkl")
    if not base_dct["base_opt"]["rerun"] and os.path.isfile(rec_pkl):
        samples, *_ = pickle.load(open(rec_pkl, "rb"))
    else:
        (samples, _) = (
            pickle.load(open(tra_pkl, "rb")) if os.path.isfile(tra_pkl) else (None, _)
        )
        # use all visible GPUs (cuda_device=0,1,... in the snakefile)
        visible_devices = jax.devices()
        samples, *_ = cfg.optimize_kl(
            position_or_samples=samples,
            callback=callback if os.path.isdir(odir + "/callback/") else None,
            devices=visible_devices if len(visible_devices) > 1 else None,
        )

    # plot the final results of the optimization: the full sky ({it}_rec) and, if there are components
    # besides the background, the sky and all components with one shared colorbar ({it}_sep)
    with plot_style(**theme):
        plot_components(
            [sky_mdl],
            samples,
            name=f"{it}_rec.png",
            odir=f"{odir}/plots",
            step=f"reconstruct · it {it}",
            **sky_dct,
            **pnl_dct,
        )
        if len(sky_mdl.models) > 1:
            plot_components(
                panel_models,
                samples,
                name=f"{it}_sep.png",
                odir=f"{odir}/plots",
                step=f"reconstruct · it {it} · components",
                **sky_dct,
                **pnl_dct,
            )

    # save the reconstructed sky model and grid to a pkl-file
    rec_val = samples.mean(sky_mdl)
    rec_val = rec_val[rec_val.shape[0] // 2] if rec_val.ndim == 3 else rec_val
    rec = ImageData(rec_val, sky_mdl.grid, f"{it}_rec")
    rec.save(name=f"{it}_rec", odir=f"{odir}/files")


if __name__ == "__main__":
    main()

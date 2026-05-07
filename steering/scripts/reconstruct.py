"""Script for running variational inference reconstruction and plotting results."""

import os
import pickle
import sys

import jax

from aim_resolve import (
    ImageData,
    Observation,
    OptimizeKLConfig,
    get_builders,
    plot_agreement,
    plot_arrays,
    plot_mean_and_std,
    plot_pullplot,
    plot_samples,
    yaml_load,
)

jax.config.update("jax_enable_x64", True)


def main():
    """Run the KL optimization for sky model reconstruction and generate diagnostic plots."""
    print(jax.devices())
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, tra_pkl, base_yml, it = files

    # load model and base yaml-files and extract output directory
    mdl_dct = yaml_load(mdl_yml)
    base_dct = yaml_load(base_yml)
    plt_dct = base_dct["base_plot"]
    odir = base_dct["base_opt"]["odir"]

    # initialize the sky model using the OptimizeKLConfig class
    fun = mdl_dct[f"lh.{it}"]["fun"]
    cfg_mode = "major" if "radio" in fun and "fast" in fun else "total"
    cfg = OptimizeKLConfig.from_file([base_yml, mdl_yml], get_builders, cfg_mode)

    # instantiate the sky models of the current iteration
    sky_mdl = cfg.instantiate_sec(f"sky.{it}")
    sky_models = [
        sky_mdl,
    ] + [m for m in sky_mdl.models if len(sky_mdl.models) > 1]

    # print and plot the data
    if int(it) == 0:
        data = cfg.instantiate_sec(f"data.{it}")
        print(data, "\n")
        if isinstance(data, ImageData):
            d_val = data.noisy_val.clip(0, None)
            plot_arrays(
                d_val, data.grid, "data", "0_data.png", f"{odir}/plots", **plt_dct
            )
            plot_arrays(
                data.val, data.grid, "truth", "0_truth.png", f"{odir}/plots", **plt_dct
            )
        elif isinstance(data, Observation):
            d_val = data.dirty_image(sky_mdl.grid)
            p_dct = plt_dct | dict(vmin=None, vmax=None)
            plot_arrays(
                d_val, sky_mdl.grid, "data", "0_data.png", f"{odir}/plots", **p_dct
            )

    # define a callback function to plot the results of the optimization after each iteration
    def callback(samples, state, *args):
        nit = args[0] if len(args) > 0 else state.nit
        plot_mean_and_std(
            model=sky_mdl,
            samples=samples,
            freq=True,
            name=f"{nit}_sky.png",
            odir=f"{odir}/callback",
            **plt_dct,
        )
        plot_mean_and_std(
            model=[md for md in sky_models],
            samples=samples,
            mode="mean",
            name=f"{nit}_components.png",
            odir=f"{odir}/callback",
            **plt_dct,
        )

    # optimize the sky model. Start from pre-fit positions if available
    rec_pkl = os.path.join(cfg.sections["opt"]["odir"], "last.pkl")
    if not base_dct["base_opt"]["rerun"] and os.path.isfile(rec_pkl):
        samples, *_ = pickle.load(open(rec_pkl, "rb"))
    else:
        (samples, _) = (
            pickle.load(open(tra_pkl, "rb")) if os.path.isfile(tra_pkl) else (None, _)
        )
        samples, *_ = cfg.optimize_kl(
            position_or_samples=samples,
            callback=callback if os.path.isdir(odir + "/callback/") else None,
        )

    # plot the final results of the optimization
    plot_mean_and_std(
        model=[md for md in sky_models],
        samples=samples,
        mode="mean",
        name=f"{it}_rec.png",
        odir=f"{odir}/plots",
        **plt_dct,
    )

    # save the reconstructed sky model and grid to a pkl-file
    rec_val = samples.mean(sky_mdl)
    rec_val = rec_val[rec_val.shape[0] // 2] if rec_val.ndim == 3 else rec_val
    rec = ImageData(rec_val, sky_mdl.grid, f"{it}_rec")
    rec.save(name=f"{it}_rec", odir=f"{odir}/files")

    # extra plots: mean (and std) of the sky model components
    if os.path.isdir(odir + "/extra/"):
        for md in sky_models:
            pf, it = md.prefix.split(".")[0], md.prefix.split(".")[1]
            for mi in ["mean", "std"]:
                plot_mean_and_std(
                    model=md,
                    samples=samples,
                    mode=mi,
                    freq=True,
                    name=f"{it}_{pf}_{mi}.png",
                    odir=f"{odir}/extra",
                    **plt_dct,
                )
        plot_samples(
            model=sky_mdl,
            samples=samples,
            name=f"{it}_sky_samples.png",
            odir=f"{odir}/extra",
            **plt_dct,
        )

        if sky_mdl.freq.size > 1:
            frq_mdl = sky_mdl.spectral_index
            frq_models = [
                frq_mdl,
            ] + [m for m in frq_mdl.models if len(frq_mdl.models) > 1]
            p_dct = plt_dct | dict(norm="linear", vmin=None, vmax=None)
            for md in frq_models:
                pf, it = md.prefix.split(".")[0], md.prefix.split(".")[1]
                for mi in ["mean", "std"]:
                    plot_mean_and_std(
                        model=md,
                        samples=samples,
                        mode=mi,
                        name=f"{it}_{pf}_alpha_{mi}.png",
                        odir=f"{odir}/extra",
                        **p_dct,
                    )
            plot_samples(
                model=frq_mdl,
                samples=samples,
                name=f"{it}_alpha_samples.png",
                odir=f"{odir}/extra",
                **p_dct,
            )

        data = cfg.instantiate_sec("data.0")
        if isinstance(data, ImageData):
            plot_agreement(
                model=sky_mdl,
                samples=samples,
                data=data,
                name=f"{it}_sky_agreement.png",
                odir=f"{odir}/extra",
                **plt_dct,
            )
            plot_pullplot(
                model=sky_mdl,
                samples=samples,
                data=data,
                name=f"{it}_sky_pullplot.png",
                odir=f"{odir}/extra",
                **plt_dct,
            )


if __name__ == "__main__":
    main()

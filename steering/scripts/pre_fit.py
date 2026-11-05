"""Script for performing pre-fit transitions between reconstruction iterations."""

import os
import pickle
import sys

import jax

from aim_resolve import (
    OptimizeKLConfig,
    SetupKLConfig,
    get_builders,
    plot_components,
    plot_style,
    split_style,
    transition_addt,
    yaml_load,
)

jax.config.update("jax_enable_x64", True)


def main():
    """Run the additive transition from old to new sky model and update component offsets."""
    print(jax.devices())
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, rec_pkl, base_yml, it = files

    # load the base yaml-file and extract output directory and plotting settings
    base_dct = yaml_load(base_yml)
    plt_dct = base_dct["base_plot"]
    theme, drange, pnl_dct = split_style(base_dct.get("base_style"))
    sky_dct = dict(
        dynamic_range=drange, vmin=plt_dct.get("vmin"), vmax=plt_dct.get("vmax"), cmap=plt_dct.get("cmap", "inferno")
    )
    odir = base_dct["base_opt"]["odir"]

    # initialize the sky model using the OptimizeKLConfig class (mode is taken from the lh sections)
    cfg = OptimizeKLConfig.from_file([base_yml, mdl_yml], get_builders)

    # instantiate the sky model of the current iteration
    sky_mdl = cfg.instantiate_sec(f"sky.{it}")

    # perform the transition from the old to the new sky model
    tra_pkl = f"{odir}/opt/{it}_pre/last.pkl"
    if not base_dct["base_opt"]["rerun"] and os.path.isfile(tra_pkl):
        samples, offsets = pickle.load(open(tra_pkl, "rb"))
    else:
        samples, offsets = transition_addt(
            key=jax.random.PRNGKey(cfg.sections["opt"]["key"]),
            samples=pickle.load(open(rec_pkl, "rb"))[0],
            it=it,
            lh_old=cfg.instantiate_sec(f"lh.{int(it) - 1}"),
            lh_new=cfg.instantiate_sec(f"lh.{it}"),
            sky_old=cfg.instantiate_sec(f"sky.{int(it) - 1}"),
            sky_new=sky_mdl,
            offsets=True,
            opt_dct=cfg.sections["base_trans"],
            odir=f"{odir}/trans" if os.path.isdir(odir) else None,
            mask=f"{odir}/files/{it}_msk.npz",
            # fixed noise std (max_std * max) instead of the noise model of the reconstruction: with a per-pixel
            # noise model the fits inflate the noise where they do not fit yet and drop faint structure
            noise=dict(max_std=1e-5, parameters={}),
            plot_dct=base_dct["base_plot"],
            style_dct=base_dct.get("base_style", {}),
            # start the noise model of the new iteration at its prior instead of taking over the learned one
            reset_noise=base_dct.get("base_prefit", {}).get("reset_noise", False),
        )
        os.makedirs(os.path.dirname(tra_pkl), exist_ok=True)
        pickle.dump((samples, offsets), open(tra_pkl, "wb"))

    # the transition fits on its own instance of the sky model: apply its offsets here, too
    for md in sky_mdl.models:
        if md.prefix in offsets:
            md.set_offset(offsets[md.prefix])

    # plot the final results of the transition: sky and components with one shared colorbar
    with plot_style(**theme):
        plot_components(
            [sky_mdl] + list(sky_mdl.models),
            samples,
            name=f"{it}_pre.png",
            odir=f"{odir}/plots",
            step=f"pre-fit · it {it}",
            **sky_dct,
            **pnl_dct,
        )

    # load and update the model config with the new offsets
    cfg = SetupKLConfig.from_file(mdl_yml)

    for pf, of in offsets.items():
        cfg.modify_sec(f"sky_{pf}", offset=of)

    # save the new model yaml-file
    cfg.to_file(f"{odir}/files/{it}_pre.yml")


if __name__ == "__main__":
    main()

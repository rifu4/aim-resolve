"""Script for initializing a reconstruction pipeline from YAML configuration files."""

import os
import sys

from aim_resolve import SetupKLConfig, merge_dicts, radio_data, yaml_load, yaml_save


def main():
    """Initialize a reconstruction run by setting up directories, config files, and model sections."""
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, base_yml, pipe_yml = files

    # load the basic model yaml-file into SetupKLConfig class
    cfg = SetupKLConfig.from_file(mdl_yml)

    # load the pipeline yaml-file and pop not needed keys
    pipe_dct = yaml_load(pipe_yml)
    pipe_dct.pop("n_it")

    # adjust model yaml-file (opt, lh, and data section)
    odir = pipe_dct.pop("odir")
    fun = pipe_dct["data"]["fun"]
    cfg.modify_sec("opt", base="base_opt", odir="->base_opt/odir + opt/0_rec")
    cfg.modify_sec("lh.0", fun=fun)
    cfg.modify_sec("data.0", **pipe_dct.pop("data"))

    # extract frequency channels if specified
    if "freq" in pipe_dct:
        freq = pipe_dct.pop("freq")
        if isinstance(freq, list) and len(freq) > 1:
            cfg.modify_sec("sky_bg.0", freq=freq, params=dict(base="params_mf"))
        elif isinstance(freq, str):
            freq = radio_data(**cfg.sections["data.0"]).freq
            cfg.modify_sec(
                "sky_bg.0", freq=freq.tolist(), params=dict(base="params_mf")
            )
    else:
        freq = [1.0]

    # do split in fast-resolve convolution if specified
    split = pipe_dct.pop("split", 0)

    # add noise scaling configuration for the likelihood
    if "noise" in pipe_dct:
        cfg.modify_sec("lh.0", noise=pipe_dct.pop("noise"))

    # get noise level for likelihood if fun is 'exp'
    if "max_std" in cfg.sections["data.0"]:
        cfg.modify_sec("lh.0", noise=dict(max_std=cfg.sections["data.0"]["max_std"]))

    # get correct kernels if fast-resolve is used
    if "radio" in fun and "fast" in fun:
        kernel_dir = "runs/kernels"
        os.makedirs(kernel_dir, exist_ok=True)
        kname = cfg.sections["data.0"]["fname"].split("/")[-1].split(".")[0]
        ksize = pipe_dct["grid_bg"]["space"][0]
        kfov = pipe_dct["grid_bg"]["fov"][0]
        cfg.modify_sec(
            sec_key="lh.0",
            split=split,
            psf_kernel_fn=f"{kernel_dir}/pk_{kname}_{len(freq)}f_{kfov}_{ksize}.pkl",
            n_inv_kernel_fn=f"{kernel_dir}/nk_{kname}_{len(freq)}f_{kfov}_{ksize}.pkl",
        )

    # extract callback, extra, and transition keys from pipe_dct
    callback = pipe_dct.pop("callback") if "callback" in pipe_dct else False
    extra = pipe_dct.pop("extra") if "extra" in pipe_dct else False
    trans = pipe_dct.pop("trans") if "trans" in pipe_dct else False
    key = pipe_dct.pop("key") if "key" in pipe_dct else 0
    rerun = pipe_dct.pop("rerun") if "rerun" in pipe_dct else True

    # load and overwrite sections of the base yaml-file with pipe_dct sections (like opt, trans, i0, grid,  plot, ...)
    base_dct = yaml_load(base_yml)
    base_dct = merge_dicts(
        [dict(base_opt=dict(odir=odir, key=key, rerun=rerun)), base_dct, pipe_dct],
        merge_base="True",
    )

    # create output directories
    os.makedirs(odir + "/files/", exist_ok=True)
    os.makedirs(odir + "/plots/", exist_ok=True)
    if callback:
        os.makedirs(odir + "/callback/", exist_ok=True)
    if extra:
        os.makedirs(odir + "/extra/", exist_ok=True)
    if trans:
        os.makedirs(odir + "/trans/", exist_ok=True)

    # save the new model yaml-file and base yaml-file
    cfg.to_file(odir + "/files/0_pre.yml")
    yaml_save(base_dct, odir + "/files/base.yml")


if __name__ == "__main__":
    main()

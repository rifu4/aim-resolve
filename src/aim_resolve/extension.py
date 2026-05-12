"""Extension utilities for multi-frequency and zoom workflows."""

from .optimize.set_config import SetupKLConfig
from .optimize.yml import yaml_load


def extension_func(
    mode,
    **kwargs,
):
    """Perform a configuration extension for multi-frequency or zoom workflows.

    Parameters
    ----------
    mode : {'freq', 'zoom'}
        Extension mode.
    **kwargs
        Additional keyword arguments forwarded to the selected extension
        function.

    Returns
    -------
    base : str
        Path to the base configuration file.
    ext_file : str
        Path to the newly created extension configuration file.

    Raises
    ------
    TypeError
        If *mode* is not recognised.
    """
    if mode == "freq":
        return freq_extension(**kwargs)
    elif mode == "zoom":
        return zoom_extension(**kwargs)
    else:
        raise TypeError(
            f"Unknown extension mode. Available modes are `zoom` and `freq`, but got mode `{mode}`."
        )


def freq_extension(
    *,
    odir,
    file,
    freq,
    base="base.yml",
    ref_freq_index=1,
    run=0,
    **kwargs,
):
    """Create a multi-frequency extension configuration.

    Reads an existing single-frequency configuration, duplicates the
    current iteration, updates frequency-dependent sections and writes
    the result to a new file.

    Parameters
    ----------
    odir : str
        Output directory containing the ``files/`` sub-directory.
    file : str
        Name of the existing configuration file inside ``odir/files/``.
    freq : list
        List of target frequencies.
    base : str, optional
        Name of the base configuration file. Default is ``'base.yml'``.
    ref_freq_index : int, optional
        Index of the reference frequency in *freq*. Default is 1.
    run : int, optional
        Run number to append to the output directory name. Default is 0 (no run number).
    **kwargs
        Additional section overrides applied before writing.

    Returns
    -------
    base : str
        Path to the base configuration file.
    ext_file : str
        Path to the newly created extension configuration file.

    Raises
    ------
    TypeError
        If *freq* is not a list.
    """
    base = f"{odir}/files/{base}"
    base_dct = yaml_load(base)
    run = "" if run == 0 else f"_{run}"

    cfg = SetupKLConfig.from_file(f"{odir}/files/{file}")

    if not isinstance(freq, list):
        raise TypeError("The `freq` parameter must be a list of frequencies.")
    else:
        freq = sorted(freq)

    cfg.add_it(fix_keys=("data.0",), del_comp=False)

    cfg.modify_sec(
        "opt",
        resume=cfg.sections["opt"]["odir"],
        odir=cfg.sections["opt"]["odir"] + f"_{len(freq)}f{run}",
    )

    for sec in cfg.sections:
        if "sky_" in sec and f".{cfg.it}" in sec:
            if "p" in sec:
                cfg.modify_sec(
                    sec,
                    freq=freq,
                    params=dict(base="params_ps", ref_freq_index=ref_freq_index),
                )
            else:
                cfg.modify_sec(
                    sec,
                    freq=freq,
                    params=dict(base="params_mf", ref_freq_index=ref_freq_index),
                )

    pk_dir = "_".join(cfg.sections[f"lh.{cfg.it}"]["psf_kernel_fn"].split("_")[:2])
    nk_dir = "_".join(cfg.sections[f"lh.{cfg.it}"]["n_inv_kernel_fn"].split("_")[:2])
    fq_rng = f"{int(min(freq) / 1e6)}mhz-{int(max(freq) / 1e6)}mhz_{len(freq)}f"
    bg_fov = base_dct["grid_bg"]["fov"][0]
    bg_ker = cfg.sections[f"lh.{cfg.it}"]["psf_kernel_fn"].split("_")[-1].split(".")[0]
    lh_dct = {
        "psf_kernel_fn": f"{pk_dir}_{fq_rng}_{bg_fov}_{bg_ker}.pkl",
        "n_inv_kernel_fn": f"{nk_dir}_{fq_rng}_{bg_fov}_{bg_ker}.pkl",
    }
    cfg.modify_sec(f"lh.{cfg.it}", **lh_dct)

    cfg.add_sec(
        f"trans.{cfg.it}",
        lh_old=f"=lh.{cfg.it - 1}",
        lh_new=f"=lh.{cfg.it}",
        mode="freq",
    )
    cfg.modify_sec(f"opt.{cfg.it}", base="base_opt.0", transitions=f"=trans.{cfg.it}")

    for key, val in kwargs.items():
        cfg.modify_sec(key, **val)

    cfg = fun2mode(cfg)

    ext_file = f"{odir}/files/{file.split('.')[0]}_{len(freq)}f{run}.yml"
    cfg.to_file(ext_file)

    return base, ext_file


def zoom_extension(
    *,
    odir,
    file,
    zoom,
    base="base.yml",
    run=0,
    **kwargs,
):
    """Create a zoom extension configuration.

    Reads an existing configuration, duplicates the current iteration,
    increases the grid resolution by *zoom* and writes the result to a
    new file.

    Parameters
    ----------
    odir : str
        Output directory containing the ``files/`` sub-directory.
    file : str
        Name of the existing configuration file inside ``odir/files/``.
    zoom : int
        Zoom factor to apply to non-background grids.
    base : str, optional
        Name of the base configuration file. Default is ``'base.yml'``.
    run : int, optional
        Run number to append to the output directory name. Default is 0 (no run number).
    **kwargs
        Additional section overrides applied before writing.

    Returns
    -------
    base : str
        Path to the base configuration file.
    ext_file : str
        Path to the newly created extension configuration file.
    """
    base = f"{odir}/files/{base}"
    base_dct = yaml_load(base)
    run = "" if run == 0 else f"_{run}"

    cfg = SetupKLConfig.from_file(f"{odir}/files/{file}")

    cfg.add_it(fix_keys=("data.0",), del_comp=False)

    cfg.modify_sec(
        "opt",
        resume=cfg.sections["opt"]["odir"],
        odir=cfg.sections["opt"]["odir"] + f"_{zoom}z{run}",
    )

    for sec in cfg.sections:
        if "sky_" in sec and f".{cfg.it}" in sec and "bg" not in sec:
            grid = cfg.sections[sec]["grid"] | dict(factor=zoom)
            cfg.modify_sec(sec, grid=grid)

    pkdir = "_".join(cfg.sections[f"lh.{cfg.it}"]["psf_kernel_fn"].split("_")[:-1])
    nkdir = "_".join(cfg.sections[f"lh.{cfg.it}"]["n_inv_kernel_fn"].split("_")[:-1])
    ksize = zoom * base_dct["grid_bg"]["space"][0]
    cfg.modify_sec(
        f"lh.{cfg.it}",
        psf_kernel_fn=f"{pkdir}_{ksize}.pkl",
        n_inv_kernel_fn=f"{nkdir}_{ksize}.pkl",
    )

    cfg.add_sec(
        f"trans.{cfg.it}",
        lh_old=f"=lh.{cfg.it - 1}",
        lh_new=f"=lh.{cfg.it}",
        mode="zoom",
        opt_dct=dict(base="base_trans"),
        odir=f"{odir}/opt/{cfg.it - 1}_rec_{zoom}z{run}/trans",
    )
    cfg.modify_sec(f"opt.{cfg.it}", base="base_opt.n", transitions=f"=trans.{cfg.it}")

    for key, val in kwargs.items():
        cfg.modify_sec(key, **val)

    cfg = fun2mode(cfg)

    ext_file = f"{odir}/files/{file.split('.')[0]}_{zoom}z{run}.yml"
    cfg.to_file(ext_file)

    return base, ext_file


def fun2mode(cfg):
    """Replace legacy ``fun`` keys with ``mode`` keys in a configuration.

    Parameters
    ----------
    cfg : SetupKLConfig
        Configuration object whose sections may contain ``fun`` entries.

    Returns
    -------
    cfg : SetupKLConfig
        The updated configuration with ``mode`` keys replacing ``fun``.
    """
    for sec in cfg.sections:
        if "fun" in cfg.sections[sec]:
            fun = cfg.sections[sec].pop("fun")
            if "lh" in sec:
                fun = (
                    "fast" if "fast" in fun else "radio" if "radio" in fun else "image"
                )
            if "data" in sec:
                fun = "radio" if "radio" in fun else "image"
            cfg.sections[sec]["mode"] = fun
    return cfg

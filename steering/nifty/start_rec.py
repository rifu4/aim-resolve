"""CLI script for starting a new reconstruction run with AIM-Resolve."""

import os

import click


@click.command()
@click.option(
    "--base",
    default=os.path.join(os.path.dirname(__file__), "config/base.yml"),
    help="Path to the base YAML config file",
)
@click.option("--config", required=True, help="Path to the YAML config file")
@click.option(
    "--cuda_device",
    default="",
    help='CUDA device to use (e.g. "0", "0,1", ...), default is "" for CPU',
)
@click.option(
    "--plot_range",
    type=float,
    default=None,
    help="Dynamic range of the sky plots (vmin = max / plot_range); default: base_style.dynamic_range or 1e5",
)
@click.option(
    "--gpu_allocator",
    type=click.Choice(["bfc", "platform", "cuda_async"]),
    default=None,
    help="XLA GPU allocator: platform (default; no pool, no fragmentation), bfc (pool, JAX's default) or cuda_async",
)
@click.option(
    "--gpu_preallocate", is_flag=True, help="Reserve the GPU memory pool up front (bfc only)"
)
@click.option(
    "--gpu_mem_fraction",
    type=float,
    default=None,
    help="Fraction of the GPU memory for the pool (bfc only), e.g. 0.8",
)
def main(base, config, cuda_device, plot_range, gpu_allocator, gpu_preallocate, gpu_mem_fraction):
    """Start a reconstruction run from base and config YAML files.

    Parameters
    ----------
    base : str
        Path to the base YAML configuration file.
    config : str
        Path to the run-specific YAML configuration file.
    cuda_device : str
        CUDA device identifier, empty string for CPU.
    plot_range : float or None
        Dynamic range of the sky plots (vmin = max / plot_range). None: the
        ``base_style.dynamic_range`` of the config, else 1e5.
    gpu_allocator : str or None
        XLA GPU allocator (XLA_PYTHON_CLIENT_ALLOCATOR): "platform" (allocated and freed directly, avoids
        out-of-memory errors from pool fragmentation), "bfc" (memory pool, JAX's default) or "cuda_async"
        (CUDA's asynchronous allocator). None: the environment setting if there is one, else "platform"
        ("bfc" if gpu_preallocate or gpu_mem_fraction is given, which only apply to the pool).
    gpu_preallocate : bool
        Reserve the memory pool at the start (XLA_PYTHON_CLIENT_PREALLOCATE); by default it grows on demand.
    gpu_mem_fraction : float or None
        Size (preallocated) or upper limit of the memory pool as a fraction of the GPU memory
        (XLA_PYTHON_CLIENT_MEM_FRACTION).
    """
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        # GPU memory: the options override the environment; default allocator "platform" (no pool, so
        # no out-of-memory errors from fragmentation), the pool ("bfc") if it is configured
        if gpu_allocator is not None:
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = gpu_allocator
        else:
            pool = gpu_preallocate or gpu_mem_fraction is not None
            os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc" if pool else "platform")
        if gpu_preallocate:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if gpu_mem_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_fraction)

    import jax
    import numpy as np

    import aim_resolve as aim

    # Instantiate the optimize-config class
    cfg = aim.OptimizeKLConfig.from_file((base, config), aim.get_builders)
    odir = cfg.sections["opt"]["odir"] + "/plots"

    # Initialize all signal models for each iteration
    sky_dct = {
        sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith("sky.")
    }
    skies = list(sky_dct.values())

    # Plot style (base_style section of the config, if any) and dynamic range of the sky plots:
    # --plot_range, else base_style.dynamic_range, else 1e5
    theme, cfg_range, pnl_dct = aim.split_style(cfg.sections.get("base_style"))
    if plot_range is not None:
        dynamic_range = float(plot_range)
    elif cfg.sections.get("base_style", {}).get("dynamic_range") is not None:
        dynamic_range = cfg_range
    else:
        dynamic_range = 1e5

    # Print and plot the data
    data_dct = {
        sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith("data")
    }
    for dk, dv in data_dct.items():
        print(dv, "\n")
        if isinstance(dv, aim.ImageData):
            d_val = np.asarray(dv.val)
        elif isinstance(dv, aim.Observation):
            d_val = np.asarray(dv.dirty_image(skies[0].grid))
        else:
            continue
        d_val = d_val[d_val.shape[0] // 2] if d_val.ndim == 3 else d_val
        vmin, vmax = aim.dynamic_limits(d_val, dynamic_range)
        with aim.plot_style(**theme):
            aim.plot_panels([dict(arrays=[d_val], vmin=vmin, vmax=vmax)], name=f"{dk}.png", odir=odir, step=dk, **pnl_dct)

    # import nifty.re as jft
    # jft.check_model(sky_models[-1], sky_models[-1].init(jax.random.PRNGKey(4)))

    # Define a callback function to plot the sky and its components after each iteration
    def callback(samples, state, *args):
        nit = args[0] if len(args) > 0 else state.nit
        with aim.plot_style(**theme):
            aim.plot_kl_callback(skies, samples, nit, odir, dynamic_range=dynamic_range, **pnl_dct)

    # Run the optimization
    visible_devices = jax.devices()

    samples, *_ = cfg.optimize_kl(
        callback=callback,
        devices=visible_devices if len(visible_devices) > 1 else None,
    )


if __name__ == "__main__":
    main()

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
    "--plot_range", default=None, help="Brigthness range for plotting the sky models"
)
@click.option(
    "--gpu_allocator",
    type=click.Choice(["bfc", "platform", "cuda_async"]),
    default=None,
    help="XLA GPU allocator: bfc (pool, JAX default), platform (no pool, no fragmentation) or cuda_async",
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
        Brightness range for plotting the sky models.
    gpu_allocator : str or None
        XLA GPU allocator (XLA_PYTHON_CLIENT_ALLOCATOR): "bfc" (memory pool, JAX default), "platform"
        (allocated and freed directly, avoids out-of-memory errors from pool fragmentation) or "cuda_async"
        (CUDA's asynchronous allocator). None keeps the environment setting.
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
        # GPU memory: the options override the environment; by default the pool grows on demand
        if gpu_allocator is not None:
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = gpu_allocator
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
    sky_models = []
    for sky in sky_dct.values():
        sky_models += [
            sky,
        ] + [m for m in sky.models if len(sky.models) > 1]

    # Print and plot the data
    data_dct = {
        sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith("data")
    }
    for dk, dv in data_dct.items():
        print(dv, "\n")
        if isinstance(dv, aim.ImageData):
            aim.plot_arrays(dv.val, name=dk, odir=odir, norm="log")
        elif isinstance(dv, aim.Observation):
            aim.plot_arrays(dv.dirty_image(sky_models[0].grid), name=dk, odir=odir)

    # import nifty.re as jft
    # jft.check_model(sky_models[-1], sky_models[-1].init(jax.random.PRNGKey(4)))

    # Define a callback function to plot the results of the optimization after each iteration
    def callback(samples, state, *args):
        nit = args[0] if len(args) > 0 else state.nit
        for sky in sky_models:
            if aim.domain_keys(sky).issubset(aim.domain_keys(samples)):
                sky_val = samples.mean(sky)
                sky_min = (
                    sky_val.max() / plot_range
                    if plot_range is not None
                    else sky_val[sky_val > 0].min()
                )
                aim.plot_arrays(
                    sky_val,
                    name=f"{nit}_{sky.prefix}",
                    odir=odir,
                    norm="log",
                    rows=1,
                    vmin=sky_min,
                )
                if sky.freq.size > 1:
                    sky_ref = samples.mean(sky.ref_freq_model)
                    aim.plot_arrays(
                        sky_ref,
                        name=f"{nit}_{sky.prefix}_ref",
                        odir=odir,
                        norm="log",
                        rows=1,
                        vmin=sky_min,
                    )
                    if isinstance(sky, aim.ComponentModel) and len(sky.models) > 1:
                        sky = sky.points_and_objects
                    alpha = samples.mean(sky.spectral_index)
                    contours = {
                        "array": sky_ref,
                        "colors": "white",
                        "levels": [sky_val.max() / d for d in [1e3, 1e2, 10]],
                        "linewidths": 0.25,
                    }
                    aim.plot_arrays(
                        np.where(sky_ref > sky_min, alpha, np.nan),
                        name=f"{nit}_{sky.prefix}_alpha",
                        odir=odir,
                        contour=contours,
                    )

    # Run the optimization
    visible_devices = jax.devices()

    samples, *_ = cfg.optimize_kl(
        callback=callback,
        devices=visible_devices if len(visible_devices) > 1 else None,
    )


if __name__ == "__main__":
    main()

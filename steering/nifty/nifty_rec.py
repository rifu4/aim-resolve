import os
import click

@click.command()
@click.option('--base', default='config/base.yml', help='Path to the base YAML config file')
@click.option('--config', required=True, help='Path to the YAML config file')
@click.option('--mode', required=True, help='Mode for NIFTy optimization, "exp", "radio" or "fast-radio"')
@click.option('--cuda_device', default='', help='CUDA device to use (e.g. "0", "0,1", ...), default is "" for CPU')

def main(base, config, mode, cuda_device):
    if str(cuda_device) == '':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    import jax
    import aim_resolve as aim
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    # Instantiate the optimize-config class
    cfg = aim.OptimizeKLConfig.from_file((base, config), aim.get_builders, mode)
    odir = cfg.sections['opt']['odir'] + '/plots'

    # Initialize all signal models for each iteration
    sky_dct = {sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith('sky.')}
    sky_models = []
    for sky in sky_dct.values():
        sky_models += [sky, ] + [m for m in sky.models if len(sky.models) > 1]

    # Print and plot the data
    data_dct = {sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith('data')}
    for dk,dv in data_dct.items():
        print(dv, '\n')
        if isinstance(dv, aim.ImageData):
            aim.plot_arrays(dv.val, name=dk, odir=odir, norm='log')
        elif isinstance(dv, aim.Observation):
            aim.plot_arrays(dv.dirty_image(sky_models[0].grid), name=dk, odir=odir)

    # Define a callback function to plot the results of the optimization after each iteration
    def callback(samples, state, *args):
        nit = args[0] if len(args) > 0 else state.nit
        for sky in sky_models:
            if aim.domain_keys(sky).issubset(aim.domain_keys(samples)):
                sky_val = samples.mean(sky)
                sky_min = sky_val.max()/5e3
                aim.plot_arrays(sky_val, name=f'{nit}_{sky.prefix}', odir=odir, norm='log', rows=1, vmin=sky_min)
                if sky.freq.size > 1:
                     sky_ref = samples.mean(sky.ref_freq_model)
                     aim.plot_arrays(sky_ref, name=f'{nit}_{sky.prefix}_ref', odir=odir, norm='log', rows=1, vmin=sky_min)
                     if isinstance(sky, aim.ComponentModel) and len(sky.models) > 1:
                         sky = sky.points_and_objects
                     alpha = samples.mean(sky.spectral_index)
                     contours = {'array': sky_ref, 'colors': 'white', 'levels': [sky_val.max() / d for d in [1e3, 1e2, 10]], 'linewidths': 0.25}
                     aim.plot_arrays(np.where(sky_ref > sky_min, alpha, np.nan), name=f'{nit}_{sky.prefix}_alpha', odir=odir, contour=contours)

    # Run the optimization
    samples, *_ = cfg.optimize_kl(callback=callback)


if __name__ == "__main__":
    main()

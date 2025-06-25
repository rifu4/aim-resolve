import os
import click

@click.command()
@click.option('--config', required=True, help='Path to the YAML config file')
@click.option('--mode', required=True, help='Mode for NIFTy optimization, "exp", "radio" or "fast-radio"')
@click.option('--cuda_device', default='', help='CUDA device to use (e.g. "0", "0,1", ...), default is "" for CPU')

def main(config, mode, cuda_device):
    if str(cuda_device) == '':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    import jax
    from aim_resolve import OptimizeKLConfig, ImageData, Observation, get_builders, plot_arrays, domain_keys

    jax.config.update("jax_enable_x64", True)

    # Instantiate the optimize-config class
    cfg = OptimizeKLConfig.from_file(('config/base.yml', config), get_builders, mode)
    odir = cfg.sections['opt']['odir'] + '/plots'

    # Initialize all signal models for each iteration
    sig_dct = {sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith('sky')}

    # Print and plot the data
    data_dct = {sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith('data')}
    for dk,dv in data_dct.items():
        print(dv, '\n')
        if isinstance(dv, ImageData):
            plot_arrays(dv.val, name=dk, odir=odir, norm='log')
        elif isinstance(dv, Observation):
            plot_arrays(dv.dirty_image(next(iter(sig_dct.values())).space), name=dk, odir=odir)

    # Define a callback function to plot the results of the optimization after each iteration
    def callback(samples, opt_state, *_):
        for key,sig in sig_dct.items():
            if domain_keys(sig).issubset(domain_keys(samples)):
                plot_arrays(samples.mean(sig), name=f'{opt_state.nit}_{key}', odir=odir, norm='log')

    # Run the optimization
    samples, *_ = cfg.optimize_kl(callback=callback)


if __name__ == "__main__":
    main()

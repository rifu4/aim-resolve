"""CLI script for generating or loading synthetic image data."""
import os
import click

@click.command()
@click.option('--config', required=True, help='Path to the YAML config file')
@click.option('--cuda_device', default='', help='CUDA device to use (e.g. "0", "0,1", ...), default is "" for CPU')

def main(config, cuda_device):
    """Generate or load synthetic image data from a YAML configuration file.

    Parameters
    ----------
    config : str
        Path to the YAML configuration file.
    cuda_device : str
        CUDA device identifier, empty string for CPU.
    """
    if str(cuda_device) == '':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    import jax
    from jax import random
    from aim_resolve import ImageDataGenerator, yaml_load

    jax.config.update("jax_enable_x64", True)

    # Load config
    dct = yaml_load(config)

    # Generate or load data
    if 'parameters' in dct:
        print('\ngenerate data:')
        data = ImageDataGenerator.build(parameters=dct['parameters'])

        key = random.PRNGKey(dct['seed'])
        key, subkey = random.split(key)

        data.draw_samples(subkey, dct['n_copies'], dct['batch_size'])
        data.save(dct['name'], dct['odir'], dct['dtype'])

    else:
        print('\nload data:\nfname:', dct['name'])
        data = ImageDataGenerator.load(dct['name'], dct['odir'])

    # Optionally plot
    if 'plot' in dct:
        print('\nplot data:')
        data.plot_samples(dct['name'], **dct['plot'])
        print('done')


if __name__ == "__main__":
    main()

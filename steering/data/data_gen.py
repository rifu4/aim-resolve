import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import sys
from jax import random
from aim_resolve import ImageDataGenerator, yaml_load

print(jax.devices())
jax.config.update("jax_enable_x64", True)



_, yfile = sys.argv[0], sys.argv[1]

dct = yaml_load(yfile)


# generate and save data or load existing data
if 'parameters' in dct:
    print('\ngenerate data:')
    data = ImageDataGenerator.build(parameters=dct['parameters'])

    key = random.PRNGKey(dct['seed'])
    key, subkey = random.split(key)

    data.draw_samples(subkey, dct['n_copies'])

    data.save(dct['name'], dct['odir'], dct['dtype'])

else:
    print('\nload data:\nfname:', dct['name'])
    data = ImageDataGenerator.load(dct['name'], dct['odir'])
    

# plot data
if 'plot' in dct:
    print('\nplot data:')
    data.plot_samples(dct['name'], dct['odir'], **dct['plot'])
    print('done')

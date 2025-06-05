import os
import pickle
import jax.numpy as jnp
import nifty8.re as jft
import numpy as np
from jax import random, lax
from jax.typing import ArrayLike
from jax_tqdm import loop_tqdm

from .components import ComponentGenerator
from ..model.space import SignalSpace
from ..model.util import check_type



class ImageDataGenerator():
    '''Generate a image data model. Use `build` function to create the model.'''

    def __init__(self, model, parameters, samples=None):
        check_type(model, ComponentGenerator)
        check_type(parameters, dict)
        check_type(samples, (np.ndarray, type(None)))

        self.model = model
        self.parameters = parameters
        self.samples = samples

    @property
    def x(self):
        return self.samples[:, 0, jnp.newaxis, :, :]
    
    @property
    def y(self):
        return self.samples[:, 1:, :, :]

    @classmethod
    def build(cls, *, parameters, samples=None):
        '''
        Build a image data generator model.

        Parameters
        ----------
        parameters : dict
            Dictionary containing the model parameters (see ComponentGenerator)
        samples : np.ndarray, optional
            Array containing the samples, by default None
        '''
        check_type(parameters, dict)
        check_type(samples, (np.ndarray, type(None)))

        model = ComponentGenerator.build(**parameters)

        return cls(model, parameters, samples)

    def draw_samples(self, key, n_copies=1):
        '''
        Draw samples from the model.

        Parameters
        ----------
        key : int or jax.random.PRNGKey
            Random key for sampling. If an int is provided, it will be used as a seed.
        n_copies : int, optional
            Number of samples to draw, by default 1
        '''
        key = random.PRNGKey(key) if isinstance(key, int) else key
        samples = jnp.empty((n_copies,) + self.model.target.shape)

        @loop_tqdm(n_copies)
        def step(i, tpl):
            smp, key = tpl
            key, subkey = random.split(key)
            xi = jft.random_like(subkey, self.model.domain)
            smp = smp.at[i].set(self.model(xi, key=subkey))
            return (smp, key)

        samples, key = lax.fori_loop(0, n_copies, step, (samples, key))
        self.samples = np.array(samples)

    def get_sample(self, index=0, prefix='data'):
        '''
        Get a sample from the model. Returns an ImageData object.
        
        Parameters
        ----------
        index : int, optional
            Index of the sample to get, by default 0
        prefix : str, optional
            Prefix for the sample, by default 'data'
        '''
        return ImageData(self.x[index, 0], self.model.space, prefix)
    
    def plot_samples(self, name, odir='', n_copies=10, space=False, label=False, **kwargs):
        '''
        Plot a number of samples.

        Parameters
        ----------
        name : str
            Name of the plot
        odir : str, optional
            Output directory for the plot, by default ''
        n_copies : int, optional
            Number of samples to plot, by default 10
        space : bool, optional
            Whether to plot the space of the model, by default False
        label : bool, optional
            Whether to add labels ['points', 'objects', 'sky'] to the plot, by default False
        **kwargs : additional keyword arguments
            Additional keyword arguments to pass to the plotting function
        '''
        from ..plot.arrays import plot_arrays

        if not isinstance(self.samples, np.ndarray):
            raise ValueError('no samples to plot - please draw samples first')

        rows = min(n_copies, self.samples.shape[0])
        vals = self.samples[:rows]

        if odir:
            if not odir.endswith(('plots', 'plots/')):
                odir = os.path.join(odir, 'plots')
            os.makedirs(odir, exist_ok=True)
        
        plot_arrays(
            array = vals,
            space = self.model.space if space else None,
            label = ['sky', 'points', 'objects'] if label else None,
            rows = rows,
            cols = 3,
            name = name,
            odir = odir,
            **kwargs,
        )

    def save(self, name, odir='', dtype='float64'):
        '''
        Save the model to a file.
        
        Parameters
        ----------
        name : str
            Name of the file to save the model to
        odir : str, optional
            Output directory for the file, by default ''
        dtype : str, optional
            Data type to save the model as, by default 'float64'
        '''
        if not name.endswith('.pkl'):
            name += '.pkl'
        if not odir.endswith(('files', 'files/')):
            odir = os.path.join(odir, 'files')
        os.makedirs(odir, exist_ok=True)

        with open(os.path.join(odir, name), 'wb') as f:
            pickle.dump((self.parameters, self.samples.astype(dtype)), f)

    @classmethod
    def load(cls, name, odir='', dtype='float64'):
        '''
        Load a model from a file.

        Parameters
        ----------
        name : str
            Name of the file to load the model from
        odir : str, optional
            Output directory for the file, by default ''
        dtype : str, optional
            Data type to load the model as, by default 'float64'
        '''
        if not name.endswith('.pkl'):
            name += '.pkl'
        if not odir.endswith(('files', 'files/')):
            odir = os.path.join(odir, 'files')
        with open(os.path.join(odir, name), 'rb') as file:
            parameters, samples = pickle.load(file)
        
        return cls.build(parameters=parameters, samples=samples.astype(dtype))



class ImageData():
    def __init__(self, val, space, prefix='data'):
        '''
        Store an image data object and its properties for nifty reconstructions.

        Parameters
        ----------
        val : ArrayLike
            array containing the image data
        space : SignalSpace
            space of the image data
        prefix : str
            Prefix for the image data
        '''
        check_type(val, ArrayLike)
        check_type(space, SignalSpace)
        check_type(prefix, str)

        self.val = np.array(val)
        self.space = space
        self.prefix = prefix
        self.noisy_val = None

    def __repr__(self) -> str:
        s = [
            f'prefix:\t{self.prefix}',
            f'image shape:\t{self.val.shape}',
            f'# pixel:\t{self.val.size}',
            f'space fov:\t{tuple(self.space.fov)}',
        ]
        return '\n'.join(['ImageData:'] + [f'  {ss}' for ss in s])
    
    def add_noise(self, key, max_std=0.001):
        '''
        Add noise to the image data.
        
        Parameters
        ----------
        key : int or jax.random.PRNGKey
            Random key for generating noise. If an int is provided, it will be used as a seed.
        max_std : float, optional
            Maximum standard deviation of the noise multiplied with the data maximum, by default 0.001
        '''
        key = random.PRNGKey(key) if isinstance(key, int) else key
        n_std = max_std * np.max(self.val)
        noise = n_std * random.normal(key, self.space.shape)
        self.noisy_val = self.val + noise

    def save(self, name, odir='', dtype='float64'):
        '''
        Save the image data to a file.
        
        Parameters
        ----------
        name : str
            Name of the file to save the image data to
        odir : str, optional
            Output directory for the file, by default ''
        dtype : str, optional
            Data type to save the image data as, by default 'float64'
        '''
        if not name.endswith('.pkl'):
            name += '.pkl'
        if odir:
            os.makedirs(odir, exist_ok=True)

        with open(os.path.join(odir, name), 'wb') as f:
            pickle.dump((self.val.astype(dtype), self.space, self.prefix), f)

    @classmethod
    def load(cls, name, odir='', dtype='float64'):
        '''
        Load the image data from a file.
        
        Parameters
        ----------
        name : str
            Name of the file to load the image data from
        odir : str, optional
            Output directory for the file, by default ''
        dtype : str, optional
            Data type to load the image data as, by default 'float64'
        '''
        if not name.endswith('.pkl'):
            name += '.pkl'
        with open(os.path.join(odir, name), 'rb') as file:
            val, space, prefix = pickle.load(file)

        return cls(val.astype(dtype), space, prefix)

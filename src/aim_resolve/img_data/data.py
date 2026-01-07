import os
import pickle
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from jax import random, lax
from jax.typing import ArrayLike
from jax_tqdm import loop_tqdm

from .components import ComponentGenerator
from ..model.grid import SignalGrid
from ..model.util import check_type



class ImageDataGenerator():
    '''Generate a image data model. Use `build` function to create the model.'''

    def __init__(self, model, parameters, samples=None):
        check_type(model, ComponentGenerator)
        check_type(parameters, dict)
        check_type(samples, (np.ndarray, type(None)))

        self.model = model
        self.grid = model.grid
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

    def draw_samples(self, key, n_copies=1, batch_size=10000):
        '''
        Draw samples from the model.

        Parameters
        ----------
        key : int or jax.random.PRNGKey
            Random key for sampling. If an int is provided, it will be used as a seed.
        n_copies : int, optional
            Number of samples to draw, by default 1
        batch_size : int, optional
            Size of the batches to use for sampling, by default 10000.
        '''
        key = random.PRNGKey(key) if isinstance(key, int) else key
        samples = np.empty((n_copies,) + self.model.target.shape)

        n_batches = (n_copies + batch_size - 1) // batch_size

        for batch_i in range(n_batches):
            start = batch_i * batch_size
            end = min(start + batch_size, n_copies)
            n_i = end - start

            samples_i = jnp.empty((n_i,) + self.model.target.shape)

            print(f'Step {batch_i + 1}/{n_batches}: ', end='', flush=True)

            @loop_tqdm(n_i)
            def step(i, tpl):
                smp, k = tpl
                k, sk = random.split(k)
                xi = jft.random_like(sk, self.model.domain)
                smp = smp.at[i].set(self.model(xi, key=sk))
                return (smp, k)

            samples_i, key = lax.fori_loop(0, n_i, step, (samples_i, key))

            samples[start:end] = np.array(samples_i)

        self.samples = samples

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
        return ImageData(self.x[index, 0], self.grid, prefix, self.y[index])
    
    def plot_samples(self, name, odir='', n_copies=10, grid=False, label=False, **kwargs):
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
        grid : bool, optional
            Whether to plot the grid of the model, by default False
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

        [kwargs.pop(key, None) for key in ('rows', 'cols')]
        
        plot_arrays(
            array = vals,
            grid = self.grid if grid else None,
            label = ['sky', 'points', 'objects'] if label else None,
            rows = rows,
            cols = 3,
            name = name,
            odir = odir,
            **kwargs,
        )

    def get_subset(self, size):
        '''
        Get a subset of the samples.

        Parameters
        ----------
        size : int
            Size of the subset to get
        '''
        if not isinstance(self.samples, np.ndarray):
            raise ValueError('no samples to get - please draw samples first')
        
        size = min(size, self.samples.shape[0])
        samples = self.samples[:size]

        return ImageDataGenerator(self.model, self.parameters, samples)

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
        with open(os.path.join(odir, name), 'rb') as file:
            parameters, samples = pickle.load(file)
        
        return cls.build(parameters=parameters, samples=samples.astype(dtype))



class ImageData():
    def __init__(self, val, grid, prefix='data', maps=None):
        '''
        Store an image data object and its properties for nifty reconstructions.

        Parameters
        ----------
        val : ArrayLike
            array containing the image data
        grid : SignalGrid
            grid of the image data
        prefix : str
            Prefix for the image data
        maps : ArrayLike
            array containing the output maps for the image data
        '''
        check_type(val, ArrayLike)
        check_type(grid, SignalGrid)
        check_type(prefix, str)
        check_type(maps, (ArrayLike, type(None)))

        self.val = np.array(val)
        self.grid = grid
        self.prefix = prefix
        self.maps = np.array(maps) if isinstance(maps, ArrayLike) else np.zeros_like(self.val)
        self.noisy_val = None

    def __repr__(self) -> str:
        s = [
            f'prefix:\t{self.prefix}',
            f'image shape:\t{self.val.shape}',
            f'# pixel:\t{self.val.size}',
            f'grid fov:\t{tuple(self.grid.fov)}',
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
        noise = n_std * random.normal(key, self.grid.shape)
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
            pickle.dump((self.val.astype(dtype), self.grid, self.prefix, self.maps.astype(dtype)), f)

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
            val, grid, prefix, maps = pickle.load(file)

        return cls(val.astype(dtype), grid, prefix, maps.astype(dtype))

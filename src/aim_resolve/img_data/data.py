"""Image data containers and generators for synthetic sky images."""

import os
import pickle

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from jax import lax, random
from jax.typing import ArrayLike
from jax_tqdm import loop_tqdm

from ..model.grid import SignalGrid
from ..model.util import check_type
from .components import ComponentGenerator


class ImageDataGenerator:
    """Generator for synthetic image data batches.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    model : ComponentGenerator
        Generative component model.
    parameters : dict
        Configuration dictionary used to build *model*.
    samples : np.ndarray or None
        Pre-drawn sample array of shape ``(n, *model.target.shape)``.
    """

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
        """Build an image data generator from configuration.

        Parameters
        ----------
        parameters : dict
            Model configuration forwarded to ``ComponentGenerator.build``.
        samples : np.ndarray or None, optional
            Pre-existing samples. Default is None.

        Returns
        -------
        ImageDataGenerator
            The constructed generator.
        """
        check_type(parameters, dict)
        check_type(samples, (np.ndarray, type(None)))

        model = ComponentGenerator.build(**parameters)

        return cls(model, parameters, samples)

    def draw_samples(self, key, n_copies=1, batch_size=10000):
        """Draw random samples from the generative model.

        Parameters
        ----------
        key : int or jax.random.PRNGKey
            Random seed or key.
        n_copies : int, optional
            Number of samples to draw. Default is 1.
        batch_size : int, optional
            Samples per batch (controls memory usage). Default is 10000.
        """
        key = random.PRNGKey(key) if isinstance(key, int) else key
        samples = np.empty((n_copies,) + self.model.target.shape)

        n_batches = (n_copies + batch_size - 1) // batch_size

        for batch_i in range(n_batches):
            start = batch_i * batch_size
            end = min(start + batch_size, n_copies)
            n_i = end - start

            samples_i = jnp.empty((n_i,) + self.model.target.shape)

            print(f"Step {batch_i + 1}/{n_batches}: ", end="", flush=True)

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

    def get_sample(self, index=0, prefix="data"):
        """Return a single sample as an ``ImageData`` object.

        Parameters
        ----------
        index : int, optional
            Sample index. Default is 0.
        prefix : str, optional
            Name prefix for the ``ImageData`` instance. Default is
            ``'data'``.

        Returns
        -------
        ImageData
            The selected sample.
        """
        return ImageData(self.x[index, 0], self.grid, prefix, self.y[index])

    def plot_samples(
        self, name, odir="", n_copies=10, grid=False, label=False, **kwargs
    ):
        """Plot a selection of drawn samples.

        Parameters
        ----------
        name : str
            Output file name for the plot.
        odir : str, optional
            Output directory. Default is ``''``.
        n_copies : int, optional
            Number of samples to plot. Default is 10.
        grid : bool, optional
            Whether to annotate with the model grid. Default is False.
        label : bool, optional
            Whether to add component labels. Default is False.
        **kwargs
            Additional keyword arguments forwarded to ``plot_arrays``.

        Raises
        ------
        ValueError
            If no samples have been drawn yet.
        """
        from ..plot.arrays import plot_arrays

        if not isinstance(self.samples, np.ndarray):
            raise ValueError("no samples to plot - please draw samples first")

        rows = min(n_copies, self.samples.shape[0])
        vals = self.samples[:rows]

        if odir:
            if not odir.endswith(("plots", "plots/")):
                odir = os.path.join(odir, "plots")
            os.makedirs(odir, exist_ok=True)

        [kwargs.pop(key, None) for key in ("rows", "cols")]

        plot_arrays(
            array=vals,
            grid=self.grid if grid else None,
            label=["sky", "points", "objects"] if label else None,
            rows=rows,
            cols=3,
            name=name,
            odir=odir,
            **kwargs,
        )

    def get_subset(self, size):
        """Return a new generator containing a subset of drawn samples.

        Parameters
        ----------
        size : int
            Maximum number of samples in the subset.

        Returns
        -------
        ImageDataGenerator
            Generator with the first *size* samples.

        Raises
        ------
        ValueError
            If no samples have been drawn yet.
        """
        if not isinstance(self.samples, np.ndarray):
            raise ValueError("no samples to get - please draw samples first")

        size = min(size, self.samples.shape[0])
        samples = self.samples[:size]

        return ImageDataGenerator(self.model, self.parameters, samples)

    def save(self, name, odir="", dtype="float64"):
        """Save the generator (parameters and samples) to a pickle file.

        Parameters
        ----------
        name : str
            File name (`.pkl` extension added automatically).
        odir : str, optional
            Output directory. Default is ``''``.
        dtype : str, optional
            Numeric type for the saved samples. Default is ``'float64'``.
        """
        if not name.endswith(".pkl"):
            name += ".pkl"
        os.makedirs(odir, exist_ok=True)

        with open(os.path.join(odir, name), "wb") as f:
            pickle.dump((self.parameters, self.samples.astype(dtype)), f)

    @classmethod
    def load(cls, name, odir="", dtype="float64"):
        """Load a generator from a pickle file.

        Parameters
        ----------
        name : str
            File name (`.pkl` extension added automatically).
        odir : str, optional
            Input directory. Default is ``''``.
        dtype : str, optional
            Numeric type for the loaded samples. Default is ``'float64'``.

        Returns
        -------
        ImageDataGenerator
            The restored generator.
        """
        if not name.endswith(".pkl"):
            name += ".pkl"
        with open(os.path.join(odir, name), "rb") as file:
            parameters, samples = pickle.load(file)

        return cls.build(parameters=parameters, samples=samples.astype(dtype))


class ImageData:
    """Container for a single image observation used in reconstruction.

    Parameters
    ----------
    val : array_like
        Clean image data array.
    grid : SignalGrid
        Spatial grid of the image.
    prefix : str, optional
        Name prefix. Default is ``'data'``.
    maps : array_like or None, optional
        Segmentation / output maps for the image. Default is None.
    """

    def __init__(self, val, grid, prefix="data", maps=None):
        check_type(val, ArrayLike)
        check_type(grid, SignalGrid)
        check_type(prefix, str)
        check_type(maps, (ArrayLike, type(None)))

        self.val = np.array(val)
        self.grid = grid
        self.prefix = prefix
        self.maps = (
            np.array(maps) if isinstance(maps, ArrayLike) else np.zeros_like(self.val)
        )
        self.noisy_val = None

    def __repr__(self) -> str:
        s = [
            f"prefix:\t{self.prefix}",
            f"image shape:\t{self.val.shape}",
            f"# pixel:\t{self.val.size}",
            f"grid fov:\t{tuple(self.grid.fov.tolist())}",
        ]
        return "\n".join(["ImageData:"] + [f"  {ss}" for ss in s])

    def add_noise(self, key, max_std=0.001):
        """Add Gaussian noise to the image and store as ``noisy_val``.

        Parameters
        ----------
        key : int or jax.random.PRNGKey
            Random seed or key.
        max_std : float, optional
            Noise standard deviation relative to the data maximum.
            Default is 0.001.
        """
        key = random.PRNGKey(key) if isinstance(key, int) else key
        n_std = max_std * np.max(self.val)
        noise = n_std * random.normal(key, self.grid.shape)
        self.noisy_val = self.val + noise

    def save(self, name, odir="", dtype="float64"):
        """Save the image data to a pickle file.

        Parameters
        ----------
        name : str
            File name (`.pkl` extension added automatically).
        odir : str, optional
            Output directory. Default is ``''``.
        dtype : str, optional
            Numeric type for the saved arrays. Default is ``'float64'``.
        """
        if not name.endswith(".pkl"):
            name += ".pkl"
        if odir:
            os.makedirs(odir, exist_ok=True)

        with open(os.path.join(odir, name), "wb") as f:
            pickle.dump(
                (
                    self.val.astype(dtype),
                    self.grid,
                    self.prefix,
                    self.maps.astype(dtype),
                ),
                f,
            )

    @classmethod
    def load(cls, name, odir="", dtype="float64"):
        """Load image data from a pickle file.

        Parameters
        ----------
        name : str
            File name (`.pkl` extension added automatically).
        odir : str, optional
            Input directory. Default is ``''``.
        dtype : str, optional
            Numeric type for the loaded arrays. Default is ``'float64'``.

        Returns
        -------
        ImageData
            The restored image data.
        """
        if not name.endswith(".pkl"):
            name += ".pkl"
        with open(os.path.join(odir, name), "rb") as file:
            val, grid, prefix, maps = pickle.load(file)

        return cls(val.astype(dtype), grid, prefix, maps.astype(dtype))

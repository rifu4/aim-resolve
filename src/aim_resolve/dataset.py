import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from torch.utils.data import DataLoader, Dataset

from .img_data.data import ImageDataGenerator
from .plot.arrays import plot_arrays



class Dataset():
    '''Create datasets from the given image data. See `build` function to create the dataset.'''

    def __init__(self, data_train, data_valid):
        self.train = data_train
        self.valid = data_valid

    def train_loader(self, **kwargs):
        return DataLoader(self.train, **kwargs)
    
    def valid_loader(self, **kwargs):
        return DataLoader(self.valid, **kwargs)

    def plot(self):
        plot_arrays(self.train[0][:10], cols=3, transpose=True)
        plot_arrays(self.train[1][:10], cols=2, transpose=True)

    @classmethod
    def build(cls, img_data, transform, coordinates=True, size=1000, split=0.8):
        '''
        Build train and validation datasets from generated ImageData.
        
        Parameters
        ----------
        img_data : dict
            Dictionary containing image data parameters (see ImageDataGenerator.load)
        transform : dict
            Dictionary containing transformation parameters (see transform_data)
        coordinates : bool, optional
            Whether to add coordinates to the data, by default True
        size : int, optional
            Number of samples to use from the dataset, by default 1000
        split : float, optional
            Fraction of data to use for training, by default 0.8
        '''
        img_data = ImageDataGenerator.load(**img_data)

        data = img_data.x[:size], img_data.y[:size]

        data = transform_data(data, **transform)

        if coordinates:
            data = add_coordinates(data, img_data.model.space.coos)

        data_train, data_valid = split_data(data, split)

        data_train = TensorDataset(data_train)
        data_valid = TensorDataset(data_valid)

        return cls(data_train, data_valid)



class TensorDataset(Dataset):
    def __init__(self, data):
        x, y = data
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return {'x': x, 'y':y}

    def __len__(self):
        return self.x.shape[0]



def transform_data(
        data,
        log = True,
        normalize = True,
        standardize = False,
        rotate = True,
        flip = True,
):
    if normalize and standardize:
        raise ValueError('normalize and standardize cannot both be True')

    xs, ys = data
    if log:
        xs = np.log(xs)
    if normalize:
        xs = jax.vmap(lambda x: (x-x.min())/(x.max()-x.min()))(xs)
    if standardize:
        xs = jax.vmap(lambda x: (x-x.mean())/x.std())(xs)
    if rotate:
        ks = np.random.randint(0, 3, size=xs.shape[0])
        xs = jax.vmap(lambda x, k: rotate_data(x, k, axes=(1, 2)))(xs, ks)
        ys = jax.vmap(lambda y, k: rotate_data(y, k, axes=(1, 2)))(ys, ks)
    if flip:
        axs = np.random.randint(0, 3, size=xs.shape[0])
        xs = jax.vmap(lambda x, a: flip_data(x, a))(xs, axs)
        ys = jax.vmap(lambda y, a: flip_data(y, a))(ys, axs)

    return (np.array(xs), np.array(ys))



def rotate_data(
        m : ArrayLike,
        k : int = 1,
        axes: tuple[int, int] = (0, 1),
):
    k = k % 4
    return jax.lax.switch(
        k,
        [lambda: m,
         lambda: jnp.rot90(m, k=1, axes=axes),
         lambda: jnp.rot90(m, k=2, axes=axes),
         lambda: jnp.rot90(m, k=3, axes=axes),]
    )



def flip_data(
        m : ArrayLike,
        axis: int = 0,
):
    axis = axis % 3
    return jax.lax.switch(
        axis,
        [lambda: m,
         lambda: jnp.flip(m, axis=1),
         lambda: jnp.flip(m, axis=2)],
    )



def add_coordinates(
        data,
        coordinates,
):
    xs, ys = data

    coordinates = np.concatenate([c[None] for c in coordinates], axis=0)
    coordinates = np.repeat(coordinates[None], xs.shape[0], axis=0)

    xs = np.concatenate((xs, coordinates), axis=1)

    return (xs, ys)



def split_data(
        data,
        split = 0.8,
):
    dataset_size = data[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in data)

    indices = np.arange(dataset_size)
    perm = np.random.permutation(indices)

    split_idx = int(split * dataset_size)
    train_idx = perm[:split_idx]
    valid_idx = perm[split_idx:]

    train_data = tuple(array[train_idx] for array in data)
    valid_data = tuple(array[valid_idx] for array in data)

    return train_data, valid_data

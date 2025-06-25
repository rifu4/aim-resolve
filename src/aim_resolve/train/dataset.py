import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from torch.utils.data import DataLoader, Dataset

from ..img_data.data import ImageDataGenerator
from ..model.util import check_type



class Dataset():
    '''Create datasets from the given image data. See `build` function to create the dataset.'''

    def __init__(self, train, valid = None):
        check_type(train, TensorDataset)
        check_type(valid, (dict, type(None), TensorDataset))

        self.train = train
        self.valid = valid

    def train_loader(self, **kwargs):
        train_loader = DataLoader(self.train, **kwargs)
        return train_loader

    def valid_loader(self, **kwargs):
        valid_loader = {}
        for k, v in self.valid.items():
            valid_loader[k] = DataLoader(v, **kwargs)
        return valid_loader

    @classmethod
    def build(cls, train, valid, transform, coordinates=True):
        '''
        Build train and validation datasets from generated ImageData.
        
        Parameters
        ----------
        train : dict
            Dictionary containing training data (see ImageDataGenerator.load)
        valid : dict of dicts
            Dictionary of dictionaries containing validation data (see ImageDataGenerator.load)
        transform : dict
            Dictionary containing transformation parameters (see transform_data)
        coordinates : bool, optional
            Whether to add coordinates to the data, by default True
        '''
        n_train = train.pop('size', False)
        image_data_train = ImageDataGenerator.load(**train)
        if n_train:
            image_data_train = image_data_train.get_subset(n_train)

        data_train = (image_data_train.x, image_data_train.y)
        data_train = transform_data(data_train, **transform)

        if coordinates:
            data_train = add_coordinates(data_train, image_data_train.model.space.coos)

        data_train = TensorDataset(data_train)

        data_valid = {}
        for k,v in valid.items():
            n_v = v.pop('size', False)
            image_data_v = ImageDataGenerator.load(**v)
            if n_v:
                image_data_v = image_data_v.get_subset(n_v)

            data_v = (image_data_v.x, image_data_v.y)
            data_v = transform_data(data_v, **transform)
            
            if coordinates:
                data_v = add_coordinates(data_v, image_data_v.model.space.coos)

            data_v = TensorDataset(data_v)
            data_valid[k] = data_v

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
        min_value = 0,
        log = True,
        normalize = True,
        standardize = False,
        rotate = True,
        flip = True,
        batch_size = 1000,
):  
    '''
    Apply various transformations to the data.

    Parameters
    ----------
    data : tuple of (images, labels)
        The input data containing images and labels.
    min_value : float, optional
        Sets the minimum value of the images, by default 0.
    log : bool, optional
        Whether to take the logarithm of the images, by default True.
    normalize : bool, optional
        Whether to normalize the images, by default True.
    standardize : bool, optional
        Whether to standardize the images, by default False.
    rotate : bool, optional
        Whether to apply random rotation to the images and labels, by default True.
    flip : bool, optional
        Whether to apply random flipping to the images and labels, by default True.
    batch_size : int, optional
        The size of the batches to process the data, by default 1000.
    '''
    if normalize and standardize:
        raise ValueError('normalize and standardize cannot both be True')
    
    images, labels = data
    n_copies = images.shape[0]
    n_batches = (n_copies + batch_size - 1) // batch_size

    for batch_i in range(n_batches):
        start = batch_i * batch_size
        end = min(start + batch_size, n_copies)

        img_i = jnp.array(images[start:end])
        lbl_i = jnp.array(labels[start:end])

        if min_value <= 0:
            min_value = np.min(img_i[img_i > 0])
        img_i = np.where(img_i > min_value, img_i, min_value)
        if log:
            img_i = np.log(img_i)
        if normalize:
            img_i = jax.vmap(lambda x: (x-x.min())/(x.max()-x.min()))(img_i)
        if standardize:
            img_i = jax.vmap(lambda x: (x-x.mean())/x.std())(img_i)
        if rotate:
            ks = np.random.randint(0, 3, size=img_i.shape[0])
            img_i = jax.vmap(lambda x, k: rotate_data(x, k, axes=(1, 2)))(img_i, ks)
            lbl_i = jax.vmap(lambda y, k: rotate_data(y, k, axes=(1, 2)))(lbl_i, ks)
        if flip:
            axs = np.random.randint(0, 3, size=img_i.shape[0])
            img_i = jax.vmap(lambda x, a: flip_data(x, a))(img_i, axs)
            lbl_i = jax.vmap(lambda y, a: flip_data(y, a))(lbl_i, axs)

        images[start:end] = np.array(img_i)
        labels[start:end] = np.array(lbl_i)

    return (images, labels)



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
    images, labels = data

    coordinates = np.concatenate([c[None] for c in coordinates], axis=0)
    coordinates = np.repeat(coordinates[None], images.shape[0], axis=0)

    images = np.concatenate((images, coordinates), axis=1)

    return (images, labels)



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

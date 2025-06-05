import jax.numpy as jnp
import numpy as np
from jax import vmap
from nifty8.re import Model

from .util import check_type, is_val, to_shape



class SignalSpace():
    '''Class to represent a signal space at a specific location in the sky. Use `build` function to create the space.'''

    def __init__(self, shape, distances, center=None, rotation=None, n_copies=1):
        check_type(shape, tuple, int)
        check_type(distances, tuple, float)
        check_type(center, tuple, (tuple, float), float)
        check_type(rotation, (tuple, float), float)
        check_type(n_copies, int)

        self.shape = shape
        self.distances = distances
        self.center = center
        self.rotation = rotation
        self.n_copies = n_copies

    def __repr__(self):
        return f'SignalSpace(shape={self.shape}, distances={self.distances}, center={self.center}, rotation={self.rotation})'
    
    def __eq__(self, other):
        return isinstance(other, SignalSpace) and self.shape == other.shape and np.all(self.coos == other.coos)

    def __mul__(self, other):
        return self.multiply_shape(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @classmethod
    def build(cls, *, shape, distances=None, fov=None, center=None, rotation=None, n_copies=1):
        '''
        Build a SignalSpace from the given parameters.
        
        Parameters
        ----------
        shape : int or tuple
            The shape of the space
        distances : float or tuple, optional
            The distance between the pixels, by default None
        fov : float or tuple, optional
            The field of view of the space, by default None
        center : float or tuple, optional
            The center of the space, by default None
        rotation : float, optional
            The rotation of the space, by default None
        n_copies : int, optional
            The number of copies of the space, by default 1
        '''
        shp = to_shape(shape, (2,), 'int64')
        dis = to_shape(distances, (2,), 'float64')
        fov = to_shape(fov, (2,), 'float64')
        cen = to_shape(center, (n_copies, 2), 'float64')
        rot = to_shape(rotation, (n_copies,), 'float64')

        shape = tuple(shp.tolist())

        if is_val(dis):
            distances = tuple(dis.tolist())
        elif is_val(fov):
            distances = tuple((fov / shp).tolist())
        else:
            distances = tuple((1 / shp).tolist())

        if not is_val(cen):
            cen = np.zeros_like(cen)
        center = tuple(map(tuple, cen.tolist()))

        if not is_val(rot):
            rot = np.zeros_like(rot)
        rotation = tuple((rot % (2*np.pi)).tolist())

        if n_copies == 1:
            center = center[0]
            rotation = rotation[0]

        return cls(shape, distances, center, rotation, n_copies)

    @property
    def shp(self):
        return np.array(self.shape)
    
    @property
    def dis(self):
        return np.array(self.distances)

    @property
    def fov(self):
        return self.shp * self.dis
    
    @property
    def cen(self):
        return np.array(self.center)
    
    @property
    def rot(self):
        return np.array(self.rotation)

    @property
    def coos(self):
        if self.n_copies == 1:
            return space_coos(self.shp, self.dis, self.cen, self.rot)
        else:
            return vmap(space_coos, in_axes=(None, None, 0, 0))(self.shp, self.dis, self.cen, self.rot)

    @property
    def lims(self):
        if self.n_copies == 1:
            return space_lims(self.fov, self.cen)
        else:
            return vmap(space_lims, in_axes=(None, 0))(self.fov, self.cen)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def coordinates(self):
        return self.coos

    @property
    def limits(self):
        return self.lims

    def multiply_shape(self, factor):
        '''Multiply the shape of the space by a factor and keep the field of view.'''
        check_type(factor, (int, float))
        shape = tuple(int(round(si * factor)) for si in self.shape)
        distances = tuple(fi / si for fi,si in zip(self.fov, shape))
        return SignalSpace(shape, distances, self.center, self.rotation)

    def multiply_fov(self, factor):
        '''Multiply the field of view and the shape of the space by a factor.'''
        check_type(factor, (int, float))
        shape = tuple(int(round(si * factor)) for si in self.shape)
        return SignalSpace(shape, self.distances, self.center, self.rotation)

    def to_dict(self, mode='fov'):
        '''Convert the space to a dictionary ({shape: [sx,sy], ...}).'''
        dct = {'shape': self.shp.tolist()}
        if mode == 'fov':
            dct['fov'] = self.fov.tolist()
        else:
            dct['distances'] = self.dis.tolist()
        if is_val(self.cen):
            dct['center'] = self.cen.tolist()
        if is_val(self.rot):
            dct['rotation'] = float(self.rot) if self.rot.size == 1 else self.rot.tolist()
        return dct



def space_coos(shp, dis, cen, rot):
    '''Generate the coordinates of the space.'''
    coos = jnp.indices(shp).astype(float)
    coos_T = coos.T.reshape(-1, 2)
    coos_T -= 0.5 * (shp - 1)
    coos_T *= dis
    coos_T = jax_rotate(coos_T, rot)
    coos_T += cen
    return coos_T.reshape(coos.T.shape).T


def space_lims(fov, cen):
    '''Generate the limits of the space.'''
    return fov[:,None] / 2 * np.array([-1, 1]) + cen[:,None]


def jax_rotate(xy, phi):
    '''Rotate the coordinates by an angle phi (jax implementation).'''
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    R = jnp.array([[c, s], [-s, c]])
    return xy @ R



class PointSpace():
    '''A class to represent point sources at several locations at the sky. Use `build` function to create the space.'''

    def __init__(self, coordinates, n_copies=1):
        from ..model.points import CoordinateModel
        check_type(coordinates, (CoordinateModel, tuple), (tuple, float), float)
        check_type(n_copies, int)

        self.coordinates = coordinates
        self.n_copies = n_copies

    def __repr__(self):
        return f'PointSpace(coordinates={self.coordinates})'
        
    def __call__(self, x):
        if isinstance(self.coordinates, Model):
            return self.coordinates(x)
        else:
            return np.array(self.coordinates)
    
    def __len__(self):
        return self.n_copies
    
    @classmethod
    def build(cls, *, coordinates, n_copies=1, **kwargs):
        '''
        Build a PointSpace object from the given parameters.
        
        Parameters
        ----------
        coordinates : dict or list
            The coordinates of the point sources
            -> if dict, a prior is defined with (mean, std) and coordinates can be learned
            -> if list, the coordinates are fixed to the given values
        n_copies : int, optional
            The number of point sources, by default 1
        kwargs : keyword arguments
            Additional parameters for the CoordinateModel (prefix)
        '''
        if isinstance(coordinates, dict):
            from ..model.points import CoordinateModel
            return cls(CoordinateModel.build(coordinates=coordinates, n_copies=n_copies, **kwargs))
        else:
            coos = to_shape(coordinates, (n_copies, 2), 'float64')
            coordinates = tuple(map(tuple, coos.tolist()))

            if n_copies == 1:
                coordinates = coordinates[0]
            
            return cls(coordinates, n_copies)

    @property
    def shape(self):
        return (1, 1)
    
    @property
    def shp(self):
        return np.array(self.shape)

    @property
    def coos(self):
        if isinstance(self.coordinates, Model):
            return self.coordinates
        else:
            return np.array(self.coordinates)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

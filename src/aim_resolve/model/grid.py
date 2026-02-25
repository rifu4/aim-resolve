"""Grid classes for defining signal and point model domains."""

import numpy as np
from jax import vmap

from .util import check_type, is_val, to_shape



class SignalGrid():
    """Signal grid at a specific location in the sky.

    Represents a rectangular pixel grid used for modeling extended signals.
    Use the ``build`` classmethod to create a grid instance.
    """

    def __init__(self, space, center=(0,0), factor=1, distances=(1.,1.), n_copies=1):
        check_type(space, tuple, int)
        check_type(center, tuple, (tuple, int), int)
        check_type(factor, int)
        check_type(distances, tuple, float)
        check_type(n_copies, int)

        self.space = space
        self.shape = tuple(s*factor for s in space)
        self.center = center
        self.factor = factor
        self.distances = distances
        self.n_copies = n_copies

    def __repr__(self):
        """Return a string representation of the SignalGrid."""
        return f'SignalGrid(space={self.space}, center={self.center}, factor={self.factor}, distances={self.distances})'
    
    def __eq__(self, other):
        if not isinstance(other, SignalGrid):
            raise ValueError('Can only compare equality for another SignalGrid.')
        return self.shape == other.shape and self.n_copies == other.n_copies and np.all(self.lims == other.lims)
    
    def __contains__(self, other):
        if not isinstance(other, (SignalGrid, PointGrid)):
            raise ValueError('Can only check containment for a SignalGrid or PointGrid.')
        s_lims = self.lims.reshape(-1,2,2).T
        o_lims = other.lims.reshape(-1,2,2).T
        return np.all(s_lims[0] <= o_lims[0]) and np.all(s_lims[1] >= o_lims[1]) 
    
    def __mul__(self, other):
        return self.multiply_space(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return self.multiply_space(1/other)
    
    @classmethod
    def build(cls, *, space, center=(0,0), factor=1, distances=None, fov=None, n_copies=1):
        """Build a SignalGrid from the given parameters.

        Parameters
        ----------
        space : int or tuple
            The space of the grid.
        center : int or tuple, optional
            The center of the space, by default (0, 0).
        factor : int, optional
            The upsampling factor of the space, by default 1.
        distances : float or tuple, optional
            The distance between the pixels, by default None.
        fov : float or tuple, optional
            The field of view of the space, by default None.
        n_copies : int, optional
            The number of copies of the space, by default 1.

        Returns
        -------
        SignalGrid
            The constructed signal grid instance.
        """
        n_c = to_shape(n_copies, (), 'int64')
        if not is_val(n_c):
            n_c = 1
        n_copies = int(n_c)

        spc = to_shape(space, (2,), 'int64')
        cen = to_shape(center, (n_copies, 2), 'int64')
        fac = to_shape(factor, (), 'int64')
        dis = to_shape(distances, (2,), 'float64')
        fov = to_shape(fov, (2,), 'float64')

        space = tuple(spc.tolist())

        if not is_val(cen):
            cen = np.zeros_like(cen)
        center = tuple(map(tuple, cen.tolist()))
        if n_copies == 1:
            center = center[0]

        if not is_val(fac):
            fac = 1
        factor = int(fac)

        if is_val(dis):
            distances = tuple(dis.tolist())
        elif is_val(fov):
            distances = tuple((fov / spc).tolist())
        else:
            distances = tuple((1 / spc).tolist())

        return cls(space, center, factor, distances, n_copies)

    @property
    def spc(self):
        """Space dimensions as a NumPy array."""
        return np.array(self.space)
    
    @property
    def shp(self):
        """Shape of the grid as a NumPy array."""
        return np.array(self.shape)

    @property
    def cen(self):
        """Center coordinates as a NumPy array."""
        return np.array(self.center)
    
    @property
    def fac(self):
        """Upsampling factor as a NumPy array."""
        return np.array(self.factor)
    
    @property
    def dis(self):
        """Pixel distances as a NumPy array."""
        return np.array(self.distances)
    
    @property
    def fov(self):
        """Field of view of the grid."""
        return self.spc * self.dis
    
    @property
    def ndim(self):
        """Number of dimensions of the grid."""
        return len(self.space)

    @property
    def size(self):
        """Total number of grid elements."""
        return self.shp.prod()
    
    @property
    def dvol(self):
        """Volume element of one pixel."""
        return self.dis.prod() / (self.fac**self.ndim)
        
    @property
    def lims(self):
        """Spatial limits of the grid."""
        if self.n_copies == 1:
            return space_lims(self.spc, self.cen)
        else:
            return vmap(space_lims, in_axes=(None, 0))(self.spc, self.cen)
        
    def refine(self, factor):
        """Multiply the resolution of the grid by a factor."""
        check_type(factor, int)
        return SignalGrid(self.space, self.center, self.factor * factor, self.distances, self.n_copies)
    
    def update(self, **kwargs):
        dct = self.to_dict()
        dct.update(kwargs)
        return SignalGrid.build(**dct)

    def multiply_space(self, factor):
        """Multiply the space of the grid by a factor."""
        check_type(factor, (int, float))
        space = tuple(int(round(si * factor)) for si in self.space)
        return SignalGrid(space, self.center, self.factor, self.distances, self.n_copies)

    def to_dict(self, *keys):
        """Convert the grid to a dictionary ({space: [sx,sy], ...})."""
        if not keys:
            keys = ('center', 'factor', 'n_copies')
        dct = {'space': self.spc.tolist()}
        if 'center' in keys and is_val(self.cen):
            dct['center'] = self.cen.tolist()
        if 'factor' in keys and self.factor != 1:
            dct['factor'] = int(self.factor)
        if 'fov' in keys:
            dct['fov'] = self.fov.tolist()
        elif 'distances' in keys:
            dct['distances'] = self.dis.tolist()
        if 'n_copies' in keys and self.n_copies != 1:
            dct['n_copies'] = int(self.n_copies)
        return dct


def space_lims(spc, cen):
    """Generate the spatial limits from space dimensions and center.

    Parameters
    ----------
    spc : array-like
        Space dimensions.
    cen : array-like
        Center coordinates.

    Returns
    -------
    numpy.ndarray
        Array of lower and upper limits for each dimension.
    """
    return spc[:,None] / 2 * np.array([-1, 1]) + cen[:,None]
    


class PointGrid():
    """Point grid for modeling point sources at specific sky coordinates.

    Use the ``build`` classmethod to create a grid instance.
    """

    def __init__(self, coordinates, factor=1, n_copies=1):
        check_type(coordinates, tuple, (tuple, float), float)
        check_type(factor, int)
        check_type(n_copies, int)

        self.shape = (1, 1)
        self.coordinates = coordinates
        self.factor = factor
        self.n_copies = n_copies

    def __repr__(self):
        """Return a string representation of the PointGrid."""
        return f'PointGrid(coordinates={self.coordinates}, factor={self.factor}, n_copies={self.n_copies})'
    
    @classmethod
    def build(cls, *, coordinates, factor=1, n_copies=1):
        """Build a PointGrid from the given parameters.

        Parameters
        ----------
        coordinates : float or tuple
            The coordinates of the point sources.
        factor : int, optional
            The upsampling factor, by default 1.
        n_copies : int, optional
            The number of point sources, by default 1.

        Returns
        -------
        PointGrid
            The constructed point grid instance.
        """
        n_c = to_shape(n_copies, (), 'int64')
        if not is_val(n_c):
            n_c = 1
        n_copies = int(n_c)

        fac = to_shape(factor, (), 'int64')
        if not is_val(fac):
            fac = 1
        factor = int(fac)

        coos = to_shape(coordinates, (n_copies, 2), 'float64')
        grid = np.linspace(1/(2*factor), 1 - 1/(2*factor), factor)
        if not np.all(np.isin(coos % 1, grid)):
            raise ValueError(f'For a factor of {factor}, coordinates % 1 must be in {grid}.')
        coordinates = tuple(map(tuple, coos.tolist()))
        if n_copies == 1:
            coordinates = coordinates[0]

            
        return cls(coordinates, factor, n_copies)

    @property
    def coos(self):
        """Coordinates as a NumPy array."""
        return np.array(self.coordinates)
    
    @property
    def shp(self):
        """Shape of the grid as a NumPy array."""
        return np.array(self.shape)
    
    @property
    def fac(self):
        """Upsampling factor as a NumPy array."""
        return np.array(self.factor)
    
    @property
    def ndim(self):
        """Number of dimensions of the grid."""
        return len(self.shape)

    @property
    def size(self):
        """Total number of grid elements."""
        return self.shp.prod()
    
    @property
    def lims(self):
        """Spatial limits of the point grid."""
        return self.coos.reshape(-1, 2)[:,:,None].repeat(2, axis=-1)
    
    def refine(self, factor):
        offsets = np.array([[-1, -1], [-1,  1], [ 1, -1], [ 1,  1]]) / (2 * self.fac * factor)
        coos = self.coos.reshape(-1, 2)[None, :, :] + offsets[:, None, :]
        coos = coos.reshape(-1, 2)
        order = np.concatenate([np.arange(i, coos.shape[0], 2) for i in range(2)])
        coos = coos[order]
        return PointGrid(tuple(map(tuple, coos.tolist())), self.factor * factor, self.n_copies * factor**2)
    
    def update(self, **kwargs):
        dct = self.to_dict()
        dct.update(kwargs)
        return PointGrid.build(**dct)

    def to_dict(self, *keys):
        """Convert the grid to a dictionary ({coordinates: [[cx,cy], ...], ...})."""
        if not keys:
            keys = ('factor', 'n_copies')
        dct = {'coordinates': self.coos.tolist()}
        if 'factor' in keys and self.factor != 1:
            dct['factor'] = int(self.factor)
        if 'n_copies' in keys and self.n_copies != 1:
            dct['n_copies'] = int(self.n_copies)
        return dct

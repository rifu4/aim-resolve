import numpy as np
from scipy.ndimage import distance_transform_edt

from .model.components import ComponentModel
from .model.map import map_signal
from .model.util import check_type, to_shape



def masks_from_maps(
        points_map,
        object_maps,
        it,
        freq = [1.],
        factor = 1,
        margin_fac = 0.2,
        margin_min = 2,
        max_objects = 5,
        tile_size = 0,
):
    '''
    Create masks from point source and object maps.

    Parameters
    ----------
    points_map : np.ndarray
        The point source map.
    object_maps : np.ndarray
        The object maps.
    it : int
        The iteration number.
    factor : int, optional
        The refinement factor for the masks. Default is 1.
    margin_fac : float, optional
        The margin factor for the object maps. Default is 0.2.
    margin_min : int, optional
        The minimum margin for the object maps. Default is 2.
    max_objects : int, optional
        The maximum number of objects to include in the masks dict. Default is 5.
    tile_size : int, optional
        The size of the tiles. Default is 0.
        -> If an object fits into the tile size, it will be added to the tile mask.
    '''
    mask_dct = {}
    margin_min *= factor
    tile_size = to_shape(tile_size, (2,), 'int64') * factor

    if np.any(points_map == 1):
        ps_coos = np.argwhere(points_map == 1)
        ps_maps = np.zeros((len(ps_coos),) + points_map.shape)
        for i,co in enumerate(ps_coos):
            ps_maps[i, co[0], co[1]] = 1
            ps_maps[i] = add_margin(ps_maps[i], margin_min, round=False)
        mask_dct[f'p0.{it}'] = np.asarray(ps_maps)

    oj_maps, ts_maps = [], []
    for i in range(object_maps.shape[0]):
        o_map = object_maps[i]
        o_pix = [1 + om.max() - om.min() for om in np.where(o_map == 1)]
        o_mrg = [max(margin_min, np.ceil(om * margin_fac).astype(int)) for om in o_pix]
        o_mrg = int(max(o_mrg))
        o_map = add_margin(o_map, o_mrg, round=False)

        o_pix = [1 + om.max() - om.min() for om in np.where(o_map>0)]
        if np.all(o_pix <= tile_size):
            ts_maps.append(o_map)
        elif i < max_objects:
            oj_maps.append(o_map)

    for i in range(len(oj_maps)):
        mask_dct[f'o{i}.{it}'] = np.asarray(oj_maps[i])

    if len(ts_maps) > 0:
        ts_maps = np.concatenate([ti[None] for ti in ts_maps], axis=0)
        mask_dct[f't0.{it}'] = np.asarray(ts_maps)

    mask_dct['sum'] = np.sum([np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_dct.values()], axis=0)

    mask_dct[f'bg.{it}'] = np.floor(1 - mask_dct['sum']).clip(0,1)

    if len(freq) > 1:
        for k, v in mask_dct.items():
            mask_dct[k] = add_freq_axis(v, freq)

    return mask_dct



def masks_from_model(
        sky,
        factor = 1,
        margin_min = 2,
):
    '''
    Create masks from a sky model.

    Parameters
    ----------
    sky : ComponentModel
        The sky model.
        -> Creates masks for all components in the model (points, objects, tiles).
    factor : int, optional
        The refinement factor for the masks. Default is 1.
    margin_min : int, optional
        The minimum margin for the point sources. Default is 2.
    '''
    check_type(sky, ComponentModel)
    mask_dct = {}
    margin_min *= factor

    for sky_pi in sky.points:
        ones_pi = remove_freq_axis(np.ones(sky_pi.shape), sky.freq)
        mask_pi = np.array(map_signal(sky_pi.points.grid, sky.grid.update(n_copies=sky_pi.n_copies))(ones_pi))
        for i in range(mask_pi.shape[0]):
            mask_pi[i] = add_margin(mask_pi[i], margin_min, round=True)
        mask_dct[sky_pi.prefix] = np.asarray(mask_pi)

    for sky_oi in sky.objects:
        ones_oi = remove_freq_axis(np.ones(sky_oi.shape), sky.freq)
        mask_oi = map_signal(sky_oi.grid, sky.grid)(ones_oi)
        mask_dct[sky_oi.prefix] = np.asarray(mask_oi)

    for sky_ti in sky.tiles:
        ones_ti = remove_freq_axis(np.ones(sky_ti.shape), sky.freq)
        mask_ti = map_signal(sky_ti.tiles.grid, sky.grid.update(n_copies=sky_ti.n_copies))(ones_ti)
        mask_dct[sky_ti.prefix] = np.asarray(mask_ti)

    mask_dct['sum'] = np.sum([np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_dct.values()], axis=0)

    mask_dct[sky.background.prefix] = np.floor(1 - mask_dct['sum']).clip(0,1)

    if sky.freq.size > 1:
        for k, v in mask_dct.items():
            mask_dct[k] = add_freq_axis(v, sky.freq)
    
    return mask_dct



def masks_to_boxes(
        sky,
        mask_dct,
):
    '''
    Maps the masks to the grids of the model components and subtracts other components from the masks.

    Parameters
    ----------
    sky : ComponentModel
        The sky model.
    mask_dct : dict
        Dictionary containing the masks for the components. 
        -> created using the `masks_from_maps` or `masks_from_model` function.
    '''
    check_type(sky, ComponentModel)

    mask_box = mask_dct.copy()

    sky_bg = sky.background
    if mask_dct[sky_bg.prefix].shape != sky_bg.grid.shape:
        mask_bg = np.floor(map_signal(sky.grid, sky_bg.grid)(mask_dct[sky_bg.prefix]))
        mask_box[sky_bg.prefix] = np.asarray(mask_bg.astype(bool))

    for sky_pi in sky.points:
        if mask_dct[sky_pi.prefix].shape != sky_pi.grid.shape:
            mask_pi = np.ceil(map_signal(sky.grid, sky_pi.grid)(mask_dct[sky_pi.prefix]))
            mask_box[sky_pi.prefix] = np.asarray(mask_pi.astype(bool))

    for sky_oi in sky.objects:  
        if mask_dct[sky_oi.prefix].shape != sky_oi.grid.shape:
            mask_oi = map_signal(sky.grid, sky_oi.grid)(mask_dct[sky_oi.prefix])
            if np.any((mask_oi > 0.) & (mask_oi < 1.)) and 'sum' in mask_dct:
                mask_oi = (2 * mask_dct[sky_oi.prefix] - mask_dct['sum']).clip(0,1)
                mask_oi = np.ceil(map_signal(sky.grid, sky_oi.grid)(mask_oi))
            mask_box[sky_oi.prefix] = np.asarray(mask_oi.astype(bool))

    for sky_ti in sky.tiles:
        if mask_dct[sky_ti.prefix].shape != sky_ti.grid.shape:
            mask_ti = map_signal(sky.grid, sky_ti.grid)(mask_dct[sky_ti.prefix])
            if np.any((mask_ti > 0.) & (mask_ti < 1.)) and 'sum' in mask_dct:
                mask_ti = (2 * mask_dct[sky_ti.prefix] - mask_dct['sum']).clip(0,1)
                mask_ti = np.ceil(map_signal(sky.grid, sky_ti.grid)(mask_ti))
            mask_box[sky_ti.prefix] = np.asarray(mask_ti.astype(bool))

    return mask_box



def add_margin(array, margin, round=False):
    '''Adds a falloff margin to the input array using the `scipy.ndimage.distance_transform_edt` function.'''
    if np.all(array == 0):
        return array
    if isinstance(margin, int):
        margin = (margin, margin)
    mx, my = margin
    new_array = distance_transform_edt(1 - array, sampling=[1/(mx+.5), 1/(my+.5)])
    new_array = (1 - new_array).clip(0,1)
    if round:
        new_array = np.ceil(new_array)
    return new_array


def add_freq_axis(array, freq):
    if len(freq) > 1:
        if array.ndim == 2:
            return array[None, :, :]
        elif array.ndim == 3:
            return array[:, None, :, :]
    return array


def remove_freq_axis(array, freq):
    if len(freq) > 1:
        if array.ndim == 4:
            return array[:, 0, :, :]
        elif array.ndim == 3:
            return array[0, :, :]
    return array

import numpy as np
from scipy.ndimage import distance_transform_edt

from .model.components import ComponentModel
from .model.map import map_signal, map_points
from .model.util import check_type, to_shape



def masks_from_maps(
        points_map,
        object_maps,
        it,
        zoom = 1,
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
    zoom : int, optional
        The zoom factor for the point source map. Default is 1.
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
    margin_min *= zoom
    tile_size = to_shape(tile_size, (2,), 'int64')

    if np.any(points_map == 1):
        ps_coos = np.argwhere(points_map == 1)
        ps_maps = np.zeros((len(ps_coos),) + points_map.shape)
        for i,co in enumerate(ps_coos):
            ps_maps[i, co[0], co[1]] = 1
            ps_maps[i] = add_margin(ps_maps[i], margin_min, round=False)
        mask_dct[f'p0.{it}'] = ps_maps

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
        mask_dct[f'o{i}.{it}'] = oj_maps[i]

    if len(ts_maps) > 0:
        ts_maps = np.concatenate([ti[None] for ti in ts_maps], axis=0)
        mask_dct[f't0.{it}'] = ts_maps

    mask_dct['sum'] = np.sum([np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_dct.values()], axis=0)

    mask_dct[f'bg.{it}'] = np.floor(1 - mask_dct['sum']).clip(0,1)

    return mask_dct



def masks_from_model(
        sky,
        margin_min = 2,
):
    '''
    Create masks from a sky model.

    Parameters
    ----------
    sky : ComponentModel
        The sky model.
        -> Creates masks for all components in the model (points, objects, tiles).
    margin_min : int, optional
        The minimum margin for the point sources. Default is 2.
    '''
    check_type(sky, ComponentModel)
    mask_dct = {}
    zoom = int(np.mean(sky.background.space.dis / sky.space.dis))
    margin_min *= zoom

    for sky_pi in sky.points:
        mask_pi = np.array(map_points(np.ones(sky_pi.shape), sky_pi.points.space.coos, sky.space, vmap_sum=False))
        for i in range(mask_pi.shape[0]):
            mask_pi[i] = add_margin(mask_pi[i], margin_min, round=True)
        mask_dct[sky_pi.prefix] = mask_pi

    for sky_oi in sky.objects:
        mask_dct[sky_oi.prefix] = map_signal(np.ones(sky_oi.shape), sky_oi.space, sky.space)

    for sky_ti in sky.tiles:
        mask_dct[sky_ti.prefix] = map_signal(np.ones(sky_ti.shape), sky_ti.tiles.space, sky.space, vmap_sum=False)

    mask_dct['sum'] = np.sum([np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_dct.values()], axis=0)

    mask_dct[sky.background.prefix] = np.floor(1 - mask_dct['sum']).clip(0,1)
    
    return mask_dct



def masks_to_boxes(
        sky, 
        mask_dct,
):
    '''
    Maps the masks to the spaces of the model compoennts and subtracts other components from the masks.

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
    if mask_dct[sky_bg.prefix].shape != sky_bg.space.shape:
        mask_box[sky_bg.prefix] = np.floor(map_signal(mask_dct[sky_bg.prefix], sky.space, sky_bg.space, order=1))

    for sky_pi in sky.points:
        if mask_dct[sky_pi.prefix].shape != sky_pi.space.shape:
            mask_box[sky_pi.prefix] = np.ceil(map_signal(mask_dct[sky_pi.prefix], sky.space, sky_pi.space, order=1))

    for sky_oi in sky.objects:  
        if mask_dct[sky_oi.prefix].shape != sky_oi.space.shape:
            mask_oi = map_signal(mask_dct[sky_oi.prefix], sky.space, sky_oi.space)
            if np.any((mask_oi > 0.) & (mask_oi < 1.)) and 'sum' in mask_dct:
                mask_oi = (2 * mask_dct[sky_oi.prefix] - mask_dct['sum']).clip(0,1)
                mask_oi = np.ceil(map_signal(mask_oi, sky.space, sky_oi.space))
            mask_box[sky_oi.prefix] = mask_oi

    for sky_ti in sky.tiles:
        if mask_dct[sky_ti.prefix].shape != sky_ti.space.shape:
            mask_ti = map_signal(mask_dct[sky_ti.prefix], sky.space, sky_ti.space)
            if np.any((mask_ti > 0.) & (mask_ti < 1.)) and 'sum' in mask_dct:
                mask_ti = (2 * mask_dct[sky_ti.prefix] - mask_dct['sum']).clip(0,1)
                mask_ti = np.ceil(map_signal(mask_ti, sky.space, sky_ti.space))
            mask_box[sky_ti.prefix] = mask_ti

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

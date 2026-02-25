"""Mask creation and manipulation utilities for sky model components."""

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
    """Create component masks from point-source and object detection maps.

    Parameters
    ----------
    points_map : np.ndarray
        Binary point-source map.
    object_maps : np.ndarray
        Array of binary object maps with shape ``(n_objects, H, W)``.
    it : int
        Iteration number used as suffix in the mask keys.
    freq : list, optional
        List of frequencies. A frequency axis is added when
        ``len(freq) > 1``. Default is ``[1.]``.
    factor : int, optional
        Refinement factor applied to margins and tile sizes. Default is 1.
    margin_fac : float, optional
        Fractional margin around objects. Default is 0.2.
    margin_min : int, optional
        Minimum margin in pixels (before applying *factor*). Default is 2.
    max_objects : int, optional
        Maximum number of individual object masks to include. Default is 5.
    tile_size : int, optional
        Tile size threshold. Objects fitting within this size are grouped
        as tiles. Default is 0.

    Returns
    -------
    mask_dct : dict
        Dictionary of mask arrays keyed by component identifier.
    """
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
    """Create component masks from an existing sky model.

    Generates one mask per model component (points, objects, tiles,
    background) based on the component grids.

    Parameters
    ----------
    sky : ComponentModel
        Sky model containing the component definitions.
    factor : int, optional
        Refinement factor applied to margins. Default is 1.
    margin_min : int, optional
        Minimum margin in pixels (before applying *factor*) for point
        sources. Default is 2.

    Returns
    -------
    mask_dct : dict
        Dictionary of mask arrays keyed by component prefix.
    """
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
    """Map masks to the grids of the corresponding model components.

    Reprojects each mask onto the component's own grid and handles
    overlap between neighbouring components.

    Parameters
    ----------
    sky : ComponentModel
        Sky model providing the component grid definitions.
    mask_dct : dict
        Mask dictionary produced by ``masks_from_maps`` or
        ``masks_from_model``.

    Returns
    -------
    mask_box : dict
        Dictionary of boolean mask arrays on the respective component
        grids.
    """
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
    """Add a smooth fall-off margin around non-zero regions.

    Uses ``scipy.ndimage.distance_transform_edt`` to create a distance-based
    fall-off that extends *margin* pixels from the boundary of the input.

    Parameters
    ----------
    array : np.ndarray
        2-D binary input array.
    margin : int or tuple of int
        Margin size in pixels. A scalar is applied to both axes.
    round : bool, optional
        If True, the result is rounded up to binary values. Default is
        False.

    Returns
    -------
    new_array : np.ndarray
        Array of the same shape with the added margin.
    """
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
    """Insert a frequency axis into a spatial array.

    Parameters
    ----------
    array : np.ndarray
        Array with 2 or 3 spatial dimensions.
    freq : array_like
        Frequency list. The axis is only added when ``len(freq) > 1``.

    Returns
    -------
    array : np.ndarray
        Array with an additional frequency axis (if applicable).
    """
    if len(freq) > 1:
        if array.ndim == 2:
            return array[None, :, :]
        elif array.ndim == 3:
            return array[:, None, :, :]
    return array


def remove_freq_axis(array, freq):
    """Remove the frequency axis from an array.

    Parameters
    ----------
    array : np.ndarray
        Array with 3 or 4 dimensions including a frequency axis.
    freq : array_like
        Frequency list. The axis is only removed when ``len(freq) > 1``.

    Returns
    -------
    array : np.ndarray
        Array with the frequency axis removed (if applicable).
    """
    if len(freq) > 1:
        if array.ndim == 4:
            return array[:, 0, :, :]
        elif array.ndim == 3:
            return array[0, :, :]
    return array

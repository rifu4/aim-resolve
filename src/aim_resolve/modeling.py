"""Sky-model construction utilities for multi-component reconstruction."""

import numpy as np

from .mask import remove_freq_axis
from .model.grid import SignalGrid, PointGrid
from .model.map import map_signal
from .model.points import PointModel
from .model.signal import SignalModel
from .model.tiles import TileModel
from .model.util import to_shape



def model_background(
        bg_mask,
        freq,
        rec_val,
    ):
    """Create a background model configuration dictionary.

    Parameters
    ----------
    bg_mask : np.ndarray
        Binary mask for the background region.
    freq : list
        List of frequencies.
    rec_val : np.ndarray
        Reconstruction values to derive the offset from.

    Returns
    -------
    bg_dct : dict
        Background model dictionary with an ``offset`` entry.
    """
    bg_dct = dict(
        offset =  get_offset('background', rec_val, bg_mask, freq),
    )
    return bg_dct



def model_points(
        ps_masks,
        ps_map,
        grid,
        freq,
        rec_sub,
    ):
    """Create a point-source model configuration dictionary.

    Extracts point-source locations from the detection map, converts
    pixel coordinates to grid coordinates and computes initial offsets.

    Parameters
    ----------
    ps_masks : np.ndarray
        Masks for the individual point sources.
    ps_map : np.ndarray
        Binary point-source detection map.
    grid : SignalGrid
        Reference grid for coordinate conversion.
    freq : list
        List of frequencies.
    rec_sub : np.ndarray
        Background-subtracted reconstruction for offset calculation.

    Returns
    -------
    ps_dct : dict or False
        Point-source model dictionary, or ``False`` if no sources found.
    """
    ps_coos = np.argwhere(ps_map == 1).astype('float64')

    # check if there are any point sources to extract, if not return empty list
    if ps_coos.size == 0:
        return False
    
    # convert the pixel values of the point sources to coordinates on the grid
    ps_coos -= 0.5 * (grid.shp - 1)
    ps_coos /= grid.fac
    ps_coos += grid.cen
    point_grid = PointGrid.build(coordinates=ps_coos, factor=grid.factor, n_copies=ps_coos.shape[0])

    # create point model dictionary
    ps_dct = dict(
        point_grid = point_grid.to_dict(),
        grid = grid.to_dict('center'),
        freq = freq,
        params = dict(
            base = 'params_ps',
        ),
        offset = get_offset('point', rec_sub, ps_masks, freq),
    )
    return ps_dct



def model_objects(
        oj_mask,
        grid,
        freq,
        rec_sub,
        gaussian = None,
):
    """Create an extended-object model configuration dictionary.

    Determines the bounding box from the mask, builds a sub-grid and
    computes initial offsets.

    Parameters
    ----------
    oj_mask : np.ndarray
        Binary mask for the extended object.
    grid : SignalGrid
        Reference grid for coordinate conversion.
    freq : list
        List of frequencies.
    rec_sub : np.ndarray
        Background-subtracted reconstruction for offset calculation.
    gaussian : dict or None, optional
        Gaussian smoothing parameters (``mean_fac``, ``std_fac``).
        Default is None.

    Returns
    -------
    oj_dct : dict
        Object model dictionary with grid, frequency and offset entries.
    """
    pix = np.argwhere(remove_freq_axis(oj_mask, freq) > 0)
    lim = np.array([pix.min(axis=0) - 1, pix.max(axis=0) + 1])
    lim = lim.clip(0, grid.shp-1)
    shp = 1 + lim[1] - lim[0]
    shp[shp%2 != 0] += 1
    cen = lim.mean(axis=0).astype('int64')
    spc = shp / grid.fac
    cen = (cen - 0.5 * (grid.shp - 1)) / grid.fac + grid.cen
    
    oj_grid = SignalGrid.build(space=spc, center=cen, factor=grid.factor)

    # create object model dictionary
    oj_dct = dict(
        grid = oj_grid.to_dict(),
        freq = freq,
        params = dict(
            base = 'params_mf' if len(freq) > 1 else 'params_sf',
        ),
        offset = get_offset('object', rec_sub, oj_mask, freq),
    )

    if gaussian:
        oj_dct['gaussian'] = dict(
            cov_x = [float(gaussian['mean_fac']), float(gaussian['std_fac'])],
            cov_y = [float(gaussian['mean_fac']), float(gaussian['std_fac'])],
        )

    return oj_dct



def model_tiles(
        ts_masks,
        grid,
        freq,
        rec_sub,
        tile_size = 32,
        gaussian = None,
):
    """Create a tile model configuration dictionary.

    Computes tile centres from the provided masks and builds a tile grid.

    Parameters
    ----------
    ts_masks : np.ndarray
        Array of binary tile masks.
    grid : SignalGrid
        Reference grid for coordinate conversion.
    freq : list
        List of frequencies.
    rec_sub : np.ndarray
        Background-subtracted reconstruction for offset calculation.
    tile_size : int, optional
        Spatial size of each tile in pixels. Default is 32.
    gaussian : dict or None, optional
        Gaussian smoothing parameters (``mean_fac``, ``std_fac``).
        Default is None.

    Returns
    -------
    ts_dct : dict
        Tile model dictionary with grid, frequency and offset entries.
    """
    tile_size = to_shape(tile_size, (2,), 'int64')

    ts_cen = []
    for tm in ts_masks:
        pix = np.argwhere(remove_freq_axis(tm, freq) > 0)
        lim = np.array([pix.min(axis=0) - 1, pix.max(axis=0) + 1])
        cen = lim.mean(axis=0).astype('int64')
        cen = cen.clip(tile_size * grid.fac // 2, grid.shp - (tile_size * grid.fac // 2) - 1)
        cen = (cen - 0.5 * (grid.shp - 1)) / grid.fac + grid.cen
        ts_cen.append(cen.tolist())

    tile_grid = SignalGrid.build(space=tile_size, center=ts_cen, factor=grid.factor, n_copies=len(ts_cen))

    # create tile model dictionary
    ts_dct = dict(
        tile_grid = tile_grid.to_dict(),
        grid = grid.to_dict('center'),
        freq = freq,
        params = dict(
            base = 'params_mf' if len(freq) > 1 else 'params_sf',
        ),
        offset = get_offset('tile', rec_sub, ts_masks, freq),
    )

    if gaussian:
        ts_dct['gaussian'] = dict(
            cov_x = [float(gaussian['mean_fac']), float(gaussian['std_fac'])],
            cov_y = [float(gaussian['mean_fac']), float(gaussian['std_fac'])],
        )

    return ts_dct



def get_offset(
        model,
        rec_sub,
        mask,
        freq,
):
    """Compute the log-scale offset for a model component.

    The offset is derived from the reconstruction within the masked
    region and depends on the model type.

    Parameters
    ----------
    model : str or Model
        Model instance or descriptive string (e.g. ``'point'``,
        ``'background'``, ``'tile'``).
    rec_sub : np.ndarray
        Background-subtracted reconstruction values.
    mask : np.ndarray
        Boolean mask selecting the relevant region.
    freq : list
        List of frequencies (used to strip the frequency axis).

    Returns
    -------
    offset : float or list of float
        Log-scale offset(s) for the component.
    """
    rec_sub = remove_freq_axis(rec_sub, freq)
    mask = remove_freq_axis(mask, freq).astype(bool)

    if isinstance(model, PointModel) or (isinstance(model, str) and 'point' in model):
        log_sum = np.log(np.sum(np.broadcast_to(rec_sub, mask.shape), axis=(1, 2), where=mask))
        offset = [round(float(ri), 1) for ri in log_sum]

    elif isinstance(model, SignalModel) or (isinstance(model, str) and any(m in model for m in ('signal', 'object', 'background'))):
        log_mean = np.log(np.mean(rec_sub, where=mask))
        offset = round(float(log_mean), 1)

    elif isinstance(model, TileModel) or (isinstance(model, str) and 'tile' in model):
        log_mean = np.log(np.mean(np.broadcast_to(rec_sub, mask.shape), axis=(1, 2), where=mask))
        offset = [round(float(ri), 1) for ri in log_mean]

    if isinstance(model, str):
        print(f'{model} offset:', offset)
    else:
        print(f'{model.prefix} offset:', offset)

    return offset



def draw_boxes(cfg_sections, grid, it):
    """Draw bounding boxes of tile and object components on a map.

    Parameters
    ----------
    cfg_sections : dict
        Configuration sections containing component grid definitions.
    grid : SignalGrid
        Reference grid that defines the output map size.
    it : int
        Iteration number used to filter relevant sections.

    Returns
    -------
    box_map : np.ndarray
        2-D map with component outlines drawn as ones.
    """
    box_map = np.zeros(grid.shape)

    for k,v in cfg_sections.items():
        if 'sky_t' in k and f'.{it}' in k:
            grd_i = SignalGrid.build(**v['tile_grid'])
            val_i = np.ones((grd_i.n_copies,) + grd_i.shape)
            val_i[:, 1:-1, 1:-1] = 0
            box_map += np.squeeze(map_signal(grd_i, grid)(val_i))
        elif 'sky_o' in k and f'.{it}' in k:
            grd_i = SignalGrid.build(**v['grid'])
            val_i = np.ones(grd_i.shape)
            val_i[1:-1, 1:-1] = 0
            box_map += np.array(map_signal(grd_i, grid)(val_i))
    
    return box_map.clip(0,1)

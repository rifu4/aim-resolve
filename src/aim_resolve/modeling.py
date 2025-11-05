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
    # create background model dictionary
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
    # extract locations of the point sources from the output map
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
    '''Sets the offsets of the sky model based on the background reconstruction and the mask.'''
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
    box_map = np.zeros(grid.shape)

    for k,v in cfg_sections.items():
        if 'sky_t' in k and f'.{it}' in k:
            si = SignalGrid.build(**v['tile_grid'])
            xi = np.ones((si.n_copies,)+si.shape)
            xi[:, 1:-1, 1:-1] = 0
            box_map += np.array(map_signal(si, grid)(xi))
        elif 'sky_o' in k and f'.{it}' in k:
            si = SignalGrid.build(**v['grid'])
            xi = np.ones(si.shape)
            xi[1:-1, 1:-1] = 0
            box_map += np.array(map_signal(si, grid)(xi))
    
    return box_map.clip(0,1)

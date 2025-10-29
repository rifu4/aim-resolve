import numpy as np

from .model.grid import SignalGrid, PointGrid
from .model.map import map_signal
from .model.util import to_shape



def model_background(
        bg_mask,
        rec_val,
    ):
    log_val = np.log(rec_val[bg_mask > 0])

    bg_mean = round(float(log_val.mean()), 1)
    bg_std = round(float(log_val.std()), 1)

    bg_dct = {
        'i0': {
            'base': 'i0_bg',
            'offset_mean': bg_mean,
            'offset_std': [max(bg_std, 1.0), 1.0],
        },
    }
    return bg_dct



def model_points(
        ps_masks,
        ps_map,
        grid,
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

    # get the i0 priors for the point sources from the reconstruction
    offsets = np.log(np.sum(rec_sub[None] * ps_masks, axis=(1,2), where=(ps_masks > 0)))

    ps_dct = {
        'point_grid': point_grid.to_dict(),
        'grid': grid.to_dict('center'),
        'i0': {
            'base': 'i0_ps',
        },
        'offset': [round(float(ri), 1) for ri in offsets],
    }
    return ps_dct



def model_objects(
        oj_mask,
        grid,
        rec_sub,
        gaussian = None,
        zero_pad = None,
):
    pix = np.argwhere(oj_mask > 0)
    lim = np.array([pix.min(axis=0) - 1, pix.max(axis=0) + 1])
    lim = lim.clip(0, grid.shp-1)
    shp = 1 + lim[1] - lim[0]
    shp[shp%2 != 0] += 1
    cen = lim.mean(axis=0).astype('int64')
    spc = shp / grid.fac
    cen = (cen - 0.5 * (grid.shp - 1)) / grid.fac + grid.cen
    
    oj_grid = SignalGrid.build(space=spc, center=cen, factor=grid.factor)

    log_mean = np.log(np.mean(rec_sub * oj_mask, where=(oj_mask > 0)))
    offset = round(float(log_mean), 1)

    oj_dct = {
        'grid': oj_grid.to_dict(),
        'i0': {
            'base': 'i0_os',
        },
        'offset': offset,
    }
    if gaussian:
        g_mean, g_std = gaussian['mean_fac'], gaussian['std_fac']
        fov_x, fov_y = oj_grid.fov
        #TODO: check if this is correct for grid
        oj_dct['gaussian'] = {
            'cov_x': [float(g_mean * fov_x), float(g_std * fov_x)],
            'cov_y': [float(g_mean * fov_y), float(g_std * fov_y)],
        }
    if zero_pad:
        oj_dct['zero_pad'] = zero_pad

    return oj_dct



def model_tiles(
        ts_masks,
        grid,
        rec_sub,
        tile_size = 32,
        gaussian = None,
):
    tile_size = to_shape(tile_size, (2,), 'int64')

    ts_cen = []
    for tm in ts_masks:
        pix = np.argwhere(tm > 0)
        lim = np.array([pix.min(axis=0) - 1, pix.max(axis=0) + 1])
        cen = lim.mean(axis=0).astype('int64')
        cen = cen.clip(tile_size * grid.fac // 2, grid.shp - (tile_size * grid.fac // 2) - 1)
        cen = (cen - 0.5 * (grid.shp - 1)) / grid.fac + grid.cen
        ts_cen.append(cen.tolist())

    tile_grid = SignalGrid.build(space=tile_size, center=ts_cen, factor=grid.factor, n_copies=len(ts_cen))

    offsets = np.log(np.mean(rec_sub[None] * ts_masks, axis=(1,2), where=(ts_masks > 0)))

    ts_dct = {
        'tile_grid': tile_grid.to_dict(),
        'grid': grid.to_dict('center'),
        'i0': {
            'base': 'i0_ts',
        },
        'offset': [round(float(ri), 1) for ri in offsets],
    }
    if gaussian:
        g_mean, g_std = gaussian['mean_fac'], gaussian['std_fac']
        fov_x, fov_y = tile_grid.fov
        #TODO: check if this is correct for grid
        ts_dct['gaussian'] = {
            'cov_x': [float(g_mean * fov_x), float(g_std * fov_x)],
            'cov_y': [float(g_mean * fov_y), float(g_std * fov_y)],
        }

    return ts_dct



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

import numpy as np

from .model.map import map_signal
from .model.space import SignalSpace
from .model.util import check_type, to_shape



def adjust_zoom(
        zoom, 
        space, 
        bg_space,
    ):
    '''Adjust the zoom factor if the space resolution is a multiple of the background space resolution.
    
    Parameters
    ----------
    zoom : int
        The zoom factor.
    space : SignalSpace
        The space of the full sky model.
    bg_space : SignalSpace
        The space of the background model.

    Returns
    -------
    zoom : int
        The adjusted zoom factor.
    '''
    check_type(space, SignalSpace)
    check_type(bg_space, SignalSpace)
    if not isinstance(zoom, int) or zoom < 1:
        raise TypeError('`zoom` has to be of type `int` and larger than 0')
    
    for zi in range(1, zoom+1):
        if space == zi * bg_space:
            zoom = zoom // zi

    return zoom



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
        space,
        rec_sub,
    ):
    # extract locations of the point sources from the output map
    ps_coos = np.argwhere(ps_map == 1).astype('float64')

    # check if there are any point sources to extract, if not return empty list
    if ps_coos.size == 0:
        return False
    
    # convert the pixel values of the point sources to coordinates in the space
    ps_coos -= 0.5 * (space.shp - 1)
    ps_coos *= space.dis
    ps_coos += space.cen

    # get the i0 priors for the point sources from the reconstruction
    log_sum = np.log(np.sum(rec_sub[None] * ps_masks, axis=(1,2), where=(ps_masks > 0)))
    offset = [round(float(ri), 1) for ri in log_sum]

    ps_dct = {
        'space': space.to_dict(),
        'coordinates': ps_coos.tolist(),
        'i0': {
            'base': 'i0_ps',
        },
        'offset': offset,
        'n_copies': len(offset),
    }
    return ps_dct



def model_objects(
        oj_mask,
        space,
        rec_sub,
        gaussian = None,
        zero_pad = None,
):
    pix = np.argwhere(oj_mask > 0)
    lim = np.array([pix.min(axis=0) - 1, pix.max(axis=0) + 1])
    lim = lim.clip(0, space.shp-1)
    shp = 1 + lim[1] - lim[0]
    shp[shp%2 != 0] += 1
    cen = lim.mean(axis=0)
    cen[cen%1 == 0] += 0.5
    cen = space.cen + space.dis * (cen - 0.5 * (space.shp - 1))
    oj_space = SignalSpace.build(shape=shp, distances=space.distances, center=cen)

    log_mean = np.log(np.mean(rec_sub * oj_mask, where=(oj_mask > 0)))
    offset = round(float(log_mean), 1)

    oj_dct = {
        'space': oj_space.to_dict(),
        'i0': {
            'base': 'i0_os',
        },
        'offset': offset,
    }
    if gaussian:
        g_mean, g_std = gaussian['mean_fac'], gaussian['std_fac']
        fov_x, fov_y = oj_space.fov
        oj_dct['gaussian'] = {
            'cov_x': [float(g_mean * fov_x), float(g_std * fov_x)],
            'cov_y': [float(g_mean * fov_y), float(g_std * fov_y)],
        }
    if zero_pad:
        oj_dct['zero_pad'] = zero_pad

    return oj_dct



def model_tiles(
        ts_masks,
        space,
        rec_sub,
        tile_size = 32,
        gaussian = None,
):
    tile_size = to_shape(tile_size, (2,), 'int64')

    ts_cen = []
    for tm in ts_masks:
        pix = np.argwhere(tm > 0)
        lim = np.array([pix.min(axis=0) - 1, pix.max(axis=0) + 1])
        lim = lim.clip(0, space.shp-1)
        cen = lim.mean(axis=0)
        cen[cen%1 == 0] += 0.5
        cen = space.cen + space.dis * (cen - 0.5 * (space.shp - 1))
        ts_cen.append(cen.tolist())

    tile_spaces = SignalSpace.build(shape=tile_size, distances=space.distances, center=ts_cen, n_copies=len(ts_cen))

    # ts_vals = rec_sub[None] * ts_masks
    # ts_sums = np.sum(ts_vals, axis=(1, 2), where=(ts_vals != 0))
    # ncounts = np.count_nonzero(ts_vals, axis=(1, 2))
    # offsets = np.log(np.divide(ts_sums, ncounts, out=np.zeros_like(ts_sums, dtype=float), where=(ncounts != 0)))
    # offsets = [round(float(os), 1) for os in offsets]

    log_mean = np.log(np.mean(rec_sub[None] * ts_masks, axis=(1,2), where=(ts_masks > 0)))
    offset = [round(float(ri), 1) for ri in log_mean]

    ts_dct = {
        'space': space.to_dict(),
        'tile_spaces': tile_spaces.to_dict(),
        'i0': {
            'base': 'i0_ts',
        },
        'offset': offset,
        'n_copies': len(ts_cen),
    }
    if gaussian:
        g_mean, g_std = gaussian['mean_fac'], gaussian['std_fac']
        fov_x, fov_y = tile_spaces.fov
        ts_dct['gaussian'] = {
            'cov_x': [float(g_mean * fov_x), float(g_std * fov_x)],
            'cov_y': [float(g_mean * fov_y), float(g_std * fov_y)],
        }

    return ts_dct



def draw_boxes(cfg_sections, space, it):
    box_map = np.zeros(space.shape)

    for k,v in cfg_sections.items():
        if 'sky_t' in k and f'.{it}' in k:
            ni = v['n_copies']
            si = SignalSpace.build(**v['tile_spaces'], n_copies=ni)
            xi = np.ones((ni,)+si.shape)
            xi[:, 1:-1, 1:-1] = 0
            box_map += map_signal(xi, si, space)
        elif 'sky_o' in k and f'.{it}' in k:
            si = SignalSpace.build(**v['space'])
            xi = np.ones(si.shape)
            xi[1:-1, 1:-1] = 0
            box_map += map_signal(xi, si, space)
    
    return box_map

import sys
import numpy as np
from aim_resolve import ImageData, SignalSpace, SetupKLConfig, yaml_load, map_signal, masks_from_maps, plot_arrays, adjust_zoom, draw_boxes, model_background, model_points, model_objects, model_tiles



def main():
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, opt_pkl, det_npz, base_yml, it = files
    
    # load the model yaml-file from the previous iteration to the SetupKLConfig class and add one iteration
    cfg = SetupKLConfig.from_file(mdl_yml)
    cfg.sections['opt']['resume'] = cfg.sections['opt']['odir']
    cfg.sections['opt']['odir'] = cfg.sections['opt']['odir'].replace(f'{cfg.it}_rec', f'{cfg.it+1}_rec')
    cfg.add_it(fix_keys=['data'], del_comp=True)

    # load the base yaml-file and get the base modelling settings, output directory and iteration number
    base_dct = yaml_load(base_yml)
    mdl_dct = base_dct.pop('base_model')
    plt_dct = base_dct['base_plot']
    odir = base_dct['base_opt']['odir']
    cfg.modify_sec(f'opt.{it}', base='base_opt.n')

    # update fast-resolve kernels depending on the new resolution
    if 'psf_pixels' in cfg.sections[f'lh.{it}']:
        rkdir = '_'.join(cfg.sections[f'lh.{it}']['response_kernel'].split('_')[:-1])
        nkdir = '_'.join(cfg.sections[f'lh.{it}']['noise_kernel'].split('_')[:-1])
        ksize = mdl_dct['zoom'] * base_dct['space_bg']['shape'][0]
        cfg.modify_sec(f'lh.{it}', response_kernel=f'{rkdir}_{ksize}.pkl', noise_kernel=f'{nkdir}_{ksize}.pkl')

    # load the reconstructed image and setup the model tile space
    rec = ImageData.load(opt_pkl, dtype='float32')
    bg_space = SignalSpace.build(**base_dct['space_bg'])

    # load the detected point sources and extended objects
    det_dct = dict(np.load(det_npz))
    ps_map = det_dct['ps_map'].astype(float)
    cl_map = det_dct['cl_map'].astype(float)

    # adjust the zoom level of the sky model and zoom the reconstructed image if necessary
    zoom = adjust_zoom(mdl_dct['zoom'], rec.space, bg_space)
    if zoom > 1:
        rec.val = map_signal(rec.val, rec.space, zoom * rec.space)
        ps_map = map_signal(ps_map, rec.space, zoom * rec.space)
        cl_map = map_signal(cl_map, rec.space, zoom * rec.space, vmap_sum=False)

    # create a mask for the detected point sources and each extended object
    if mdl_dct['tiles'] and mdl_dct['tiles']['tile_size']:
        mdl_dct['masks'] |= {'tile_size': mdl_dct['tiles']['tile_size']}
    mask_dct = masks_from_maps(ps_map, cl_map, it, mdl_dct['zoom'], **mdl_dct['masks'])

    bg_dct = model_background(mask_dct[f'bg.{it}'], rec.val)
    cfg.modify_sec(f'sky_bg.{it}', merge_base=True, **bg_dct)
    rec_sub = rec.val - np.exp(bg_dct['i0']['offset_mean'])
    rec_sub = rec_sub.clip(0, None)

    if mdl_dct['points']:
        for pi in filter(lambda x: 'p' in x, mask_dct):
            pi_dct = model_points(mask_dct[pi], ps_map, zoom * rec.space, rec_sub)
            if pi_dct:
                cfg.add_sec(f'sky_{pi}', prefix=pi, **pi_dct)
                cfg.modify_sec(f'sky.{it}', **{pi: f'=sky_{pi}'})

    if mdl_dct['objects']:
        for oi in filter(lambda x: 'o' in x, mask_dct):
            oi_dct = model_objects(mask_dct[oi], zoom * rec.space, rec_sub, **mdl_dct['objects'])
            if oi_dct:
                cfg.add_sec(f'sky_{oi}', prefix=oi, **oi_dct)
                cfg.modify_sec(f'sky.{it}', **{oi: f'=sky_{oi}'})

    if mdl_dct['tiles']:
        for ti in filter(lambda x: 't' in x, mask_dct):
            ti_dct = model_tiles(mask_dct[ti], zoom * rec.space, rec_sub, **mdl_dct['tiles'])
            if ti_dct:
                cfg.add_sec(f'sky_{ti}', prefix=ti, **ti_dct)
                cfg.modify_sec(f'sky.{it}', **{ti: f'=sky_{ti}'})

    # get the positions of the detected point sources and the boxes of the extended objects and plot them as markers
    px, py = np.argwhere(ps_map == 1).T
    ps_mrk = dict(x=px, y=py, s=25, c='white', marker='+')
    box_map = draw_boxes(cfg.sections, zoom * rec.space, it)
    ox, oy = np.argwhere(box_map == 1).T
    oj_mrk = dict(x=ox, y=oy, s=1, c='white', marker=',')
    plot_arrays(
        array = rec.val, 
        space = rec.space, 
        label = 'detected components',
        name = f'{it}_mdl.png',
        odir = f'{odir}/plots',
        marker = (ps_mrk, oj_mrk),
        **plt_dct,
    )

    # plot and save the point source and object masks
    p_dct = plt_dct | {'norm': 'linear', 'vmin': 0, 'vmax': 1}
    plot_arrays(
        array = [np.sum(v, axis=0) if v.ndim == 3 else v for v in mask_dct.values()],
        label = [f'mask {k}' for k in mask_dct],
        name = f'{it}_msk.png',
        odir = f'{odir}/plots',
        **p_dct,
    )
    np.savez(f'{odir}/files/{it}_msk', **mask_dct)

    # save the new model yaml-file
    cfg.to_file(f'{odir}/files/{it}_mdl.yml')
    

if __name__ == '__main__':
    main()

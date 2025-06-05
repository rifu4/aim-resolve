import os
import sys
import numpy as np
from aim_resolve import ImageData, yaml_load, plot_arrays, plot_classes, clustering, unet_predict



def main():
    _, files = sys.argv[0], sys.argv[1:]
    opt_pkl, unet_pth, base_yml, it = files
    
    # load model and base yaml-files and extract output directory
    base_dct = yaml_load(base_yml)
    plt_dct = base_dct['base_plot']
    odir = base_dct['base_opt']['odir']

    # load the reconstructed image and space
    rec = ImageData.load(opt_pkl, dtype='float32')

    # detect point sources and objects in the reconstructed image using the U-Net model
    ps_map, oj_map = unet_predict(np.log(rec.val), unet_pth)
    # ps_map, oj_map = brightest_pixels(rec_val, fac=5)

    # load the clustering settings and cluster the detected objects
    cl_dct = base_dct['base_clu']
    cl_alg = cl_dct.pop('alg')
    cl_map = clustering(oj_map, cl_alg, **cl_dct)
    
    # sort the cluster maps by the sizes of the objects in descending order
    ones_count = np.sum(cl_map, axis=(1, 2))
    sorted_indices = np.argsort(-ones_count)
    cl_map = cl_map[sorted_indices]

    # plot the detected point sources and clustered objects
    plot_classes(
        points_map = ps_map,
        object_maps = cl_map,
        space = rec.space, 
        label = 'points & objects', 
        name = f'{it}_det.png',
        odir = f'{odir}/plots',
        **plt_dct,
    )

    # save the detected point sources and clustered objects
    np.savez(f'{odir}/files/{it}_det', ps_map=ps_map, cl_map=cl_map)

    # extra plots: single maps of point sources and objects (before and after clustering)
    if os.path.isdir(odir + '/extra/'):
        plot_classes(
            object_maps= cl_map,
            space = rec.space, 
            label = 'objects', 
            name = f'{it}_map_cl.png',
            odir = f'{odir}/extra',
            **plt_dct,
        )
        plt_dct |= {'norm': 'linear', 'vmin': 0, 'vmax': 1}
        for val,lbl,nms in zip([ps_map, oj_map], ['points', 'objects'], ['map_ps', 'map_oj']):
            plot_arrays(
                array = val,
                space = rec.space, 
                label = lbl, 
                name = f'{it}_{nms}.png',
                odir = f'{odir}/extra',
                **plt_dct,
            )


if __name__ == '__main__':
    main()

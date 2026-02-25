"""Script for detecting point sources and extended objects in reconstructed images."""
import os
import sys
import numpy as np
from aim_resolve import ImageData, yaml_load, model_predict, dbscan_clustering, objects2points, plot_arrays, plot_classes



def main():
    """Run source detection on a reconstructed image using a U-Net model and DBSCAN clustering."""
    _, files = sys.argv[0], sys.argv[1:]
    opt_pkl, base_yml, it = files
    
    # load model and base yaml-files and extract output directory
    base_dct = yaml_load(base_yml)
    plt_dct = base_dct['base_plot']
    odir = base_dct['base_opt']['odir']

    # load the reconstructed image and grid
    rec = ImageData.load(opt_pkl, dtype='float32')

    # detect point sources and objects in the reconstructed image using the U-Net model
    seg_dct = base_dct['base_seg']
    ps_map, oj_map = model_predict(rec, **seg_dct, print_ps=False)

    # load the clustering settings and cluster the detected objects
    cl_dct = base_dct['base_clu']
    cl_alg = cl_dct.pop('alg')
    cl_map, noise_map = dbscan_clustering(oj_map, print_cl=True, **cl_dct)

    # convert one-pixel objects to point sources
    ps_map = objects2points(ps_map, noise_map, print_ps=True, **cl_dct)

    # plot the detected point sources and clustered objects
    plot_classes(
        points_map = ps_map,
        object_maps = cl_map,
        grid = rec.grid, 
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
            grid = rec.grid, 
            label = 'objects', 
            name = f'{it}_map_cl.png',
            odir = f'{odir}/extra',
            **plt_dct,
        )
        plt_dct |= {'norm': 'linear', 'vmin': 0, 'vmax': 1}
        for val,lbl,nms in zip([ps_map, oj_map], ['points', 'objects'], ['map_ps', 'map_oj']):
            plot_arrays(
                array = val,
                grid = rec.grid, 
                label = lbl, 
                name = f'{it}_{nms}.png',
                odir = f'{odir}/extra',
                **plt_dct,
            )


if __name__ == '__main__':
    main()

"""Clustering utilities for source detection in U-Net output maps."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



def dbscan_clustering(objects_map, print_cl=True, **cl_kwargs):
    """Cluster extended objects in an output map using DBSCAN.

    Parameters
    ----------
    objects_map : np.ndarray
        Binary objects map from the U-Net.
    print_cl : bool, optional
        Whether to print the number of detected objects and noise points.
        Default is True.
    **cl_kwargs
        Keyword arguments forwarded to ``sklearn.cluster.DBSCAN``.

    Returns
    -------
    cluster_maps : np.ndarray
        Array of shape ``(n_objects, *objects_map.shape)`` with one binary
        map per detected object, sorted by descending pixel count.
    noise_map : np.ndarray
        Binary map of the same shape as *objects_map* containing the
        noise points.
    """
    # extract locations of the extended objects from the output map
    X = np.argwhere(objects_map == 1)

    # check if there are any objects to cluster, if not return empty array
    if X.size == 0:
        if print_cl:
            print('n objects:', 0)
            print('n noise points:', 0)
        return np.zeros((0,) + objects_map.shape)
    
    # initialize clustering method
    cl_alg = DBSCAN(**cl_kwargs)
    
    # scale input and apply selected clustering method
    X_scaled = StandardScaler().fit_transform(X)
    clu = cl_alg.fit(X_scaled)
    labels = clu.labels_

    # get number of detected objects and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # print number of detected objects and noise points
    if print_cl:
        print(f'n objects: {n_clusters}')
        print('n noise points: %d' % n_noise)

    # create one output map for each detected object and an empty map for the background
    cluster_maps = np.zeros((n_clusters,) + objects_map.shape)
    for k in range(n_clusters):
        mask = labels == k
        loc = X[mask].T
        cluster_maps[k][loc[0], loc[1]] = 1

    # sort the cluster maps by the sizes of the objects in descending order
    ones_count = np.sum(cluster_maps, axis=(1, 2))
    sorted_indices = np.argsort(-ones_count)
    cluster_maps = cluster_maps[sorted_indices]

    # create a map for the noise points
    noise_map = np.zeros_like(objects_map)
    mask = labels == -1
    loc = X[mask].T
    noise_map[loc[0], loc[1]] = 1

    return cluster_maps, noise_map



def objects2points(points_map, noise_map, print_ps=True, **cl_kwargs):
    """Convert single-pixel noise clusters to point sources.

    Re-clusters the noise map and adds clusters consisting of exactly one
    pixel to *points_map*.

    Parameters
    ----------
    points_map : np.ndarray
        Binary point-source map from the U-Net.
    noise_map : np.ndarray
        Binary noise map from ``dbscan_clustering``.
    print_ps : bool, optional
        Whether to print the total number of points. Default is True.
    **cl_kwargs
        Keyword arguments forwarded to ``dbscan_clustering``.

    Returns
    -------
    points_map : np.ndarray
        Updated point-source map with single-pixel noise clusters added.
    """
    if np.sum(noise_map) == 0:
        if print_ps:
            print('n points:', np.sum(points_map == 1))
        return points_map
    
    cl_kwargs.pop('min_samples', None)
    noise_maps, _ = dbscan_clustering(noise_map, min_samples=1, print_cl=False, **cl_kwargs)
    mask = np.sum(noise_maps == 1, axis=(1, 2)) == 1
    add_points = np.sum(noise_maps[mask], axis=0).astype(points_map.dtype)

    points_map += add_points

    if print_ps:
        print('n points:', np.sum(points_map == 1))

    return points_map.clip(0, 1)

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler



def clustering(oj_map, cl_alg='dbscan', print_cl=True, **cl_kwargs):
    '''
    function to cluster the extended objects in the output map of the U-Net.

    Parameters
    ----------
    oj_map : np.ndarray
        The output map of the U-Net.
    clu_method : str
        The clustering method to use. Default is 'dbscan'.
    print_clu : bool
        Whether to print the number of detected objects and noise points. Default is True.
    **clu_kwargs
        Necessary keyword arguments for the clustering method.

    Returns
    -------
    cs_maps : np.ndarray
        An array of output maps, one for each detected object and an empty map for the background.
    '''
    # extract locations of the extended objects from the output map
    X = np.argwhere(oj_map == 1)

    # check if there are any objects to cluster, if not return empty array
    if X.size == 0:
        if print_cl:
            print('n objects:', 0)
            print('n noise points:', 0)
        return np.zeros((0,) + oj_map.shape)
    
    # initialize clustering method
    if cl_alg.lower() == 'dbscan':
        cl_alg = DBSCAN(**cl_kwargs)
    elif cl_alg.lower() == 'kmeans':
        cl_alg = KMeans(**cl_kwargs)
    else:
        raise NotImplementedError('only DBSCAN is implemented so far')
    
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
    cs_maps = np.zeros((n_clusters,) + oj_map.shape)
    for k in range(n_clusters):
        mask = labels == k
        loc = X[mask].T
        cs_maps[k][loc[0], loc[1]] = 1

    return cs_maps

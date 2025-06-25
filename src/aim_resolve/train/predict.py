import numpy as np
from torch.utils.data import DataLoader

from .dataset import TensorDataset, transform_data, add_coordinates
from .model import SegmentationModel
from ..img_data.data import ImageData
from ..model.util import check_type



def model_predict(reconstruction, seg_model, transform, n_orders=None, coordinates=False, print_ps=True):
    '''
    Detect point sources and extended objects in a reconstructed image.

    Parameters
    ----------
    reconstruction : ImageData
        The input reconstruction.
    seg_model : dict
        Dictionary containing the model name and odir (see SegmentationModel.load).
    transform : dict
        Dictionary containing transformation parameters (see transform_data)
    n_orders : int or None, optional
        If provided the reconstruction is cut to `n_orders` orders of magnitude from below, by default None
    coordinates : bool, optional
        Whether to add coordinates to the data, by default True
    print_ps : bool, optional
        Whether to print the number of detected point sources, by default True
    
    Returns
    -------
    points_map : np.ndarray
        The detected point sources.
    object_maps : np.ndarray
        The detected extended objects.
    '''
    check_type(reconstruction, ImageData)
    check_type(n_orders, (int, type(None)))

    seg_model = SegmentationModel.load(**seg_model)

    rec_val = np.expand_dims(reconstruction.val, axis=(0,1))

    min_value = rec_val.max() / (10**n_orders) if n_orders else 0

    dataset = (rec_val, np.zeros_like(rec_val))
    dataset = transform_data(dataset, min_value=min_value, **transform)

    if coordinates:
        dataset = add_coordinates(dataset, reconstruction.space)

    dataset = TensorDataset(dataset)
    
    rec_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sample = next(iter(rec_loader))
    pred = seg_model.sigmoid_predict(sample['x'])
    pred = pred.detach().numpy()

    if print_ps:
        print('n points:', np.sum(pred[0] == 1))

    return pred[0], pred[1]



def brightest_pixels(reconstruction, transform, n_orders=None, cutoff=0.5, print_ps=True, **kwargs):
    check_type(reconstruction, ImageData)
    check_type(n_orders, (int, type(None)))

    rec_val = np.expand_dims(reconstruction.val, axis=(0,1))

    if n_orders:
        print('recon min/max:', rec_val.min(), rec_val.max())
        print('orders cutoff min:', rec_val.max() / (10**n_orders))
        rec_val = rec_val.clip(rec_val.max()/(10**n_orders), None)

    dataset = (rec_val, np.zeros_like(rec_val))
    dataset = transform_data(dataset, **transform)

    ps_map = np.zeros_like(rec_val)
    oj_map = np.where(dataset[0] > cutoff, 1, 0)

    if print_ps:
        print('n points:', np.sum(ps_map[0] == 1))

    return ps_map[0,0], oj_map[0,0]

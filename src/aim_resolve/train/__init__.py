"""Training subpackage for neural network source detection models.

Requires the ``train`` extra dependencies::

    pip install aim-resolve[train]
"""

try:
    from .dataset import Dataset
    from .model import SegmentationModel
    from .predict import brightest_pixels, model_predict
except ImportError as exc:
    raise ImportError(
        "The aim_resolve.train subpackage requires additional dependencies. "
        "Install them with:  pip install aim-resolve[train]"
    ) from exc

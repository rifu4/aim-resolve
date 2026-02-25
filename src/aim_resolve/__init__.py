"""AIM-Resolve: Astronomical Image Modeling and Reconstruction.

Provides tools for Bayesian image reconstruction of astronomical data,
including sky modeling, likelihood evaluation, optimization, and
transition utilities for multi-resolution and multi-frequency workflows.
"""

from . import enforce_float64

# --- fast_resolve ---
from .fast_resolve import (
    PSFConvolve, PSFSplitConvolve, NInvConvolve,
    build_psf_kernel, build_n_inv_kernel,
    fast_optimize_kl,
    build_exact_responses,
)

# --- img_data ---
from .img_data import (
    BackgroundGenerator,
    ComponentGenerator,
    ImageData, ImageDataGenerator,
    gaussian_filter2d, rotate_data, flip_data,
    ObjectGenerator,
    PointGenerator,
    TileGenerator,
)

# --- model ---
from .model import (
    ComponentModel,
    gaussian_model,
    IntegerPrior, integer_model,
    map_signal,
    NoiseModel,
    normal_model,
    PointModel,
    prior_model, correlated_field_model, inverse_gamma_model, uniform_model,
    SignalModel,
    SignalGrid, PointGrid,
    TileModel,
    check_type,
)

# --- optimize ---
from .optimize import (
    OptimizeKLConfig,
    optimize_kl,
    MyOptimizeVI,
    MySamples, get_samples, domain_tree, domain_keys, model_init, random_init,
    SetupKLConfig,
    merge_dicts,
    yaml_load, yaml_save,
)

# --- plot ---
from .plot import (
    plot_arrays,
    plot_classes,
    plot_mean_and_std, plot_samples, plot_agreement, plot_pullplot,
    plot_image,
    plot_power,
    plot_models,
)

# --- resolve ---
from .resolve import (
    str2rad,
    SignalResponse, PointResponse, TileResponse, ComponentResponse,
    Observation,
    point_response, signal_response, ducc_response, finu_response, rotate,
)

# --- train ---
from .train import (
    Dataset,
    SegmentationModel,
    model_predict, brightest_pixels,
)

# --- top-level modules ---
from .builders import get_builders
from .clustering import dbscan_clustering, objects2points
from .data import data_func, image_data, radio_data
from .extension import extension_func, freq_extension, zoom_extension
from .likelihood import (
    likelihood_func, image_likelihood, radio_likelihood,
    fast_likelihood, likelihood_sum,
)
from .mask import (
    masks_from_maps, masks_from_model, masks_to_boxes,
    add_freq_axis, remove_freq_axis,
)
from .modeling import (
    draw_boxes, model_background, model_points,
    model_objects, model_tiles,
)
from .transition import transition_func, transition_anew, transition_addt

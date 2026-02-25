"""AIM-Resolve: Astronomical Image Modeling and Reconstruction.

Provides tools for Bayesian image reconstruction of astronomical data,
including sky modeling, likelihood evaluation, optimization, and
transition utilities for multi-resolution and multi-frequency workflows.
"""

from . import enforce_float64

# --- top-level modules ---
from .builders import get_builders
from .clustering import dbscan_clustering, objects2points
from .data import data_func, image_data, radio_data
from .extension import extension_func, freq_extension, zoom_extension

# --- fast_resolve ---
from .fast_resolve import (
    NInvConvolve,
    PSFConvolve,
    PSFSplitConvolve,
    build_exact_responses,
    build_n_inv_kernel,
    build_psf_kernel,
    fast_optimize_kl,
)

# --- img_data ---
from .img_data import (
    BackgroundGenerator,
    ComponentGenerator,
    ImageData,
    ImageDataGenerator,
    ObjectGenerator,
    PointGenerator,
    TileGenerator,
    flip_data,
    gaussian_filter2d,
    rotate_data,
)
from .likelihood import (
    fast_likelihood,
    image_likelihood,
    likelihood_func,
    likelihood_sum,
    radio_likelihood,
)
from .mask import (
    add_freq_axis,
    masks_from_maps,
    masks_from_model,
    masks_to_boxes,
    remove_freq_axis,
)

# --- model ---
from .model import (
    ComponentModel,
    IntegerPrior,
    NoiseModel,
    PointGrid,
    PointModel,
    SignalGrid,
    SignalModel,
    TileModel,
    check_type,
    correlated_field_model,
    gaussian_model,
    integer_model,
    inverse_gamma_model,
    map_signal,
    normal_model,
    prior_model,
    uniform_model,
)
from .modeling import (
    draw_boxes,
    model_background,
    model_objects,
    model_points,
    model_tiles,
)

# --- optimize ---
from .optimize import (
    MyOptimizeVI,
    MySamples,
    OptimizeKLConfig,
    SetupKLConfig,
    domain_keys,
    domain_tree,
    get_samples,
    merge_dicts,
    model_init,
    optimize_kl,
    random_init,
    yaml_load,
    yaml_save,
)

# --- plot ---
from .plot import (
    plot_agreement,
    plot_arrays,
    plot_classes,
    plot_image,
    plot_mean_and_std,
    plot_models,
    plot_power,
    plot_pullplot,
    plot_samples,
)

# --- resolve ---
from .resolve import (
    ComponentResponse,
    Observation,
    PointResponse,
    SignalResponse,
    TileResponse,
    ducc_response,
    finu_response,
    point_response,
    rotate,
    signal_response,
    str2rad,
)

# --- train ---
from .train import (
    Dataset,
    SegmentationModel,
    brightest_pixels,
    model_predict,
)
from .transition import transition_addt, transition_anew, transition_func

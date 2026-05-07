"""Multy-frequency subpackage containing function from J-UBIK."""

from .spectral_product_mf_sky import build_simple_spectral_sky
from .spectral_product_utils.frequency_deviations import (
    build_frequency_deviations_model_with_degeneracies,
)

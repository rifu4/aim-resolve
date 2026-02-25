"""Enable 64-bit floating-point precision in JAX.

This module is imported at package initialization to ensure that all
JAX computations use float64 precision by default.
"""

from jax import config
config.update('jax_enable_x64', True)

"""Shared pytest fixtures for aim-resolve tests.

Fixtures defined here are available to every test module without any import.
Only truly cross-cutting objects live here; file-specific helpers stay local.
"""

import jax
import numpy as np
import pytest

from aim_resolve.model.grid import SignalGrid
from aim_resolve.model.signal import SignalModel

# ---------------------------------------------------------------------------
# JAX helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def jax_key():
    """A canonical JAX PRNG key with seed 0.

    Use wherever a deterministic random key is needed without caring about the
    specific value.  For tests that require independence between samples, call
    ``jax.random.split(jax_key, n)`` inside the test body.
    """
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Grid fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_grid():
    """A small 8×8 SignalGrid with default distances (1/8 arcsec per pixel)."""
    return SignalGrid.build(space=(8, 8))


@pytest.fixture
def medium_grid():
    """A medium 16×16 SignalGrid with default distances (1/16 arcsec per pixel)."""
    return SignalGrid.build(space=(16, 16))


@pytest.fixture
def large_grid():
    """A larger 32×32 SignalGrid with default distances (1/32 arcsec per pixel)."""
    return SignalGrid.build(space=(32, 32))


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def background_signal(medium_grid):
    """A minimal background :class:`~aim_resolve.model.signal.SignalModel` (16×16).

    Suitable for tests that need a concrete SignalModel instance without caring
    about the specific prior parameters.
    """
    return SignalModel.build(
        grid=dict(space=(16, 16)),
        params=dict(i0=dict(mean=0.0, std=1.0)),
        prefix="bg",
    )


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """A seeded NumPy default RNG.  Produces reproducible random arrays."""
    return np.random.default_rng(0)

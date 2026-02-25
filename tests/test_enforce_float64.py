"""Tests for aim_resolve.enforce_float64."""

import jax


def test_float64_enabled():
    """JAX 64-bit precision should be enabled after importing aim_resolve."""
    import aim_resolve.enforce_float64  # noqa: F401

    assert jax.config.jax_enable_x64 is True
